import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F

from collections import defaultdict
from datasets import load_dataset
from hdbscan import HDBSCAN, validity_index
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.nn import Identity
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

"""
Run various experiments on transformer models.

We use the following shorthand notation in comments:

N: number of tokens
d: dimensionality of token space
d': dimensionality of query/key space
L: number of layers
S: sample size
"""


class FirstKDims(torch.nn.Module):
    def __init__(self, module, k):
        super().__init__()
        self.module = module
        self.k = k

    def forward(self, x):
        return self.module(x)[..., :self.k]


def seed_everything(seed):
    """
    credit goes to https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def stack_padded(ndarray_list, pad_value=0):
    assert ndarray_list  # not empty
    target_shape = [len(ndarray_list),
                    *[max(a.shape[dim] for a in ndarray_list) for dim in range(ndarray_list[0].ndim)]]
    stacked_array = np.full(target_shape, fill_value=pad_value, dtype=ndarray_list[0].dtype)
    for i, a in enumerate(ndarray_list):
        stacked_array[i, *[slice(0, a.shape[dim]) for dim in range(a.ndim)]] = a
    return stacked_array


def build_model(model_id, random_params, no_dense_layers, num_hidden_layers):
    config = AutoConfig.from_pretrained(model_id)
    if num_hidden_layers:  # override depth (only works for models with shared params)
        config.num_hidden_layers = num_hidden_layers
    if random_params:  # initialise params randomly
        model = AutoModel.from_config(config)
    else:
        model = AutoModel.from_pretrained(model_id, config=config)
    if no_dense_layers:  # set dense layer params to zero
        disable_dense_layers(model)
    return model


def disable_dense_layers(model):
    model_type = model.config.model_type
    match model_type:
        case "albert":
            with torch.no_grad():
                model.encoder.albert_layer_groups[0].albert_layers[0].attention.dense.weight.fill_(0.0)
                model.encoder.albert_layer_groups[0].albert_layers[0].attention.dense.bias.fill_(0.0)
        case "bert":
            for i in range(len(model.encoder.layer)):
                with torch.no_grad():
                    model.encoder.layer[i].attention.output.dense.weight.fill_(0.0)
                    model.encoder.layer[i].attention.output.dense.bias.fill_(0.0)


def make_outdir(model, dataset, random_params, no_dense_layers):
    run_id = "randomtext" if dataset is None else dataset.info.dataset_name
    run_id += f"_{model.config._name_or_path}"
    run_id += f"_params-{'random' if random_params else 'trained'}"
    run_id += f"_dense-{'off' if no_dense_layers else 'on'}"
    run_id += f"_layers-{model.config.num_hidden_layers}"
    outdir = f"rawsults/{run_id}"
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_random_input(dataset, tokeniser):
    MIN_NUM_TOKENS = 100
    while True:
        if dataset is not None:
            index = torch.randint(low=0, high=len(dataset), size=(1,)).item()
            original_text = dataset[index]["text"]
        else:
            original_text = tokeniser.decode(torch.randint(tokeniser.vocab_size, (tokeniser.model_max_length,)))
        input = tokeniser(original_text, return_tensors="pt", truncation=True, return_attention_mask=False)
        if input.input_ids.shape[1] >= MIN_NUM_TOKENS:
            break
    tokenised_text = "(" + ")(".join(tokeniser.batch_decode(input.input_ids.squeeze())) + ")"
    return input, original_text, tokenised_text


def compute_similarity_matrices(hidden_states, queries, keys):
    sim_matrices = []
    for X, query, key in zip(hidden_states, queries, keys):
        X = torch.from_numpy(X)  # shape: N x d
        Q_unit = F.normalize(query(X), dim=-1)  # shape: N x d'
        K_unit = F.normalize(key(X), dim=-1)  # shape: N x d'
        sim_matrix = torch.matmul(Q_unit, K_unit.transpose(0, 1))  # shape: N x N
        sim_matrices.append(sim_matrix.numpy(force=True))
    return sim_matrices


def extract_queries_and_keys(model):
    num_layers = model.config.num_hidden_layers
    head_ndim = model.config.hidden_size // model.config.num_attention_heads
    model_type = model.config.model_type
    match model_type:
        case "albert":
            Q = FirstKDims(model.encoder.albert_layer_groups[0].albert_layers[0].attention.query, head_ndim)
            K = FirstKDims(model.encoder.albert_layer_groups[0].albert_layers[0].attention.key, head_ndim)
            return num_layers * [Q], num_layers * [K]
        case "bert":
            queries = [FirstKDims(l.attention.self.query, head_ndim) for l in model.encoder.layer]
            keys = [FirstKDims(l.attention.self.key, head_ndim) for l in model.encoder.layer]
            return queries, keys
        case _:
            raise NotImplementedError("Unsupported model type", model_type)


def compute_clustering(data, min_cluster_size=4):
    metrics = []
    labels = []
    for i, X in enumerate(data):
        X = StandardScaler().fit_transform(X)  # standardise to prevent numerical issues
        pca = PCA(n_components=64, random_state=42).fit(X)
        clustering = HDBSCAN(min_cluster_size).fit(pca.transform(X))
        labels.append(clustering.labels_)
        try:
            num_clusters = clustering.labels_.max() + 1
            outlier_rate = (clustering.labels_ == -1).sum() / len(clustering.labels_)
            silh_score = silhouette_score(X, clustering.labels_)
            dbcv_score = validity_index(X, clustering.labels_)
        except ValueError:  # single-cluster case (rarely happens)
            silh_score = dbcv_score = 1.0  # best possible score
        metrics.append(np.array([num_clusters, outlier_rate, silh_score, dbcv_score]))
    return metrics, labels


def compute_tsne_embeddings(data):
    return [TSNE(n_components=2, perplexity=5).fit_transform(X) for X in data]


def run_experiment(dataset, model_id, random_params=False, no_dense_layers=False,
                   num_hidden_layers=None, sample_size=10, num_bins=100):
    """
    This function produces artifact in the form of .npy files on the disk with the following shapes:

    "similarity_matrices_((XXᵀ)|(XQᵀKXᵀ)).npy": S x L x N x N
    "cluster_metrics_(X|XQᵀKXᵀ).npy": S x L x 4
    "cluster_labels_(X|XQᵀKXᵀ).npy": S x L x N
    "t-SNE_embeddings_(X|XQᵀKXᵀ).npy": S x L x N x 2
    """
    print("EXPERIMENT:", locals())
    seed_everything(42)  # ensure reproducibility

    model = build_model(model_id, random_params, no_dense_layers, num_hidden_layers)
    tokeniser = AutoTokenizer.from_pretrained(model_id)
    outdir = make_outdir(model, dataset, random_params, no_dense_layers)

    overview_df = pd.DataFrame(columns=["original_text", "tokenised_text", "num_tokens"])
    results = defaultdict(list)  # non-existent keys are initialised with empty list

    for sample_idx in tqdm(range(sample_size), desc="Analysing each sample"):
        # input from dataset
        input, original_text, tokenised_text = get_random_input(dataset, tokeniser)
        overview_df.loc[sample_idx] = (original_text, tokenised_text, input.input_ids.numel())

        # output from model
        output = model(**input, output_hidden_states=True)
        hidden_states = [X.squeeze(0).numpy(force=True)  # shape: N x d
                         for X in output.hidden_states]

        # similarity matrices
        identities = model.config.num_hidden_layers * [Identity()]  # setting Q = I, K = I
        sim_matrices_XXt = compute_similarity_matrices(hidden_states[1:], identities, identities)
        results["similarity_matrices_XXᵀ"].append(np.stack(sim_matrices_XXt))  # shape: L x N x N
        queries, keys = extract_queries_and_keys(model)  # using the model's actual Q, K
        sim_matrices_XQtKXt = compute_similarity_matrices(hidden_states[:-1], queries, keys)
        results["similarity_matrices_XQᵀKXᵀ"].append(np.stack(sim_matrices_XQtKXt))  # shape: L x N x N

        # clustering and t-SNE
        for target in ["X", "XQᵀKXᵀ"]:
            data = hidden_states if target == "X" else sim_matrices_XQtKXt
            metrics, labels = compute_clustering(data)
            tsnes = compute_tsne_embeddings(data)
            results[f"cluster_metrics_{target}"].append(np.stack(metrics))  # shape: L x 4
            results[f"cluster_labels_{target}"].append(np.stack(labels))  # shape: L x N
            results[f"t-SNE_embeddings_{target}"].append(np.stack(tsnes))  # L x N x 2

    overview_df.to_csv(f"{outdir}/0verview.csv", index_label="sample_idx")
    for key, val in results.items():
        if val:  # ignore empty lists
            np.save(f"{outdir}/{key}.npy", stack_padded(val))  # shape: S x previous


if __name__ == "__main__":
    wikitext = load_dataset("wikitext", "wikitext-103-v1", split="all")
    # imdb = load_dataset("stanfordnlp/imdb", split="all")

    run_experiment(wikitext, "albert-large-v2", sample_size=3)

    # for dataset in [None, wikitext, imdb]:  # None corresponds to random tokens
    #     for model_id in ["albert-large-v2", "bert-large-uncased"]:
    #         for random_params in [False, True]:
    #             for no_dense_layers in [False, True]:
    #                 try:
    #                     run_experiment(dataset,
    #                                    model_id,
    #                                    random_params,
    #                                    no_dense_layers,
    #                                    num_hidden_layers=72 if model_id == "albert-large-v2" else None)
    #                 except Exception as e:
    #                     print(">>> FAILED: ", dataset, model_id, random_params, no_dense_layers)
    #                     print(repr(e))
