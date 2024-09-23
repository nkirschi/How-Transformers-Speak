import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import load_dataset
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from torch.nn import Identity
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


def stack_padded(ndarray_list, pad_value=0):
    assert ndarray_list  # not empty
    target_shape = [len(ndarray_list),
                    *[max(a.shape[dim] for a in ndarray_list) for dim in range(ndarray_list[0].ndim)]]
    stacked_array = np.full(target_shape, fill_value=pad_value, dtype=ndarray_list[0].dtype)
    for i, a in enumerate(ndarray_list):
        stacked_array[i, *[slice(0, a.shape[dim]) for dim in range(a.ndim)]] = a
    return stacked_array


def build_model(model_id, random_params, no_dense_layers, num_hidden_layers):
    config = AutoConfig.from_pretrained(model_id, num_attention_heads=1)
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


def make_outdir(model, dataset, use_queries_and_keys, random_params, no_dense_layers):
    runid = f"{model.config._name_or_path}"
    runid += "_uniform-tokens" if dataset is None else "_" + dataset.info.dataset_name
    runid += "_⟨Qx,Ky⟩" if use_queries_and_keys else "_⟨x,y⟩"
    runid += "_random-params" if random_params else "_trained-params"
    runid += "_dense-off" if no_dense_layers else "_dense-on"
    runid += f"_{model.config.num_hidden_layers}-layers"
    outdir = f"rawsults/{runid}"
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


def compute_correlations(hidden_states, queries, keys):
    corrs_list = []
    for X, query, key in zip(hidden_states, queries, keys):
        Q_unit = F.normalize(query(X), dim=1)  # shape: N x d'
        K_unit = F.normalize(key(X), dim=1)  # shape: N x d'
        similarities = torch.matmul(Q_unit, K_unit.transpose(0, 1))  # shape: N x N
        assert (similarities.abs() < 1 + 1e-5).all()
        corrs_list.append(similarities.detach())
    return corrs_list


def extract_queries_and_keys(model, replace_with_identity=False):
    num_layers = model.config.num_hidden_layers
    if replace_with_identity:
        return num_layers * [Identity()], num_layers * [Identity()]
    model_type = model.config.model_type
    match model_type:
        case "albert":
            Q = model.encoder.albert_layer_groups[0].albert_layers[0].attention.query
            K = model.encoder.albert_layer_groups[0].albert_layers[0].attention.key
            return num_layers * [Q], num_layers * [K]
        case "bert":
            queries = [l.attention.self.query for l in model.encoder.layer]
            keys = [l.attention.self.key for l in model.encoder.layer]
            return queries, keys
        case _:
            raise NotImplementedError("Unsupported model type", model_type)


def compute_clustering(hidden_states):
    clustereval_arrays = []
    label_arrays = []
    for X in hidden_states:
        # median_dist = np.median(pairwise_distances(layer_latent, metric=metric))
        # eps = median_dist * (3/4 if metric == "euclidean" else 2/3 if metric == "cosine" else None)
        # clustering = DBSCAN(eps=eps, min_samples=2, metric=metric).fit(layer_latent)
        X = X.numpy()
        clusterer = HDBSCAN(min_cluster_size=4)
        clustering = clusterer.fit(X)
        label_arrays.append(clustering.labels_)

        num_clusters = clustering.labels_.max() + 1
        outlier_rate = (clustering.labels_ == -1).sum() / len(clustering.labels_)
        try:
            sil_score = silhouette_score(X, clustering.labels_)
            cal_score = calinski_harabasz_score(X, clustering.labels_)
            dav_score = davies_bouldin_score(X, clustering.labels_)
        except ValueError:  # single-cluster case
            sil_score = cal_score = dav_score = float("nan")
        clustereval_arrays.append(np.array([num_clusters, outlier_rate, sil_score, cal_score, dav_score]))
    return clustereval_arrays, label_arrays


def compute_tsne_embeddings(hidden_states):
    return [TSNE(n_components=2, perplexity=5).fit_transform(X) for X in hidden_states]


def run_experiment(dataset, model_id, use_queries_and_keys=False, random_params=False, no_dense_layers=False,
                   num_hidden_layers=None, sample_size=10, num_bins=100):
    print("EXPERIMENT:")
    print("\n".join(map(lambda x: f"{x[0]}: {x[1]}", locals().items())))

    model = build_model(model_id, random_params, no_dense_layers, num_hidden_layers)
    tokeniser = AutoTokenizer.from_pretrained(model_id)
    outdir = make_outdir(model, dataset, use_queries_and_keys, random_params, no_dense_layers)

    overview_df = pd.DataFrame(columns=["original_text", "tokenised_text", "num_tokens"])
    results = {
                  "histograms": [],
                  "sim_matrices": []
              } | {
                  f"{key}_{target}": []
                  for key in ["clustereval", "labels", "tsne"] for target in ["X", "XX.T"]
              }

    for sample_idx in tqdm(range(sample_size), desc="Analysing each sample"):
        input, original_text, tokenised_text = get_random_input(dataset, tokeniser)
        overview_df.loc[sample_idx] = (original_text, tokenised_text, input.input_ids.numel())

        output = model(**input, output_hidden_states=True)
        hidden_states = [X.squeeze(0).clone().detach().requires_grad_(False) for X in output.hidden_states[1:]]
        queries, keys = extract_queries_and_keys(model, replace_with_identity=not use_queries_and_keys)
        sim_list = compute_correlations(hidden_states, queries, keys)

        # histograms
        results["histograms"].append(np.stack([
            np.histogram(correls.flatten(), bins=num_bins, range=(-1, 1), density=True)[0]
            for correls in sim_list
        ]))

        # similarity matrices
        results["sim_matrices"].append(np.stack(sim_list).astype(np.float16))

        # clustering and T-SNE
        for target in ["X", "XX.T"]:
            data = hidden_states if target == "X" else sim_list
            clustereval_arrays, label_arrays = compute_clustering(data)
            tsne_arrays = compute_tsne_embeddings(data)
            results[f"clustereval_{target}"].append(np.stack(clustereval_arrays))
            results[f"labels_{target}"].append(np.stack(label_arrays))
            results[f"tsne_{target}"].append(np.stack(tsne_arrays))

    overview_df.to_csv(f"{outdir}/0verview.csv", index_label="sample_idx")
    for key, val in results.items():
        np.save(f"{outdir}/{key}.npy", stack_padded(val))


if __name__ == "__main__":
    wikitext = load_dataset("wikitext", "wikitext-103-v1", split="all")
    imdb = load_dataset("stanfordnlp/imdb", split="all")

    # for dataset in [None, wikitext, imdb]:
    #     for model_id in ["albert-xlarge-v2", "bert-large-uncased"]:
    #         for use_queries_and_keys in [False, True]:
    #             for random_params in [False, True]:
    #                 for no_dense_layers in [False, True]:
    #                     run_experiment(dataset, model_id, use_queries_and_keys, random_params, no_dense_layers)
    #
    # for dataset in [None, wikitext, imdb]:
    #     for use_queries_and_keys in [False, True]:
    #         run_experiment(dataset, "albert-xlarge-v2", use_queries_and_keys, num_hidden_layers=192)
    for dataset in [imdb]:
        for use_queries_and_keys in [False]:
            run_experiment(dataset, "albert-xlarge-v2", use_queries_and_keys, num_hidden_layers=192)
