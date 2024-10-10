import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp

from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

TARGET_TITLES = {
    "token_similarity": r"cos similarity $\langle \frac{x_i}{\lVert x_i \rVert}, \frac{x_j}{\lVert x_j \rVert} \rangle$",
    "token": r"tokens $x_i$",
    "attention_logits": r"attention logits $\frac{1}{\sqrt{d'}} \langle Qx_i, Kx_j \rangle$",
    "attention_logit": r"error functions $\mathcal{E}_i$"
}
TITLE_FN = lambda plot, target, run_id: f"{plot} of {TARGET_TITLES[target]}\n($\\texttt{{{run_id}}})$\n"


def plot_histograms(run_id, num_bins=100, conf_level=0.99):
    for target in ["token_similarity", "attention_logits"]:
        outdir = f"visualisation/{run_id}/{target}"
        os.makedirs(outdir, exist_ok=True)

        sim_tensor = np.load(f"rawsults/{run_id}/{target}.npy")
        seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]
        min, max = sim_tensor.min(), sim_tensor.max()
        count_tensor = np.stack([np.stack([
            np.histogram(layer[:seq_lens[i], :seq_lens[i]].flatten(),
                         bins=num_bins,
                         density=True,
                         range=(min, max))[0]
            for layer in sample]) for i, sample in enumerate(sim_tensor)
        ])
        count_mean = np.mean(count_tensor, axis=0)  # shape num_layer x num_bins
        t_score = sp.stats.t.ppf(q=1 - (1 - conf_level) / 2, df=count_tensor.shape[0] - 1)
        count_conf = t_score * np.std(count_tensor, axis=0) / np.sqrt(count_tensor.shape[0])  # t-distribution confidence deviation
        num_bins = count_tensor.shape[-1]
        max_density = count_tensor.max()

        num_layers = count_mean.shape[0]
        for page in range(num_layers // 24):
            layers_to_plot = range(page * 24, (page + 1) * 24)
            plt.figure(figsize=(12, 16))
            plt.suptitle(TITLE_FN("Histograms", target, run_id))
            for i, layer in enumerate(layers_to_plot):
                plt.subplot(6, 4, i + 1)
                plt.stairs(count_mean[layer], np.linspace(min, max, num_bins + 1))
                x = np.linspace(min + 1 / num_bins, max - 1 / num_bins, num_bins)
                plt.fill_between(x, count_mean[layer] - count_conf[layer], count_mean[layer] + count_conf[layer],
                                 alpha=0.5, step="mid")
                if target == "token_similarity":
                    plt.xlim(-1, 1)
                plt.ylim(0, max_density)  # set a consistent y-axis limit
                plt.title(f"layer {layer + 1}")
            plt.tight_layout()
            plt.savefig(f"{outdir}/histograms_layers{layers_to_plot[0]}-{layers_to_plot[-1]}.pdf")
            plt.close()


def plot_heatmaps(run_id):
    for target in ["token_similarity", "attention_logits"]:
        outdir = f"visualisation/{run_id}/{target}"
        os.makedirs(outdir, exist_ok=True)

        dotprod_tensor = np.load(f"rawsults/{run_id}/{target}.npy")
        seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]

        for sample in range(dotprod_tensor.shape[0]):
            num_layers = dotprod_tensor.shape[1]
            for page in range(num_layers // 24):
                layers_to_plot = range(page * 24, (page + 1) * 24)
                plt.figure(figsize=(12, 16))
                plt.suptitle(TITLE_FN("Heatmaps", target, run_id))
                for i, layer in enumerate(layers_to_plot):
                    dotprod_matrix = dotprod_tensor[sample, layer, :seq_lens[sample], :seq_lens[sample]]
                    plt.subplot(6, 4, i + 1)
                    if target == "token_similarity":
                        plt.imshow(dotprod_matrix, vmin=-1, vmax=1, cmap="coolwarm")
                    else:
                        plt.imshow(dotprod_matrix, cmap="viridis")
                    plt.colorbar()
                    plt.xlabel("token $j$")
                    plt.ylabel("token $i$")
                    plt.title(f"layer {layer + 1}")
                plt.tight_layout()
                plt.savefig(f"{outdir}/heatmaps_sample{sample}_layers{layers_to_plot[0]}-{layers_to_plot[-1]}.pdf")
                plt.close()


def plot_cluster_metrics(run_id):
    for target in ["token", "attention_logit"]:
        outdir = f"visualisation/{run_id}/{target}_clustering"
        os.makedirs(outdir, exist_ok=True)

        metrics_tensor = np.load(f"rawsults/{run_id}/{target}_cluster_metrics.npy")

        for sample in range(metrics_tensor.shape[0]):
            plt.figure(figsize=(6, 6))
            plt.suptitle(TITLE_FN("HDBSCAN cluster evaluation", target, run_id))
            for i, title in enumerate(["number of clusters", "outlier rate", "Silhouette score", "DBCV score"]):
                plt.subplot(2, 2, 1 + i)
                plt.title(title)
                plt.plot(metrics_tensor[sample, :, i])
                plt.xlabel("layer")
                if i == 1:
                    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
                    plt.ylim(0, 1)
                if i >= 2:
                    plt.ylim(-1, 1)
            plt.tight_layout()
            plt.savefig(f"{outdir}/cluster_metrics_sample{sample}.pdf")
            plt.close()


def plot_cluster_sizes(run_id):
    for target in ["token", "attention_logit"]:
        outdir = f"visualisation/{run_id}/{target}_clustering"
        os.makedirs(outdir, exist_ok=True)

        metrics_tensor = np.load(f"rawsults/{run_id}/{target}_cluster_metrics.npy")
        labels_tensor = np.load(f"rawsults/{run_id}/{target}_cluster_labels.npy")
        seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]

        for sample in range(metrics_tensor.shape[0]):
            num_layers = metrics_tensor.shape[1]
            for page in range(num_layers // 24):
                layers_to_plot = range(page * 24, (page + 1) * 24)
                plt.figure(figsize=(12, 16))
                plt.suptitle(TITLE_FN("HDBSCAN cluster sizes", target, run_id))
                max_num_clusters = max(max(labels_tensor[sample, layer, :seq_lens[sample]]) + 1
                                       for layer in layers_to_plot)
                for i, layer in enumerate(layers_to_plot):
                    labels = labels_tensor[sample, layer, :seq_lens[sample]]
                    cluster_sizes = np.bincount(labels[labels != -1])  # ignore outlier bin
                    colours = plt.colormaps["viridis"](np.linspace(0, 1, len(cluster_sizes) + 1)[:-1])
                    cluster_sizes = np.concatenate([cluster_sizes, np.zeros(max_num_clusters - len(cluster_sizes))])
                    cluster_sizes = np.sort(cluster_sizes)[::-1]  # sort by decreasing size
                    plt.subplot(6, 4, i + 1)
                    plt.bar(range(1, len(cluster_sizes) + 1), cluster_sizes, color=colours)
                    plt.title(f"layer {layer + 1}")
                    plt.xlabel("k-th largest cluster")
                    plt.xticks([1, len(cluster_sizes) + 1])
                    if i % 4 == 0:
                        plt.ylabel("cluster size")
                plt.tight_layout()
                plt.savefig(f"{outdir}/cluster_sizes_sample{sample}_layers{layers_to_plot[0]}-{layers_to_plot[-1]}.pdf")
                plt.close()

def plot_tsne(run_id):
    for target in ["token", "attention_logit"]:
        outdir = f"visualisation/{run_id}/{target}_clustering"
        os.makedirs(outdir, exist_ok=True)

        tsne_tensor = np.load(f"rawsults/{run_id}/{target}_t-SNE_embeddings.npy")
        labels_tensor = np.load(f"rawsults/{run_id}/{target}_cluster_labels.npy")
        seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]

        for sample in range(tsne_tensor.shape[0]):
            num_layers = tsne_tensor.shape[1]
            for page in range(num_layers // 24):
                layers_to_plot = range(page * 24, (page + 1) * 24)
                plt.figure(figsize=(12, 16))
                plt.suptitle(TITLE_FN("t-SNE embeddings", target, run_id))
                for i, layer in enumerate(layers_to_plot):
                    embeds = tsne_tensor[sample, layer, :seq_lens[sample]]
                    labels = labels_tensor[sample, layer, :seq_lens[sample]]
                    cluster_sizes = np.bincount(labels[labels != -1])  # ignore outlier bin
                    sorting_idxs = np.argsort(cluster_sizes)[::-1]
                    labels = [np.where(sorting_idxs == l)[0][0] if l >= 0 else len(cluster_sizes) for l in labels]
                    plt.subplot(6, 4, i + 1)
                    plt.scatter(embeds[:, 0], embeds[:, 1], s=1, c=labels, cmap="viridis")
                    plt.title(f"layer {layer + 1}")
                    plt.xlim(-100, 100)
                    plt.ylim(-100, 100)
                plt.tight_layout()
                plt.savefig(f"{outdir}/t-SNE_sample{sample}_layers{layers_to_plot[0]}-{layers_to_plot[-1]}.pdf")
                plt.close()


if __name__ == "__main__":
    matplotlib.rc("text", usetex=True)
    matplotlib.rc("font", **{"family": "serif"})
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})
    run_pbar = tqdm(sorted(os.listdir("rawsults/")), desc="Plotting")
    for run_id in run_pbar:
        for plot_fn in [plot_histograms, plot_heatmaps, plot_cluster_sizes, plot_cluster_metrics, plot_tsne]:
            run_pbar.set_postfix_str(plot_fn.__name__)
            plot_fn(run_id)
