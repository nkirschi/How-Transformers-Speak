import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp

from matplotlib.ticker import PercentFormatter
from tqdm import tqdm


def plot_histograms(run_id, conf_level=0.99):
    outdir = f"visualisation/{run_id}"
    os.makedirs(outdir, exist_ok=True)

    count_tensor = np.load(f"rawsults/{run_id}/histograms.npy")
    count_mean = np.mean(count_tensor, axis=0)  # shape num_layer x num_bins
    t_score = sp.stats.t.ppf(q=1 - (1 - conf_level) / 2, df=count_tensor.shape[0] - 1)
    count_conf = t_score * np.std(count_tensor, axis=0) / np.sqrt(
        count_tensor.shape[0])  # t-distribution confidence deviation
    num_bins = count_tensor.shape[-1]
    max_density = count_tensor.max()

    plt.figure(figsize=(12, 16))
    plt.suptitle(f"cos similarity histograms\n({run_id})\n")
    num_layers = count_mean.shape[0]
    stride = num_layers // 24
    for i, layer in enumerate(range(stride, num_layers + stride, stride)):
        idx = layer - 1
        plt.subplot(6, 4, i + 1)
        plt.stairs(count_mean[idx], np.linspace(-1, 1, num_bins + 1))
        x = np.linspace(-1 + 1 / num_bins, 1 - 1 / num_bins, num_bins)
        plt.fill_between(x, count_mean[idx] - count_conf[idx], count_mean[idx] + count_conf[idx], alpha=0.5, step="mid")
        plt.xlim(-1, 1)
        plt.ylim(0, max_density)  # set a consistent y-axis limit
        plt.title(f"after layer {layer}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/histograms.pdf")
    plt.close()


def plot_heatmaps(run_id):
    outdir = f"visualisation/{run_id}"
    os.makedirs(outdir, exist_ok=True)

    heatmap_tensor = np.load(f"rawsults/{run_id}/sim_matrices.npy")
    seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]

    for sample in range(heatmap_tensor.shape[0]):
        plt.figure(figsize=(12, 16))
        plt.suptitle(f"cos similarity heatmaps\n({run_id})\n")
        num_layers = heatmap_tensor.shape[1]
        stride = num_layers // 24
        for i, layer in enumerate(range(stride, num_layers + stride, stride)):
            sim_matrix = heatmap_tensor[sample, layer - 1, :seq_lens[sample], :seq_lens[sample]]
            plt.subplot(6, 4, i + 1)
            plt.imshow(sim_matrix, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(label="cosine similarity")
            plt.xlabel("token j")
            plt.ylabel("token i")
            plt.title(f"after layer {layer}")
        plt.tight_layout()
        plt.savefig(f"{outdir}/heatmaps_sample{sample}.pdf")
        plt.close()


def plot_tsne(run_id):
    for target in ["X", "XX.T"]:
        outdir = f"visualisation/{run_id}/tsne_{target}"
        os.makedirs(outdir, exist_ok=True)

        tsne_tensor = np.load(f"rawsults/{run_id}/tsne_{target}.npy")
        labels_tensor = np.load(f"rawsults/{run_id}/labels_{target}.npy")
        seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]

        for sample in range(tsne_tensor.shape[0]):
            plt.figure(figsize=(12, 16))
            plt.suptitle(f"t-SNE visualisations\n({run_id})\n")
            num_layers = tsne_tensor.shape[1]
            stride = num_layers // 24
            for i, layer in enumerate(range(stride, num_layers + stride, stride)):
                embeds = tsne_tensor[sample, layer - 1, :seq_lens[sample]]
                labels = labels_tensor[sample, layer - 1, :seq_lens[sample]]
                plt.subplot(6, 4, i + 1)
                plt.scatter(embeds[:, 0], embeds[:, 1], s=1, c=labels, cmap="viridis")
                plt.title(f"after layer {layer}")
                plt.xlim(-100, 100)
                plt.ylim(-100, 100)
            plt.tight_layout()
            plt.savefig(f"{outdir}/tsne_sample{sample}.pdf")
            plt.close()


def plot_clustering(run_id):
    for target in ["X", "XX.T"]:
        outdir = f"visualisation/{run_id}/clustereval_{target}"
        os.makedirs(outdir, exist_ok=True)

        clustereval_tensor = np.load(f"rawsults/{run_id}/clustereval_{target}.npy")
        labels_tensor = np.load(f"rawsults/{run_id}/labels_{target}.npy")
        seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]

        for sample in range(clustereval_tensor.shape[0]):
            plt.figure(figsize=(6, 8))
            plt.suptitle(f"HDBSCAN cluster evaluation\n({run_id})\n")
            for i, title in enumerate(["number of clusters", "outlier rate", "Silhouette score",
                                       "Calinski-Harabasz score", "Davies-Bouldin score"]):
                plt.subplot(3, 2, 1 + i)
                plt.title(title)
                plt.plot(clustereval_tensor[sample, :, i])
                plt.xlabel("layer")
                if i == 1:
                    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
            plt.tight_layout()
            plt.savefig(f"{outdir}/cluster_evaluation_sample{sample}.pdf")
            plt.close()

            plt.figure(figsize=(12, 16))
            plt.suptitle(f"HDBSCAN cluster sizes\n({run_id})\n")
            num_layers = clustereval_tensor.shape[1]
            stride = num_layers // 24
            for i, layer in enumerate(range(stride, num_layers + stride, stride)):
                labels = labels_tensor[sample, layer - 1, :seq_lens[sample]]
                cluster_sizes = np.bincount(labels[labels != -1])  # ignore outlier bin
                cluster_sizes = np.sort(cluster_sizes)[::-1]  # sort by decreasing size
                plt.subplot(6, 4, i + 1)
                plt.bar(range(1, len(cluster_sizes) + 1), cluster_sizes)
                plt.title(f"after layer {layer}")
                plt.xlabel("k-th largest cluster")
                plt.xticks(range(1, len(cluster_sizes) + 1))
                if i % 4 == 0:
                    plt.ylabel("cluster size")
            plt.tight_layout()
            plt.savefig(f"{outdir}/cluster_size_sample{sample}.pdf")
            plt.close()


if __name__ == "__main__":
    run_pbar = tqdm(os.listdir("rawsults/"))
    for run_id in run_pbar:
        for plot_fn in [plot_histograms, plot_heatmaps, plot_clustering, plot_tsne]:
            run_pbar.set_postfix_str(plot_fn.__name__)
            plot_fn(run_id)
