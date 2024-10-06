import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp

from matplotlib.ticker import PercentFormatter
from tqdm import tqdm


def plot_histograms(run_id, num_bins=100, conf_level=0.99):
    for target in ["XXᵀ", "XQᵀKXᵀ"]:
        outdir = f"visualisation/{run_id}/similarities_{target}"
        os.makedirs(outdir, exist_ok=True)

        sim_tensor = np.load(f"rawsults/{run_id}/similarity_matrices_{target}.npy")
        seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]
        count_tensor = np.stack([np.stack([
            np.histogram(layer[:seq_lens[i], :seq_lens[i]].flatten(), bins=num_bins, range=(-1, 1), density=True)[0]
            for layer in sample]) for i, sample in enumerate(sim_tensor)
        ])
        count_mean = np.mean(count_tensor, axis=0)  # shape num_layer x num_bins
        t_score = sp.stats.t.ppf(q=1 - (1 - conf_level) / 2, df=count_tensor.shape[0] - 1)
        count_conf = t_score * np.std(count_tensor, axis=0) / np.sqrt(count_tensor.shape[0])  # t-distribution confidence deviation
        num_bins = count_tensor.shape[-1]
        max_density = count_tensor.max()

        plt.figure(figsize=(12, 16))
        plt.suptitle(f"cos similarity histograms\n({run_id})\n")
        num_layers = count_mean.shape[0]
        layers_to_plot = [i if i < 4
                          else num_layers - 24 + i if i >= 20
                          else 3 + round((i - 3) * (num_layers - 8) / 16)
                          for i in range(24)]
        for i, layer in enumerate(layers_to_plot):
            plt.subplot(6, 4, i + 1)
            plt.stairs(count_mean[layer], np.linspace(-1, 1, num_bins + 1))
            x = np.linspace(-1 + 1 / num_bins, 1 - 1 / num_bins, num_bins)
            plt.fill_between(x, count_mean[layer] - count_conf[layer], count_mean[layer] + count_conf[layer],
                             alpha=0.5, step="mid")
            plt.xlim(-1, 1)
            plt.ylim(0, max_density)  # set a consistent y-axis limit
            plt.title(f"after layer {layer + 1}")
        plt.tight_layout()
        plt.savefig(f"{outdir}/histograms.pdf")
        plt.close()


def plot_heatmaps(run_id):
    for target in ["XXᵀ", "XQᵀKXᵀ"]:
        outdir = f"visualisation/{run_id}/similarities_{target}"
        os.makedirs(outdir, exist_ok=True)

        sim_tensor = np.load(f"rawsults/{run_id}/similarity_matrices_{target}.npy")
        seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]

        for sample in range(sim_tensor.shape[0]):
            plt.figure(figsize=(12, 16))
            plt.suptitle(f"cos similarity heatmaps\n({run_id})\n")
            num_layers = sim_tensor.shape[1]
            layers_to_plot = [i if i < 4
                              else num_layers - 24 + i if i >= 20
                              else 3 + round((i - 3) * (num_layers - 8) / 16)
                              for i in range(24)]
            for i, layer in enumerate(layers_to_plot):
                sim_matrix = sim_tensor[sample, layer, :seq_lens[sample], :seq_lens[sample]]
                plt.subplot(6, 4, i + 1)
                plt.imshow(sim_matrix, cmap="coolwarm", vmin=-1, vmax=1)
                plt.colorbar(label="cosine similarity")
                plt.xlabel("token j")
                plt.ylabel("token i")
                plt.title(f"after layer {layer + 1}")
            plt.tight_layout()
            plt.savefig(f"{outdir}/heatmaps_sample{sample}.pdf")
            plt.close()


def plot_cluster_metrics(run_id):
    for target in ["X", "XQᵀKXᵀ"]:
        outdir = f"visualisation/{run_id}/clustering_{target}"
        os.makedirs(outdir, exist_ok=True)

        metrics_tensor = np.load(f"rawsults/{run_id}/cluster_metrics_{target}.npy")

        for sample in range(metrics_tensor.shape[0]):
            plt.figure(figsize=(6, 8))
            plt.suptitle(f"HDBSCAN cluster evaluation\n({run_id})\n")
            for i, title in enumerate(["number of clusters", "outlier rate", "Silhouette score", "DBCV score"]):
                plt.subplot(2, 2, 1 + i)
                plt.title(title)
                plt.plot(metrics_tensor[sample, :, i])
                plt.xlabel("layer")
                if i == 1:
                    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
            plt.tight_layout()
            plt.savefig(f"{outdir}/cluster_metrics_sample{sample}.pdf")
            plt.close()


def plot_cluster_sizes(run_id):
    for target in ["X", "XQᵀKXᵀ"]:
        outdir = f"visualisation/{run_id}/clustering_{target}"
        os.makedirs(outdir, exist_ok=True)

        metrics_tensor = np.load(f"rawsults/{run_id}/cluster_metrics_{target}.npy")
        labels_tensor = np.load(f"rawsults/{run_id}/cluster_labels_{target}.npy")
        seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]

        for sample in range(metrics_tensor.shape[0]):
            plt.figure(figsize=(12, 16))
            plt.suptitle(f"HDBSCAN cluster sizes\n({run_id})\n")
            num_layers = metrics_tensor.shape[1]
            layers_to_plot = [i if i < 4
                              else num_layers - 24 + i if i >= 20
                              else 3 + round((i - 3) * (num_layers - 8) / 16)
                              for i in range(24)]
            for i, layer in enumerate(layers_to_plot):
                labels = labels_tensor[sample, layer, :seq_lens[sample]]
                cluster_sizes = np.bincount(labels[labels != -1])  # ignore outlier bin
                cluster_sizes = np.sort(cluster_sizes)[::-1]  # sort by decreasing size
                colours = plt.colormaps["viridis"](np.linspace(0, 1, len(cluster_sizes) + 1)[:-1])
                plt.subplot(6, 4, i + 1)
                plt.bar(range(1, len(cluster_sizes) + 1), cluster_sizes, color=colours)
                plt.title(f"after layer {layer + 1}")
                plt.xlabel("k-th largest cluster")
                plt.xticks(range(1, len(cluster_sizes) + 1))
                if i % 4 == 0:
                    plt.ylabel("cluster size")
            plt.tight_layout()
            plt.savefig(f"{outdir}/cluster_sizes_sample{sample}.pdf")
            plt.close()

def plot_tsne(run_id):
    for target in ["X", "XQᵀKXᵀ"]:
        outdir = f"visualisation/{run_id}/clustering_{target}"
        os.makedirs(outdir, exist_ok=True)

        tsne_tensor = np.load(f"rawsults/{run_id}/t-SNE_embeddings_{target}.npy")
        labels_tensor = np.load(f"rawsults/{run_id}/cluster_labels_{target}.npy")
        seq_lens = pd.read_csv(f"rawsults/{run_id}/0verview.csv", index_col=0)["num_tokens"]

        for sample in range(tsne_tensor.shape[0]):
            plt.figure(figsize=(12, 16))
            plt.suptitle(f"t-SNE visualisations\n({run_id})\n")
            num_layers = tsne_tensor.shape[1]
            layers_to_plot = [i if i < 4
                              else num_layers - 24 + i if i >= 20
                              else 3 + round((i - 3) * (num_layers - 8) / 16)
                              for i in range(24)]
            for i, layer in enumerate(layers_to_plot):
                embeds = tsne_tensor[sample, layer, :seq_lens[sample]]
                labels = labels_tensor[sample, layer, :seq_lens[sample]]
                cluster_sizes = np.bincount(labels[labels != -1])  # ignore outlier bin
                sorting_idxs = np.argsort(cluster_sizes)[::-1]
                labels = [np.where(sorting_idxs == l)[0][0] if l >= 0 else len(cluster_sizes) for l in labels]
                plt.subplot(6, 4, i + 1)
                plt.scatter(embeds[:, 0], embeds[:, 1], s=1, c=labels, cmap="viridis")
                plt.title(f"after layer {layer + 1}")
                plt.xlim(-100, 100)
                plt.ylim(-100, 100)
            plt.tight_layout()
            plt.savefig(f"{outdir}/t-SNE_sample{sample}.pdf")
            plt.close()


if __name__ == "__main__":
    run_pbar = tqdm(os.listdir("rawsults/"))
    for run_id in run_pbar:
        for plot_fn in [plot_histograms, plot_heatmaps, plot_cluster_sizes, plot_cluster_metrics, plot_tsne]:
            run_pbar.set_postfix_str(plot_fn.__name__)
            plot_fn(run_id)
