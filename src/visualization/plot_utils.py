import matplotlib.pyplot as plt

def plot_histograms_by_class(refused_values, jailbreak_values, benign_values, labels=None):
    """
    Plot overlapping histograms for the three arrays of values.
    refused_values, jailbreak_values, benign_values: 1D arrays or lists
    labels: optional list/tuple of the labels to use in legend
    """
    if labels is None:
        labels = ["refused", "jailbreak", "benign"]
    plt.hist(refused_values, bins=20, alpha=0.6, label=labels[0])
    plt.hist(jailbreak_values, bins=20, alpha=0.6, label=labels[1])
    plt.hist(benign_values, bins=20, alpha=0.6, label=labels[2])
    plt.legend()
    plt.show()

def plot_layer_linecharts(
    harmful_array, refused_array, harmless_array,
    seq_start=20
):
    """
    Plot line charts for each layer. 
    Each array shape is (layers, samples, seq_len).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    layers = harmful_array.shape[0]
    fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(32, 32))
    axes = axes.flatten()

    colors = {'harmful': 'blue', 'rejected': 'orange', 'harmless': 'green'}
    datasets = {
        'harmful': harmful_array,
        'rejected': refused_array,
        'harmless': harmless_array
    }

    for index in range(min(layers, len(axes))):
        ax = axes[index]
        for label, data in datasets.items():
            # data[index, :, seq_start:] => shape (samples, subseq_len)
            ax.plot(data[index, :, seq_start:].T, color=colors[label], alpha=0.5, linewidth=1)
        ax.set_title(f"Layer {index}")
        ax.set_xticks([])

    # Hide any unused axes
    for index in range(layers, len(axes)):
        axes[index].axis('off')

    handles = [plt.Line2D([0], [0], color=c, lw=2, label=l) for l,c in colors.items()]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()
