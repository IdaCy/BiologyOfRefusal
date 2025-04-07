import matplotlib.pyplot as plt
import numpy as np

def plot_histograms_by_class(refused_values, jailbreak_values, benign_values, labels=None):
    if labels is None:
        labels = ["refused", "jailbreak", "benign"]
    
    # Opaqueness
    alpha = 1.0

    plt.hist(refused_values, bins=20, alpha=alpha, label=labels[0])
    plt.hist(jailbreak_values, bins=20, alpha=alpha, label=labels[1])
    plt.hist(benign_values, bins=20, alpha=alpha, label=labels[2])
    plt.legend()
    plt.show()

def plot_layer_linecharts(harmful_array, refused_array, harmless_array, seq_start=20):
    """
    A simpler single-figure line chart approach.
    Each array shape is (layers, samples, seq_len).
    """
    layers = harmful_array.shape[0]
    plt.figure(figsize=(10, 6))

    # We average across samples (so you see one line per layer).
    harmful_mean = harmful_array.mean(axis=1)  # shape: (layers, seq_len)
    refused_mean = refused_array.mean(axis=1)
    harmless_mean = harmless_array.mean(axis=1)

    for layer_idx in range(layers):
        plt.plot(
            np.arange(seq_start, harmful_mean.shape[1]),
            harmful_mean[layer_idx, seq_start:],
            label=f"Harmful L{layer_idx}" if layer_idx == 0 else "",
            color="blue", alpha=0.1 + 0.9*(1.0/(layer_idx+1))
        )
        plt.plot(
            np.arange(seq_start, refused_mean.shape[1]),
            refused_mean[layer_idx, seq_start:],
            label=f"Refused L{layer_idx}" if layer_idx == 0 else "",
            color="orange", alpha=0.1 + 0.9*(1.0/(layer_idx+1))
        )
        plt.plot(
            np.arange(seq_start, harmless_mean.shape[1]),
            harmless_mean[layer_idx, seq_start:],
            label=f"Harmless L{layer_idx}" if layer_idx == 0 else "",
            color="green", alpha=0.1 + 0.9*(1.0/(layer_idx+1))
        )

    plt.legend()
    plt.title("Layer linecharts (avg projection over samples)")
    plt.xlabel("Token index (skipping first 20 tokens)")
    plt.ylabel("Projection")
    plt.show()

def plot_triple_cosine_across_layers(refdir, baddir, jbdir, cosine_fn):
    """
    Plots the triple similarities across layers:
      - ref vs. bad
      - bad vs. jail
      - ref vs. jail
    Takes in each direction of shape (layers, hidden_dim).
    """
    ref_bad_sims, bad_jail_sims, ref_jail_sims = [], [], []

    for r_layer, b_layer, j_layer in zip(refdir, baddir, jbdir):
        ref_bad_sims.append(cosine_fn(r_layer, b_layer))
        bad_jail_sims.append(cosine_fn(b_layer, j_layer))
        ref_jail_sims.append(cosine_fn(r_layer, j_layer))

    # Plot them
    plt.figure(figsize=(8, 5))
    plt.plot(ref_bad_sims, label="Ref vs. Bad")
    plt.plot(bad_jail_sims, label="Bad vs. Jail")
    plt.plot(ref_jail_sims, label="Ref vs. Jail")
    plt.legend()
    plt.title("Cosine similarities across layers (ref vs. bad vs. jail)")
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity")
    plt.show()

    return ref_bad_sims, bad_jail_sims, ref_jail_sims

def plot_big_subplots_of_projection(harmful_array, refused_array, harmless_array, seq_start=20):
    """
    Large subplots; for each layer: plots lines across tokens (skipping seq_start).
    harmful_array, refused_array, harmless_array 
      -> shape: (layers, samples, seq_len)

    Set to:
      - 7 x 5 grid
      - skip first seq_start tokens
      - alpha=0.5 for lines
      - 'axes[:len(harmful_array)]' so we only plot up to the number of layers
    """
    fig, axes = plt.subplots(7, 5, figsize=(32, 32))
    axes = axes.flatten()

    colors = {'harmful': 'blue', 'rejected': 'orange', 'harmless': 'green'}
    datasets = {
        'harmful': harmful_array,
        'rejected': refused_array,
        'harmless': harmless_array
    }

    # used "for index, ax in enumerate(axes[:len(harmfulmeanedproj)])"
    # - now 'harmful_array.shape[0]' for consistency
    for index, ax in enumerate(axes[:harmful_array.shape[0]]):
        for label, data in datasets.items():
            # data[index].shape => (samples, seq_len)
            # skip the first seq_start tokens, transpose so each sample is a line
            ax.plot(data[index, :, seq_start:].T, color=colors[label], alpha=0.5, linewidth=1)
        ax.set_title(f"Layer {index}")
        ax.set_xticks([])

    # Add a single legend outside the subplots
    handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=lbl)
        for lbl, color in colors.items()
    ]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.show()



def plot_triple_cosine_across_layers(refdir, baddir, jbdir, cosine_fn):
    """
    Plots the triple similarities across layers:
      - ref vs. bad
      - bad vs. jail
      - ref vs. jail
    Takes in each direction of shape (layers, hidden_dim).
    """
    ref_bad_sims, bad_jail_sims, ref_jail_sims = [], [], []

    for r_layer, b_layer, j_layer in zip(refdir, baddir, jbdir):
        ref_bad_sims.append(cosine_fn(r_layer, b_layer))
        bad_jail_sims.append(cosine_fn(b_layer, j_layer))
        ref_jail_sims.append(cosine_fn(r_layer, j_layer))

    # Plot them
    plt.figure(figsize=(8, 5))
    plt.plot(ref_bad_sims, label="Ref vs. Bad")
    plt.plot(bad_jail_sims, label="Bad vs. Jail")
    plt.plot(ref_jail_sims, label="Ref vs. Jail")
    plt.legend()
    plt.title("Cosine similarities across layers (ref vs. bad vs. jail)")
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity")
    plt.show()

    return ref_bad_sims, bad_jail_sims, ref_jail_sims

def plot_quadruple_cosine_across_layers(refdir, baddir, jbdir, newdir, cosine_fn):
    """
    Plots the triple similarities across layers:
      - ref vs. bad
      - bad vs. jail
      - ref vs. jail
      - new
    Takes in each direction of shape (layers, hidden_dim).
    """
    ref_bad_sims, bad_jail_sims, ref_jail_sims, new_jame_sims = [], [], [], []

    for r_layer, b_layer, j_layer, n_layer in zip(refdir, baddir, jbdir, newdir):
        ref_bad_sims.append(cosine_fn(r_layer, b_layer))
        bad_jail_sims.append(cosine_fn(b_layer, j_layer))
        ref_jail_sims.append(cosine_fn(r_layer, j_layer))
        new_jame_sims.append(cosine_fn(b_layer, n_layer))

    # Plot them
    plt.figure(figsize=(8, 5))
    plt.plot(ref_bad_sims, label="Ref vs. Bad")
    plt.plot(bad_jail_sims, label="Bad vs. Jail")
    plt.plot(ref_jail_sims, label="Ref vs. Jail")
    plt.plot(new_jame_sims, label="Bad vs. Good")
    plt.legend()
    plt.title("Cosine similarities across layers (ref vs. bad vs. jail)")
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity")
    plt.show()

    return ref_bad_sims, bad_jail_sims, ref_jail_sims, new_jame_sims


"""def plot_big_subplots_of_projection(harmful_array, refused_array, harmless_array, new_array, seq_start=20):
    """
    #Large subplots; for each layer: plots lines across tokens (skipping seq_start).
    #harmful_array, refused_array, harmless_array, new_array
"""
    fig, axes = plt.subplots(7, 5, figsize=(32, 32))
    axes = axes.flatten()

    colors = {'harmful': 'blue', 'rejected': 'orange', 'harmless': 'green', 'new': 'red'}
    datasets = {
        'harmful': harmful_array,
        'rejected': refused_array,
        'harmless': harmless_array,
        'new': new_array
    }

    # used "for index, ax in enumerate(axes[:len(harmfulmeanedproj)])"
    # - now 'harmful_array.shape[0]' for consistency
    for index, ax in enumerate(axes[:harmful_array.shape[0]]):
        for label, data in datasets.items():
            # data[index].shape => (samples, seq_len)
            # skip the first seq_start tokens, transpose so each sample is a line
            ax.plot(data[index, :, seq_start:].T, color=colors[label], alpha=0.5, linewidth=1)
        ax.set_title(f"Layer {index}")
        ax.set_xticks([])

    # Add a single legend outside the subplots
    handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=lbl)
        for lbl, color in colors.items()
    ]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.show()"""
