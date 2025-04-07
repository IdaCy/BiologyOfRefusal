import numpy as np

def compute_difference_vector(hs_array, indices_a, indices_b):
    """
    For example: (mean of group A) - (mean of group B) across all layers.
    hs_array shape: (layers, samples, seq_len, hidden_dim).
    indices_a, indices_b: list/array of sample indices.
    """
    mean_a = hs_array[:, indices_a].mean(axis=1).mean(axis=1)
    mean_b = hs_array[:, indices_b].mean(axis=1).mean(axis=1)
    return mean_a - mean_b

def project_onto_direction(activations, direction):
    """
    Project onto a direction.
    """
    norm_dir = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    projection = np.sum(activations * norm_dir[:, None, None, :], axis=-1)
    return projection

def cosine_similarity(vec1, vec2):
    """
    Get cosine similarity.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
