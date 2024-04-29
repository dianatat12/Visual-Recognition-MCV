

import torch


def calculate_distance(vector1, vector2, method: str = "l2"):
    if method == "l2":
        distance = torch.norm(vector1 - vector2)
    elif method == "cosine":
        similarity_matrix = torch.nn.functional.cosine_similarity(
            vector1.unsqueeze(0), vector2.unsqueeze(0)
        )
        distance = 1 - similarity_matrix  # Distance is 1 - similarity
    elif method == "manhattan":
        distance = torch.sum(torch.abs(vector1 - vector2))
    elif method == "minkowski":
        p = 2  # Euclidean distance
        distance = torch.pow(
            torch.sum(torch.pow(torch.abs(vector1 - vector2), p)), 1 / p
        )
    elif method == "pearson":
        vector1_mean = torch.mean(vector1)
        vector2_mean = torch.mean(vector2)
        numerator = torch.sum((vector1 - vector1_mean) * (vector2 - vector2_mean))
        denominator = torch.sqrt(
            torch.sum((vector1 - vector1_mean) ** 2)
            * torch.sum((vector2 - vector2_mean) ** 2)
        )
        distance = 1 - (numerator / denominator)
    elif method == "jaccard":
        intersection = torch.sum(torch.minimum(vector1, vector2))
        union = torch.sum(torch.maximum(vector1, vector2))
        distance = 1 - (intersection / union)
    else:
        raise ValueError(
            "Invalid method name. Supported methods are 'l2', 'cosine', 'manhattan', 'minkowski', 'pearson', and 'jaccard'"
        )

    return distance.item()  # Convert to Python scalar if needed
