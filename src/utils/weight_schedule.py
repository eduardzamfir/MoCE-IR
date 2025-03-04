import torch
import math

def get_cosine_weight(epoch, total_epochs, target_weight):
    """
    Get weight based on inverted cosine annealing schedule.

    Args:
        epoch (int): Current epoch.
        total_epochs (int): Total number of epochs.
        target_weight (float): Maximum weight.

    Returns:
        float: Weight for the current epoch.
    """
    cos = math.cos(math.pi * (epoch / total_epochs))
    weight = (1 - cos) / 2 * target_weight
    return weight

# Example usage:
total_epochs = 100
target_weight = 1.0

