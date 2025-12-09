import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_patching_heatmap(scores: np.ndarray, save_path=None):
    """
    Graph heatmap with layer on Y, head on X
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(scores, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Logit Difference Change'})
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Activation Patching Scores Heatmap")
    if save_path:
        plt.savefig(save_path)

