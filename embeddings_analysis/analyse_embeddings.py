import os
import numpy as np
import torch
import argparse
from scripts.utils import (
    cosine_similarity,
    calculate_similarity_matrix,
    calculate_preference_margin
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

def load_embeddings(file_path: str) -> Dict[str, Any]:
    """
    Load embeddings from .npy file.
    
    Args:
        file_path: Path to the .npy file containing embeddings
        
    Returns:
        Dictionary containing embeddings and metadata
    """
    data = np.load(file_path, allow_pickle=True).item()
    return data

def analyze_embeddings(data: Dict[str, Any], output_dir: str):
    """
    Analyze embeddings using various similarity metrics.
    
    Args:
        data: Dictionary containing embeddings and metadata
        output_dir: Directory to save analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert lists to numpy arrays and reshape if necessary
    image_embeddings = np.stack(data['image_embeddings'])
    tumor_embeddings = np.stack(data['tumor_text_embeddings'])
    healthy_embeddings = np.stack(data['healthy_text_embeddings'])
    
    # Reshape if needed (remove extra dimensions)
    if image_embeddings.ndim > 2:
        image_embeddings = image_embeddings.reshape(image_embeddings.shape[0], -1)
    if tumor_embeddings.ndim > 2:
        tumor_embeddings = tumor_embeddings.reshape(tumor_embeddings.shape[0], -1)
    if healthy_embeddings.ndim > 2:
        healthy_embeddings = healthy_embeddings.reshape(healthy_embeddings.shape[0], -1)
    
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Tumor embeddings shape: {tumor_embeddings.shape}")
    print(f"Healthy embeddings shape: {healthy_embeddings.shape}")
    
    # Calculate preference margins
    tumor_sims, healthy_sims = calculate_preference_margin(
        image_embeddings,
        tumor_embeddings,
        healthy_embeddings
    )
    
    # Calculate margins (difference between tumor and healthy similarities)
    margins = tumor_sims - healthy_sims
    
    # Create analysis results
    experiment_name = data.get('experiment_name', 'experiment')
    results = {
        'tumor_similarities': tumor_sims,
        'healthy_similarities': healthy_sims,
        'preference_margins': margins,
        'image_ids': data['image_ids'],
        'prompt_pairs': data['prompt_pairs']
    }
    
    # Save results
    np.save(os.path.join(output_dir, f'{experiment_name}_analysis_results.npy'), results)
    
    # Generate visualizations
    plot_similarity_distributions(tumor_sims, healthy_sims, margins, output_dir, prefix=experiment_name)
    plot_similarity_matrix(image_embeddings, output_dir, prefix=experiment_name)

    # --- Text-to-Text Similarity Matrix ---
    all_text_embeddings = np.vstack([tumor_embeddings, healthy_embeddings])
    text_labels = ['tumor'] * len(tumor_embeddings) + ['healthy'] * len(healthy_embeddings)
    text_sim_matrix = cosine_similarity(all_text_embeddings, all_text_embeddings)
    plt.figure(figsize=(10, 8))
    sns.heatmap(text_sim_matrix, cmap='viridis')
    plt.title('Text-to-Text Embedding Similarity Matrix')
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_text_similarity_matrix.png'))
    plt.close()

    # --- Save average text-to-text similarity ---
    # Ensure text_sim_matrix is a numpy array
    if not isinstance(text_sim_matrix, np.ndarray):
        text_sim_matrix = text_sim_matrix.detach().cpu().numpy() if hasattr(text_sim_matrix, 'detach') else np.array(text_sim_matrix)
    avg_text_sim = np.mean(text_sim_matrix)
    with open(os.path.join(output_dir, f'{experiment_name}_avg_text_similarity.txt'), 'w') as f:
        f.write(f'Average text-to-text cosine similarity: {avg_text_sim}\n')

    # --- PCA Visualization of Text Embeddings ---
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_text_embeddings)
    colors = ['red' if l == 'tumor' else 'blue' for l in text_labels]
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.7)
    plt.title('PCA of Text Embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_text_pca.png'))
    plt.close()

    # --- t-SNE Visualization of Text Embeddings ---
    tsne = TSNE(n_components=2, random_state=42)
    tsne_proj = tsne.fit_transform(all_text_embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors, alpha=0.7)
    plt.title('t-SNE of Text Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_text_tsne.png'))
    plt.close()

    # --- Silhouette Score ---
    # Convert labels to 0/1 for silhouette_score
    label_ids = [0 if l == 'tumor' else 1 for l in text_labels]
    score = silhouette_score(all_text_embeddings, label_ids)
    print(f'Silhouette Score (healthy vs tumor): {score}')
    with open(os.path.join(output_dir, f'{experiment_name}_silhouette_score.txt'), 'w') as f:
        f.write(f'Silhouette Score (healthy vs tumor): {score}\n')

def plot_similarity_distributions(
    tumor_sims: np.ndarray,
    healthy_sims: np.ndarray,
    margins: np.ndarray,
    output_dir: str,
    prefix: str = ''
):
    """
    Plot distributions of similarity scores and margins.
    
    Args:
        tumor_sims: Tumor similarity scores
        healthy_sims: Healthy similarity scores
        margins: Preference margins
        output_dir: Directory to save plots
        prefix: Prefix for output file names
    """
    plt.figure(figsize=(15, 5))
    
    # Plot tumor similarities
    plt.subplot(131)
    sns.histplot(tumor_sims.flatten(), bins=30)
    plt.title('Tumor Similarities')
    plt.xlabel('Cosine Similarity')
    
    # Plot healthy similarities
    plt.subplot(132)
    sns.histplot(healthy_sims.flatten(), bins=30)
    plt.title('Healthy Similarities')
    plt.xlabel('Cosine Similarity')
    
    # Plot margins
    plt.subplot(133)
    sns.histplot(margins.flatten(), bins=30)
    plt.title('Preference Margins')
    plt.xlabel('Margin (Tumor - Healthy)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_similarity_distributions.png'))
    plt.close()

def plot_similarity_matrix(embeddings: np.ndarray, output_dir: str, prefix: str = ''):
    """
    Plot similarity matrix for all embeddings.
    
    Args:
        embeddings: Image embeddings
        output_dir: Directory to save plot
        prefix: Prefix for output file names
    """
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title('Embedding Similarity Matrix')
    plt.savefig(os.path.join(output_dir, f'{prefix}_similarity_matrix.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser('Embedding Analysis')
    parser.add_argument('--embeddings-path', required=True, type=str,
                      help='Path to the embeddings .npy file')
    parser.add_argument('--output-dir', required=True, type=str,
                      help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Load embeddings
    print("Loading embeddings...")
    data = load_embeddings(args.embeddings_path)
    
    # Analyze embeddings
    print("Analyzing embeddings...")
    analyze_embeddings(data, args.output_dir)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
