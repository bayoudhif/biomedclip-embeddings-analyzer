# MedCLIP-SAMv2 : BioMedClip Embedding Analysis

This folder contains scripts for generating and analyzing text-image embeddings using the MedCLIP-SAMv2 model. The workflow allows you to:
- Generate embeddings for images and paired healthy/tumor prompts
- Analyze the semantic separation and alignment in the embedding space
- Visualize similarities and clustering between healthy and tumor concepts

## Workflow Overview

1. **Generate Embeddings**
2. **Analyze Embeddings**
3. **Interpret Results**

---

## 1. Generate Embeddings

Use `generate_embeddings.py` to extract image and text embeddings for your dataset and prompt pairs.

**Example Command:**
```bash
python embeddings_analysis/generate_embeddings.py \
  --input-path "<path_to_images>" \
  --base-path "<base_path>" \
  --output-path "<output_dir>" \
  --model-name "BiomedCLIP" \
  --device "cuda" \
  --json-path "<path_to_prompts_json>" \
  --finetuned
```
- `--input-path`: Directory with input images
- `--base-path`: Base path for model loading
- `--output-path`: Where to save the embeddings
- `--json-path`: JSON file with paired healthy/tumor prompts
- `--finetuned`: Use finetuned weights (optional)

**Output:**
- `<experiment_name>_embeddings.npy` (contains all embeddings and metadata)

---

## 2. Analyze Embeddings

Use `analyse_embeddings.py` to compute similarity metrics, visualize the embedding space, and assess semantic separation.

**Example Command:**
```bash
python embeddings_analysis/analyse_embeddings.py \
  --embeddings-path "<output_dir>/<experiment_name>_embeddings.npy" \
  --output-dir "<analysis_output_dir>"
```

**Outputs:**
- `{experiment_name}_analysis_results.npy`: All computed metrics and raw results
- `{experiment_name}_similarity_distributions.png`: Histograms of image-to-text similarities and preference margins
- `{experiment_name}_similarity_matrix.png`: Image-to-image similarity heatmap
- `{experiment_name}_text_similarity_matrix.png`: Text-to-text similarity heatmap
- `{experiment_name}_text_pca.png`: PCA plot of text embeddings (healthy vs tumor)
- `{experiment_name}_text_tsne.png`: t-SNE plot of text embeddings (healthy vs tumor)
- `{experiment_name}_silhouette_score.txt`: Silhouette score for healthy/tumor text embedding separation
- `{experiment_name}_avg_text_similarity.txt`: Average cosine similarity between all text prompt embeddings

---

## 3. Interpret Results

- **similarity_distributions.png**: Shows how well images align with healthy vs tumor prompts and the margin between them.
- **similarity_matrix.png**: Visualizes similarity between all image embeddings.
- **text_similarity_matrix.png**: Visualizes similarity between all text prompt embeddings.
- **text_pca.png / text_tsne.png**: Show clustering/separation of healthy and tumor prompts in embedding space.
- **silhouette_score.txt**: Higher values mean better separation between healthy and tumor prompts.
- **avg_text_similarity.txt**: Lower values mean healthy and tumor prompts are more distinct in embedding space.

---

## 4. Convert Analysis Results to JSON (Optional)

Use `npy_to_json.py` to convert `.npy` results to `.json` for easier inspection or sharing.

**Example Command:**
```bash
python embeddings_analysis/npy_to_json.py \
  --npy-path "<analysis_output_dir>/<experiment_name>_analysis_results.npy" \
  --json-path "<analysis_output_dir>/<experiment_name>_analysis_results.json"
```

---

## Requirements
- Python 3.8+
- torch, numpy, matplotlib, seaborn, scikit-learn, transformers, PIL
- (Optional) tqdm for progress bars

Install requirements with:
```bash
pip install torch numpy matplotlib seaborn scikit-learn transformers pillow tqdm
```

---

## Contact
For questions or issues, please contact the repository maintainer. 