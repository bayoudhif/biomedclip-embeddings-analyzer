import warnings
warnings.filterwarnings('ignore')
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import argparse
import pandas as pd
import itertools
import torch
import json
import random
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from PIL import Image

# Disable parallel tokenization warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Main function to load the model, handle input/output, and generate saliency maps
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading models ...")
    
    # Load the appropriate model based on the arguments
    if(args.model_name == "BiomedCLIP" and args.finetuned):
        model = AutoModel.from_pretrained(f"{args.base_path}/embeddings_analysis/model", trust_remote_code=True).to(args.device)
        processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    elif(args.model_name == "BiomedCLIP" and not args.finetuned):
        model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True).to(args.device)
        processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    elif(args.model_name == "CLIP" and not args.finetuned):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(args.device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
    elif(args.model_name == "CLIP" and args.finetuned):
        model = AutoModel.from_pretrained("./model", trust_remote_code=True).to(args.device)

    # Create output directory for embeddings if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load prompt pairs from JSON
    with open(args.json_path) as json_file:
        prompt_data = json.load(json_file)
    
    experiment_name = prompt_data['experiment_name']
    prompt_pairs = prompt_data['prompt_pairs']

    # Dictionary to store embeddings
    embeddings_data = {
        'experiment_name': experiment_name,
        'image_embeddings': [],
        'tumor_text_embeddings': [],
        'healthy_text_embeddings': [],
        'image_ids': [],
        'prompt_pairs': []
    }

    # Process each image
    for image_id in tqdm(sorted(os.listdir(args.input_path))):
        try:
            # Load and preprocess image
            image = Image.open(f"{args.input_path}/{image_id}").convert('RGB')
            image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
            
            # Process each prompt pair
            for pair in prompt_pairs:
                # Get tumor and healthy text embeddings
                tumor_text = pair['tumor']
                healthy_text = pair['healthy']
                
                # Get embeddings using forward pass
                with torch.no_grad():
                    # Get image embeddings
                    image_outputs = model.get_image_features(pixel_values=image_feat)
                    image_embeds = image_outputs.cpu().numpy()
                    
                    # Get tumor text embeddings
                    tumor_inputs = tokenizer(tumor_text, return_tensors="pt", padding=True).to(args.device)
                    tumor_outputs = model.get_text_features(**tumor_inputs)
                    tumor_embeds = tumor_outputs.cpu().numpy()
                    
                    # Get healthy text embeddings
                    healthy_inputs = tokenizer(healthy_text, return_tensors="pt", padding=True).to(args.device)
                    healthy_outputs = model.get_text_features(**healthy_inputs)
                    healthy_embeds = healthy_outputs.cpu().numpy()
                
                # Store embeddings and metadata
                embeddings_data['image_embeddings'].append(image_embeds)
                embeddings_data['tumor_text_embeddings'].append(tumor_embeds)
                embeddings_data['healthy_text_embeddings'].append(healthy_embeds)
                embeddings_data['image_ids'].append(image_id)
                embeddings_data['prompt_pairs'].append({
                    'pair_id': pair['pair_id'],
                    'tumor': tumor_text,
                    'healthy': healthy_text
                })
            
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue

    # Save embeddings to file
    output_file = os.path.join(args.output_path, f'{experiment_name}_embeddings.npy')
    np.save(output_file, embeddings_data)
    print(f"Embeddings saved to {output_file}")

# Entry point for the script
if __name__ == '__main__':
    # Define argument parser for input/output paths and hyperparameters
    parser = argparse.ArgumentParser('BiomedCLIP Embedding Extractor')
    parser.add_argument('--input-path', required=True, default="data/input_images", type=str, help='path to the images')
    parser.add_argument('--base-path', required=True, default="data/input_images", type=str, help='base path to the images')
    parser.add_argument('--output-path', required=True, default="embeddings_output", type=str, help='path to save embeddings')
    parser.add_argument('--model-name', type=str, default="BiomedCLIP", help="Which CLIP model to use")
    parser.add_argument('--finetuned', action='store_true', help="Whether to use finetuned weights or not")
    parser.add_argument('--device', type=str, default="cuda", help="Device to run the model on")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--json-path', type=str, required=True, help="Path to the JSON file containing the prompt pairs")
    args = parser.parse_args()
    main(args)
    