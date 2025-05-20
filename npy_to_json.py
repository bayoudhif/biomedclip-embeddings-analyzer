import numpy as np
import torch
import json
import argparse

def npy_to_json(npy_path, json_path):
    data = np.load(npy_path, allow_pickle=True).item()
    # Convert all numpy arrays and torch tensors to lists for JSON serialization
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, torch.Tensor):
            return item.cpu().tolist()
        elif isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert(v) for v in item]
        else:
            return item
    data = convert(data)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy (dict) file to .json file.")
    parser.add_argument('--npy-path', required=True, help='Path to the .npy file')
    parser.add_argument('--json-path', required=True, help='Path to save the .json file')
    args = parser.parse_args()
    npy_to_json(args.npy_path, args.json_path) 