"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA
"""
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

class mySequential(nn.Sequential):
    def forward(self, *input, **kwargs):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input

def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    """
    Replace a given module within a parent module with some third module
    Useful for injecting new layers in an existing model.
    """
    def replace_in(model: nn.Module, target: nn.Module, replacement: nn.Module):
        for name, submodule in model.named_children():
            if submodule == target:
                if isinstance(model, nn.ModuleList):
                    model[int(name)] = replacement
                elif isinstance(model, nn.Sequential):
                    model[int(name)] = replacement
                else:
                    print(3, replacement)
                    model.__setattr__(name, replacement)
                return True
            elif len(list(submodule.named_children())) > 0:
                if replace_in(submodule, target, replacement):
                    return True
    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ + " is not a child of given parent of type " + model.__class__.__name__)


class CosSimilarity:
    """ Target function """
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity()
        return cos(model_output, self.features)
    
class ImageFeatureExtractor(torch.nn.Module):
    """ Image feature wrapper """
    def __init__(self, model):
        super(ImageFeatureExtractor, self).__init__()
        self.model = model
                
    def __call__(self, x):
        return self.model.get_image_features(x)

class TextFeatureExtractor(torch.nn.Module):
    """ Text feature wrapper """
    def __init__(self, model):
        super(TextFeatureExtractor, self).__init__()   
        self.model = model
                
    def __call__(self, x):
        return self.model.get_text_features(x)
    
def image_transform(t, height=7, width=7):
    """ Transformation for CAM (image) """
    if t.size(1) == 1: t = t.permute(1,0,2)
    result = t[:, 1 :  , :].reshape(t.size(0), height, width, t.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def text_transform(t):
    """ Transformation for CAM (text) """
    if t.size(1) == 1: t = t.permute(1,0,2)
    result = t[:, :  , :].reshape(t.size(0), 1, -1, t.size(2))
    return result

def cosine_similarity(
    x: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate cosine similarity between two sets of embeddings.
    
    Args:
        x: First set of embeddings (numpy array or torch tensor)
        y: Second set of embeddings (numpy array or torch tensor)
        
    Returns:
        Cosine similarity scores between x and y
    """
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        # Convert to torch tensors for computation
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    
    # Ensure inputs are 2D
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    
    # Normalize the vectors
    x_norm = x / x.norm(dim=-1, keepdim=True)
    y_norm = y / y.norm(dim=-1, keepdim=True)
    
    # Calculate cosine similarity
    similarity = torch.matmul(x_norm, y_norm.transpose(-2, -1))
    
    # Convert back to numpy if inputs were numpy
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        similarity = similarity.numpy()
    
    return similarity

def calculate_similarity_matrix(
    embeddings: Union[np.ndarray, torch.Tensor],
    normalize: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate pairwise cosine similarity matrix for a set of embeddings.
    
    Args:
        embeddings: Set of embeddings (numpy array or torch tensor)
        normalize: Whether to normalize embeddings before calculation
        
    Returns:
        Pairwise similarity matrix
    """
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    
    if normalize:
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    similarity_matrix = torch.matmul(embeddings, embeddings.transpose(-2, -1))
    
    if isinstance(embeddings, np.ndarray):
        similarity_matrix = similarity_matrix.numpy()
    
    return similarity_matrix

def calculate_preference_margin(
    image_embeddings: Union[np.ndarray, torch.Tensor],
    tumor_embeddings: Union[np.ndarray, torch.Tensor],
    healthy_embeddings: Union[np.ndarray, torch.Tensor]
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Calculate preference margins between tumor and healthy embeddings for each image.
    
    Args:
        image_embeddings: Image embeddings
        tumor_embeddings: Tumor text embeddings
        healthy_embeddings: Healthy text embeddings
        
    Returns:
        Tuple of (tumor_similarities, healthy_similarities)
    """
    # Calculate similarities
    tumor_similarities = cosine_similarity(image_embeddings, tumor_embeddings)
    healthy_similarities = cosine_similarity(image_embeddings, healthy_embeddings)
    
    return tumor_similarities, healthy_similarities

