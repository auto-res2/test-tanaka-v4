#!/usr/bin/env python3
"""
Data preprocessing module for RDEIL experiments
Handles data loading, normalization, and preparation for training.
"""

import torch
import numpy as np
import os

def generate_dummy_expert_data(num_samples=1000, state_dim=10, action_dim=3, seed=42):
    """
    Generate dummy expert demonstration data for experiments.
    In a real scenario, this would load actual expert trajectories.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    states = torch.randn(num_samples, state_dim)
    
    action_weights = torch.randn(state_dim, action_dim) * 0.5
    actions = torch.matmul(states, action_weights) + torch.randn(num_samples, action_dim) * 0.1
    
    return states, actions

def normalize_data(states, actions):
    """
    Normalize states and actions to have zero mean and unit variance.
    Returns normalized data and normalization parameters.
    """
    state_mean = states.mean(dim=0, keepdim=True)
    state_std = states.std(dim=0, keepdim=True) + 1e-8
    normalized_states = (states - state_mean) / state_std
    
    action_mean = actions.mean(dim=0, keepdim=True)
    action_std = actions.std(dim=0, keepdim=True) + 1e-8
    normalized_actions = (actions - action_mean) / action_std
    
    normalization_params = {
        'state_mean': state_mean,
        'state_std': state_std,
        'action_mean': action_mean,
        'action_std': action_std
    }
    
    return normalized_states, normalized_actions, normalization_params

def split_data(states, actions, train_ratio=0.8, val_ratio=0.1):
    """
    Split data into training, validation, and test sets.
    """
    num_samples = states.shape[0]
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    
    indices = torch.randperm(num_samples)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    
    train_states, train_actions = states[train_indices], actions[train_indices]
    val_states, val_actions = states[val_indices], actions[val_indices]
    test_states, test_actions = states[test_indices], actions[test_indices]
    
    return (train_states, train_actions), (val_states, val_actions), (test_states, test_actions)

def save_processed_data(data_dict, save_dir="data/"):
    """
    Save processed data to disk for later use.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            torch.save(value, os.path.join(save_dir, f"{key}.pt"))
        else:
            torch.save(value, os.path.join(save_dir, f"{key}.pt"))
    
    print(f"Processed data saved to {save_dir}")

def load_processed_data(save_dir="data/"):
    """
    Load previously processed data from disk.
    """
    data_dict = {}
    
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            if filename.endswith('.pt'):
                key = filename[:-3]  # Remove .pt extension
                data_dict[key] = torch.load(os.path.join(save_dir, filename))
        
        print(f"Loaded processed data from {save_dir}")
    else:
        print(f"No processed data found in {save_dir}")
    
    return data_dict

def prepare_experiment_data(num_samples=1000, state_dim=10, action_dim=3, normalize=True, save_data=True):
    """
    Complete data preparation pipeline for RDEIL experiments.
    """
    print("Preparing experiment data...")
    
    states, actions = generate_dummy_expert_data(num_samples, state_dim, action_dim)
    print(f"Generated {num_samples} expert demonstrations")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    if normalize:
        states, actions, norm_params = normalize_data(states, actions)
        print("Data normalized")
    else:
        norm_params = None
    
    (train_states, train_actions), (val_states, val_actions), (test_states, test_actions) = split_data(states, actions)
    
    print(f"Data split - Train: {train_states.shape[0]}, Val: {val_states.shape[0]}, Test: {test_states.shape[0]}")
    
    data_dict = {
        'train_states': train_states,
        'train_actions': train_actions,
        'val_states': val_states,
        'val_actions': val_actions,
        'test_states': test_states,
        'test_actions': test_actions
    }
    
    if norm_params:
        data_dict['normalization_params'] = norm_params
    
    if save_data:
        save_processed_data(data_dict)
    
    print("Data preparation completed")
    return data_dict
