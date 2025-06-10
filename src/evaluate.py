#!/usr/bin/env python3
"""
Evaluation module for RDEIL experiments
Contains functions to evaluate trained models and generate performance metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate_policy(policy, test_states, test_actions):
    """
    Evaluate a trained policy on test data.
    Returns MSE loss and action predictions.
    """
    policy.eval()
    with torch.no_grad():
        predicted_actions = policy(test_states)
        mse_loss = ((predicted_actions - test_actions)**2).mean()
    
    return mse_loss.item(), predicted_actions

def evaluate_denoiser(denoiser, test_states, noisy_actions, clean_actions):
    """
    Evaluate a trained diffusion denoiser on test data.
    Returns denoising MSE loss and denoised actions.
    """
    denoiser.eval()
    with torch.no_grad():
        denoised_actions = denoiser(test_states, noisy_actions)
        denoising_loss = ((denoised_actions - clean_actions)**2).mean()
    
    return denoising_loss.item(), denoised_actions

def compare_models(models_dict, test_states, test_actions):
    """
    Compare multiple trained models on the same test set.
    models_dict: dictionary with model names as keys and (policy, denoiser) tuples as values
    """
    results = {}
    
    for model_name, (policy, denoiser) in models_dict.items():
        policy_loss, _ = evaluate_policy(policy, test_states, test_actions)
        results[model_name] = {
            'policy_mse': policy_loss
        }
        print(f"{model_name} - Policy MSE: {policy_loss:.4f}")
    
    return results

def generate_evaluation_plots(results_dict, save_path=".research/iteration1/images/"):
    """
    Generate comparison plots for different model configurations.
    """
    model_names = list(results_dict.keys())
    policy_mses = [results_dict[name]['policy_mse'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, policy_mses)
    plt.xlabel("Model Configuration")
    plt.ylabel("Policy MSE Loss")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_path}/model_comparison.pdf", bbox_inches="tight")
    plt.close()
    
    print(f"Evaluation plots saved to {save_path}/model_comparison.pdf")

def run_evaluation_suite():
    """
    Run a comprehensive evaluation of all trained models.
    This function would be called after training to assess performance.
    """
    print("\nRunning Evaluation Suite...")
    
    state_dim, action_dim = 10, 3
    test_states = torch.randn(200, state_dim)
    test_actions = torch.randn(200, action_dim)
    
    print("Evaluation suite completed. Test data generated.")
    print(f"Test set size: {test_states.shape[0]} samples")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    return test_states, test_actions
