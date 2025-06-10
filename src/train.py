#!/usr/bin/env python3
"""
Training module for RDEIL (Robust Diffusion-Enhanced Imitation Learning)
Contains Policy and DiffusionDenoiser models and training functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.net(state)

class DiffusionDenoiser(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiffusionDenoiser, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state, noisy_action):
        x = torch.cat([state, noisy_action], dim=-1)
        denoised_action = self.net(x)
        return denoised_action

def corrupt_actions(actions, noise_std):
    """Inject fixed Gaussian noise to expert actions."""
    noise = torch.randn_like(actions) * noise_std
    return actions + noise

def compute_adaptive_noise(state, base_std=0.1):
    """
    Dummy adaptive noise scheduler:
    For illustration, states with larger norm (considered more 'critical')
    receive lower noise (inverse relation).
    """
    state_norm = torch.norm(state, dim=-1, keepdim=True)
    adaptive_std = base_std / (1.0 + state_norm)
    return adaptive_std

def corrupt_actions_adaptive(actions, states, base_std=0.1):
    """Inject adaptive noise using the computed noise_std for each state."""
    noise_std = compute_adaptive_noise(states, base_std=base_std)
    noise = torch.randn_like(actions) * noise_std
    return actions + noise

def train_experiment1(noise_std=0.1, diffusion_weight=1.0, epochs=50):
    """
    Experiment 1: Robustness in the Presence of Noisy Expert Demonstrations
    """
    print("\nRunning Experiment 1: Robustness in the Presence of Noisy Expert Demonstrations")
    
    state_dim, action_dim = 10, 3
    policy = Policy(state_dim, action_dim)
    denoiser = DiffusionDenoiser(state_dim, action_dim)
    
    optimizer = optim.Adam(list(policy.parameters()) + list(denoiser.parameters()), lr=1e-3)

    expert_states = torch.randn(1000, state_dim)
    expert_actions = torch.randn(1000, action_dim)

    bc_losses = []
    diff_losses = []
    total_losses = []
    
    for epoch in range(epochs):
        noisy_expert_actions = corrupt_actions(expert_actions, noise_std=noise_std)
        
        policy_actions = policy(expert_states)
        
        bc_loss = ((policy_actions - noisy_expert_actions)**2).mean()
        
        denoised_actions = denoiser(expert_states, noisy_expert_actions)
        diffusion_loss = ((denoised_actions - expert_actions)**2).mean()
        
        total_loss = bc_loss + diffusion_weight * diffusion_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        bc_losses.append(bc_loss.item())
        diff_losses.append(diffusion_loss.item())
        total_losses.append(total_loss.item())
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, BC Loss: {bc_loss.item():.4f}, Diffusion Loss: {diffusion_loss.item():.4f}')

    plt.figure()
    plt.plot(total_losses, label="Total Loss")
    plt.plot(bc_losses, label="BC Loss")
    plt.plot(diff_losses, label="Diffusion Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Experiment1: Loss Curves")
    plt.legend()
    plt.savefig(".research/iteration1/images/training_loss.pdf", bbox_inches="tight")
    plt.close()
    
    print("Experiment 1 completed. Loss curve saved as training_loss.pdf")
    return policy, denoiser

def train_ablation_experiment(loss_type='combined', noise_std=0.1, diffusion_weight=1.0, epochs=50):
    """
    Experiment 2: Ablation Study on Dual Loss Components
    """
    print(f"\nRunning Experiment 2: Ablation Study, Loss Type: {loss_type}")
    
    state_dim, action_dim = 10, 3
    policy = Policy(state_dim, action_dim)
    denoiser = DiffusionDenoiser(state_dim, action_dim)
    optimizer = optim.Adam(list(policy.parameters()) + list(denoiser.parameters()), lr=1e-3)
    
    expert_states = torch.randn(1000, state_dim)
    expert_actions = torch.randn(1000, action_dim)
    
    losses = []
    
    for epoch in range(epochs):
        noisy_expert_actions = corrupt_actions(expert_actions, noise_std=noise_std)
        policy_actions = policy(expert_states)
        
        if loss_type == 'bc':
            loss = ((policy_actions - noisy_expert_actions)**2).mean()
        elif loss_type == 'diff':
            denoised_actions = denoiser(expert_states, noisy_expert_actions)
            loss = ((denoised_actions - expert_actions)**2).mean()
        elif loss_type == 'combined':
            bc_loss = ((policy_actions - noisy_expert_actions)**2).mean()
            denoised_actions = denoiser(expert_states, noisy_expert_actions)
            diffusion_loss = ((denoised_actions - expert_actions)**2).mean()
            loss = bc_loss + diffusion_weight * diffusion_loss
        else:
            raise ValueError("Invalid loss_type. Choose from 'bc', 'diff', or 'combined'.")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f'Loss Type: {loss_type}, Epoch: {epoch:03d}, Loss: {loss.item():.4f}')
    
    filename = f".research/iteration1/images/training_loss_{loss_type}.pdf"
    plt.figure()
    plt.plot(losses, label=f"{loss_type} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Experiment2: Loss Curve ({loss_type})")
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    
    print(f"Experiment 2 ({loss_type}) completed. Loss curve saved as training_loss_{loss_type}.pdf")
    return policy, denoiser

def train_adaptive_noise_experiment(use_adaptive=False, diffusion_weight=1.0, epochs=50):
    """
    Experiment 3: Evaluating Adaptive Noise Schedules
    """
    mode = "Adaptive" if use_adaptive else "Fixed"
    print(f"\nRunning Experiment 3: Adaptive Noise Schedule ({mode})")
    
    state_dim, action_dim = 10, 3
    policy = Policy(state_dim, action_dim)
    denoiser = DiffusionDenoiser(state_dim, action_dim)
    optimizer = optim.Adam(list(policy.parameters()) + list(denoiser.parameters()), lr=1e-3)
    
    expert_states = torch.randn(1000, state_dim)
    expert_actions = torch.randn(1000, action_dim)
    
    bc_losses = []
    diff_losses = []
    total_losses = []
    
    for epoch in range(epochs):
        if use_adaptive:
            noisy_expert_actions = corrupt_actions_adaptive(expert_actions, expert_states, base_std=0.1)
        else:
            noisy_expert_actions = corrupt_actions(expert_actions, noise_std=0.1)
        
        policy_actions = policy(expert_states)
        bc_loss = ((policy_actions - noisy_expert_actions)**2).mean()
        denoised_actions = denoiser(expert_states, noisy_expert_actions)
        diffusion_loss = ((denoised_actions - expert_actions)**2).mean()
        total_loss = bc_loss + diffusion_weight * diffusion_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        bc_losses.append(bc_loss.item())
        diff_losses.append(diffusion_loss.item())
        total_losses.append(total_loss.item())
        
        if epoch % 10 == 0:
            print(f'Mode: {mode}, Epoch: {epoch:03d}, BC Loss: {bc_loss.item():.4f}, Diffusion Loss: {diffusion_loss.item():.4f}')
    
    filename = f".research/iteration1/images/training_loss_{mode.lower()}.pdf"
    plt.figure()
    plt.plot(total_losses, label="Total Loss")
    plt.plot(bc_losses, label="BC Loss")
    plt.plot(diff_losses, label="Diffusion Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Experiment3: Loss Curves ({mode} Noise)")
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    
    print(f"Experiment 3 ({mode} Noise) completed. Loss curve saved as training_loss_{mode.lower()}.pdf")
    return policy, denoiser
