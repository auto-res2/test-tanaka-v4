#!/usr/bin/env python3
"""
Main experiment runner for RDEIL (Robust Diffusion-Enhanced Imitation Learning)
Orchestrates the complete experimental pipeline from data preprocessing to evaluation.
"""

import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import prepare_experiment_data
from train import train_experiment1, train_ablation_experiment, train_adaptive_noise_experiment
from evaluate import run_evaluation_suite, compare_models, generate_evaluation_plots

def run_tests():
    """
    Run quick tests on all three experiments with reduced epochs to verify functionality.
    """
    print("========== Running Quick Tests ==========")
    print("Testing RDEIL implementation with reduced epochs for verification...")
    
    print("\n" + "="*60)
    policy1, denoiser1 = train_experiment1(epochs=10)
    
    print("\n" + "="*60)
    policy2_bc, denoiser2_bc = train_ablation_experiment(loss_type='bc', epochs=10)
    policy2_diff, denoiser2_diff = train_ablation_experiment(loss_type='diff', epochs=10)
    policy2_combined, denoiser2_combined = train_ablation_experiment(loss_type='combined', epochs=10)
    
    print("\n" + "="*60)
    policy3_fixed, denoiser3_fixed = train_adaptive_noise_experiment(use_adaptive=False, epochs=10)
    policy3_adaptive, denoiser3_adaptive = train_adaptive_noise_experiment(use_adaptive=True, epochs=10)
    
    print("\n" + "="*60)
    print("All tests finished successfully. Check standard output and generated PDF plots for details.")
    
    return {
        'experiment1': (policy1, denoiser1),
        'experiment2_bc': (policy2_bc, denoiser2_bc),
        'experiment2_diff': (policy2_diff, denoiser2_diff),
        'experiment2_combined': (policy2_combined, denoiser2_combined),
        'experiment3_fixed': (policy3_fixed, denoiser3_fixed),
        'experiment3_adaptive': (policy3_adaptive, denoiser3_adaptive)
    }

def run_full_experiments():
    """
    Run the complete RDEIL experimental suite with full epochs.
    """
    print("========== Running Full RDEIL Experiments ==========")
    
    print("\nStep 1: Data Preparation")
    data_dict = prepare_experiment_data(num_samples=1000, state_dim=10, action_dim=3)
    
    print("\nStep 2: Experiment 1 - Robustness in Noisy Expert Demonstrations")
    policy1, denoiser1 = train_experiment1(noise_std=0.1, diffusion_weight=1.0, epochs=100)
    
    print("\nStep 3: Experiment 2 - Ablation Study on Dual Loss Components")
    policy2_bc, denoiser2_bc = train_ablation_experiment(loss_type='bc', epochs=100)
    policy2_diff, denoiser2_diff = train_ablation_experiment(loss_type='diff', epochs=100)
    policy2_combined, denoiser2_combined = train_ablation_experiment(loss_type='combined', epochs=100)
    
    print("\nStep 4: Experiment 3 - Evaluating Adaptive Noise Schedules")
    policy3_fixed, denoiser3_fixed = train_adaptive_noise_experiment(use_adaptive=False, epochs=100)
    policy3_adaptive, denoiser3_adaptive = train_adaptive_noise_experiment(use_adaptive=True, epochs=100)
    
    print("\nStep 5: Model Evaluation and Comparison")
    test_states, test_actions = run_evaluation_suite()
    
    models_dict = {
        'Experiment1_RDEIL': (policy1, denoiser1),
        'Experiment2_BC_Only': (policy2_bc, denoiser2_bc),
        'Experiment2_Diff_Only': (policy2_diff, denoiser2_diff),
        'Experiment2_Combined': (policy2_combined, denoiser2_combined),
        'Experiment3_Fixed_Noise': (policy3_fixed, denoiser3_fixed),
        'Experiment3_Adaptive_Noise': (policy3_adaptive, denoiser3_adaptive)
    }
    
    results = compare_models(models_dict, test_states, test_actions)
    generate_evaluation_plots(results)
    
    print("\n" + "="*60)
    print("Full experimental suite completed successfully!")
    
    return results

def update_research_status(status="stopped"):
    """
    Update the research_history.json file with the current status.
    """
    research_file = ".research/research_history.json"
    
    research_data = {
        "experiment_name": "RDEIL - Robust Diffusion-Enhanced Imitation Learning",
        "status_enum": status,
        "timestamp": datetime.now().isoformat(),
        "experiments_completed": [
            "Experiment 1: Robustness in Noisy Expert Demonstrations",
            "Experiment 2: Ablation Study on Dual Loss Components", 
            "Experiment 3: Evaluating Adaptive Noise Schedules"
        ],
        "results_location": ".research/iteration1/images/",
        "models_saved": "models/",
        "data_location": "data/"
    }
    
    try:
        with open(research_file, 'w') as f:
            json.dump(research_data, f, indent=2)
        print(f"Research status updated to '{status}' in {research_file}")
    except Exception as e:
        print(f"Warning: Could not update research status: {e}")

def main():
    """
    Main function to run the RDEIL experiments.
    """
    print("RDEIL (Robust Diffusion-Enhanced Imitation Learning) Experiment Suite")
    print("="*70)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running in TEST mode with reduced epochs...")
        models = run_tests()
    else:
        print("Running FULL experiments...")
        results = run_full_experiments()
    
    update_research_status("stopped")
    
    print("\n" + "="*70)
    print("RDEIL experiments completed successfully!")
    print("Check .research/iteration1/images/ for generated plots")
    print("Check data/ for processed datasets")
    print("Check models/ for saved model checkpoints")

if __name__ == '__main__':
    main()
