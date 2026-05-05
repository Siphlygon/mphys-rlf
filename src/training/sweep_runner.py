import wandb
import subprocess
import os
import string
import secrets


def launch():
    """
    Launches one DDP run using torchrun
    """
    host_ip = os.environ.get("HOST_IP", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12365")
    env = os.environ.copy()
    
    # Set the model name for this run to avoid FileExistsError in sweeps.py
    model_name = env["MODEL_NAME"]
    suffix = int(model_name.split("_")[-1])  # Get the current suffix number
    base_name = "sweep" + "_" + model_name.split("_")[1]  # Get the base name without the suffix
    new_model_name = f"{base_name}_{suffix}"
    while os.path.exists(f"model_results/{new_model_name}"):
        suffix += 1
        new_model_name = f"{base_name}_{suffix}"
    env["MODEL_NAME"] = new_model_name
    print(f"Set MODEL_NAME to {new_model_name} for this run.")
    
    # Launch the training script using torchrun for DDP
    subprocess.run([
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=2",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=" + host_ip + ":" + master_port,
        "src/training/sweeps.py"
    ], check=True, env=env)

if __name__ == "__main__":
    # Initialise wandb logging
    wandb.login(key=os.environ.get("WANDB_KEY"))
    project = "diffusion-radio-galaxies-sweeps"

    # Define the sweep configuration
    sweep_config = {
                'name': 'hyperparameter_sweep',
                'method': 'bayes',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'parameters': {
                    'dropout': {'distribution': 'uniform', 'min': 0.05, 'max': 0.2},
                    'batch_size': {'values': [16, 32, 64]},
                    'learning_rate': {'distribution': 'uniform', 'min': 1e-5, 'max': 4e-5},
                    'iterations': {'value': 20000},
                    'ema_rate': {'values': [0.999, 0.9999, 0.99999]},
                    'P_mean': {'distribution': 'uniform', 'min': -5, 'max': -1.25},
                    'P_std': {'distribution': 'uniform', 'min': 0.9, 'max': 3.6},
                    'context_dropout': {'distribution': 'uniform', 'min': 0.05, 'max': 0.2}
                },
                'early_terminate': {
                    'type': 'hyperband',
                    'min_iter': 1000
                }
            }

    # Run the sweep agent to execute the hyperparameter sweep
    if "WANDB_SWEEP_ID" in os.environ:
        sweep_id = os.environ["WANDB_SWEEP_ID"]
    else:
        sweep_id = wandb.sweep(sweep_config, project=project)
        print("Created sweep:", sweep_id)
    
    # Initialise model name to avoid FileExistsError in sweeps.py
    os.environ["MODEL_NAME"] = f"sweep_{sweep_id}_0"    

    wandb.agent(sweep_id, function=launch)