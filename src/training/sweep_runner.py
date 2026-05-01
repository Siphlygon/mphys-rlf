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
    
    # Generate a new WANDB_RUN_ID for each run to ensure proper logging
    alphanumeric = string.ascii_lowercase + string.digits
    random = "".join(secrets.choice(alphanumeric) for _ in range(6))
    env["WANDB_RUN_ID"] = f"{random}"
    
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
                    'batch_size': {'values': [16, 32, 64, 128]},
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
    wandb.agent(sweep_id, function=launch)