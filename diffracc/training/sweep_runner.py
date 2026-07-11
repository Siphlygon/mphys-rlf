"""
A script to run a hyperparameter sweep using Weights & Biases (wandb) and torchrun for distributed training. The script
reads the sweep parameters from a JSON file, constructs the sweep configuration, and launches the training script for
each set of hyperparameters. The script also handles the initialization of the model name to avoid file existence errors
during the sweep.
"""
import json
import os
import subprocess

import wandb


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
        "-m diffracc.training.sweeps"
    ], check=True, env=env)


if __name__ == "__main__":
    # Initialise wandb logging#
    if "WANDB_API_PATH" in os.environ and os.environ["WANDB_API_PATH"]:
        wandb_api_path = os.environ["WANDB_API_PATH"]
        if os.path.isfile(wandb_api_path):
            with open(wandb_api_path, "r", encoding="utf-8") as f:
                wandb_api_key = f.read().strip()
            os.environ["WANDB_API_KEY"] = wandb_api_key
    wandb.login(key=os.environ.get("WANDB_KEY"))
    PROJECT = "diffusion-radio-galaxies-sweeps"

    # Define the sweep configuration
    PARAMETERS_PATH = "diffracc/training/sweep_params.json"
    if os.path.exists(PARAMETERS_PATH):
        try:
            with open(PARAMETERS_PATH, 'r', encoding='utf-8') as f:
                parameters = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {PARAMETERS_PATH}") from exc
    else:
        raise FileNotFoundError(f"Sweep parameters file not found at {PARAMETERS_PATH}")

    parameters = parameters.get("parameters")
    print("Loaded sweep parameters:", parameters)

    sweep_config = {
                'name': 'hyperparameter_sweep',
                'method': 'bayes',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'parameters': parameters,
                'early_terminate': {
                    'type': 'hyperband',
                    'min_iter': 1000
                }
            }
    print("Constructed sweep configuration:", sweep_config)

    # Run the sweep agent to execute the hyperparameter sweep
    if "WANDB_SWEEP_ID" in os.environ:
        sweep_id = os.environ["WANDB_SWEEP_ID"]
    else:
        sweep_id = wandb.sweep(sweep_config, project=PROJECT)
        print("Created sweep:", sweep_id)

    # Initialise model name to avoid FileExistsError in sweeps.py
    os.environ["MODEL_NAME"] = f"sweep_{sweep_id}_0"

    wandb.agent(sweep_id, function=launch)
