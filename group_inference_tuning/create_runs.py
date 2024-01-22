import subprocess

# Define lists of hyperparameter values
pretrained = True
infer_lr_values = [1e-3, 1e-4, 1e-5]
infer_weight_decay_values = [1e-2, 1e-1, 1]
infer_num_epochs_values = [1, 2]
batch_size = 64

# GPUs to distribute runs
gpu_devices = [0, 1, 2, 3]

# Calculate number of runs per GPU
runs_per_gpu = len(infer_lr_values) * len(infer_weight_decay_values) * len(infer_num_epochs_values) // len(gpu_devices)

# Create tmux sessions for each GPU
for gpu_device in gpu_devices:
    # Generate grid of commands for the current GPU
    for i in range(runs_per_gpu):
        session_name = f"gpu_{gpu_device}_{i}"
        
        idx = (i + gpu_device * runs_per_gpu) % len(infer_lr_values)
        infer_lr = infer_lr_values[idx]
        
        idx = ((i + gpu_device * runs_per_gpu) // len(infer_lr_values)) % len(infer_weight_decay_values)
        infer_weight_decay = infer_weight_decay_values[idx]
        
        idx = ((i + gpu_device * runs_per_gpu) // (len(infer_lr_values) * len(infer_weight_decay_values))) % len(infer_num_epochs_values)
        infer_num_epochs = infer_num_epochs_values[idx]
        
        command = f"python waterbirds_spare.py --pretrained={pretrained} --infer_lr={infer_lr} --infer_weight_decay={infer_weight_decay} --infer_num_epochs={infer_num_epochs} --batch_size={batch_size} --device {gpu_device}"

        # Construct the tmux command
        tmux_command = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            f'bash -c "cd /path/to/your/script && {command} ; read"'
        ]

        # Run the tmux command and detach from the session
        subprocess.run(tmux_command)

# Inform the user that the sessions have been started
print("Tmux sessions have been started.")
