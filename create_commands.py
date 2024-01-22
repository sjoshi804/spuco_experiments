import itertools

# Define the possible values for each argument
infer_lr_values = [1e-3, 1e-4, 1e-5]
infer_weight_decay_values = [1e-2, 1e-1, 1]
infer_num_epochs_values = [1, 2]
val_size_pct_values = [5, 15]

# Create all possible combinations of values
combinations = list(itertools.product(
    infer_lr_values,
    infer_weight_decay_values,
    infer_num_epochs_values,
    val_size_pct_values
))

# Define other fixed arguments
pretrained = "--pretrained"
batch_size = "--batch_size=64"

# Generate and print the command for each combination
for combo in combinations:
    infer_lr, infer_weight_decay, infer_num_epochs, val_size_pct = combo
    command = f"python waterbirds_eiil.py {pretrained} --infer_lr={infer_lr} --infer_weight_decay={infer_weight_decay} --infer_num_epochs={infer_num_epochs} {batch_size} --val-size-pct={val_size_pct}"
    print(command)
