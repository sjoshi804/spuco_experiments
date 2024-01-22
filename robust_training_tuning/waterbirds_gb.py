import argparse
import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD
from wilds import get_dataset
from PIL import Image
import numpy as np 

from spuco.datasets import WILDSDatasetWrapper
from spuco.evaluate import Evaluator, GradCamEvaluator
from spuco.robust_train import GroupBalanceBatchERM
from spuco.models import model_factory
from spuco.utils import set_seed

parser = argparse.ArgumentParser()

# Tuning
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--results_csv", type=str, default="/home/sjoshi/spuco_experiments/robust_training_tuning/results.csv")

parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="waterbirds", download=True, root_dir=args.root_dir)

base_transform =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

# Load masks
split_mask = dataset.split_array == dataset.split_dict["val"]
val_split = np.where(split_mask)[0]
val_filenames = [dataset._input_array[i].replace(".jpg", ".png") for i in val_split]
masks = [np.array(base_transform(Image.open(f"/data/waterbirds_v1.0/segmentations/{filename}").convert('L')))/255 for filename in val_filenames]

# Load data
train_data = dataset.get_subset(
    "train",
    transform=transform
)

val_data = dataset.get_subset(
    "val",
    transform=transform
)

# Get the training set
test_data = dataset.get_subset(
    "test",
    transform=transform
)
trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="background", verbose=True)
testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)

# Load model
model = model_factory("resnet50", trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

# Initialize robust trainer
robust_trainer = GroupBalanceBatchERM(
    model=model,
    group_partition=trainset.group_partition,
    num_epochs=args.num_epochs,
    trainset=trainset,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)

curr_results_df = pd.DataFrame()
for epoch in range(args.num_epochs):
    # Train
    robust_trainer.train_epoch(epoch)
    
    if (epoch + 1) % 10 != 0:
        continue
    
    # Initialize results log
    results = pd.DataFrame(index=[0])
    results["alg"] = "gb"
    results["timestamp"] = pd.Timestamp.now()
    args_dict = vars(args)
    for key in args_dict.keys():
        if key == args.num_epochs:
            continue
        results[key] = args_dict[key]
    results["num_epochs"] = epoch + 1

    # Evaluate and log
    gradcam_evaluator = GradCamEvaluator(
        model=model, 
        dataset=valset, 
        masks=masks,
        device=device,
        verbose=True
    )
    results["val_gradcam_iou"] = gradcam_evaluator.evaluate()
    
    val_evaluator = Evaluator(
        testset=valset,
        group_partition=valset.group_partition,
        group_weights=valset.group_weights,
        batch_size=args.batch_size,
        model=model,
        device=device,
        verbose=True
    )
    val_evaluator.evaluate()
    results["val_worst_group_accuracy"] = val_evaluator.worst_group_accuracy[1]
    results["val_avg_min_group_accuracy"] = np.min([val_evaluator.accuracies[(0,1)], val_evaluator.accuracies[(1,0)]])
    results["val_average_accuracy"] = val_evaluator.average_accuracy
    
    evaluator = Evaluator(
        testset=testset,
        group_partition=testset.group_partition,
        group_weights=trainset.group_weights,
        batch_size=args.batch_size,
        model=model,
        device=device,
        verbose=True
    )
    
    evaluator.evaluate()
    results["test_spurious_attribute_prediction"] = evaluator.evaluate_spurious_attribute_prediction()
    results["test_worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
    results["test_average_accuracy"] = evaluator.average_accuracy
    
    # Append to current results
    print(results)
    curr_results_df = pd.concat([curr_results_df, results], ignore_index=True)

if os.path.exists(args.results_csv):
    results_df = pd.read_csv(args.results_csv)
else:
    results_df = pd.DataFrame()

results_df = pd.concat([results_df, curr_results_df], ignore_index=True)
results_df.to_csv(args.results_csv, index=False)

print('Done!')
print('Results saved to', args.results_csv)