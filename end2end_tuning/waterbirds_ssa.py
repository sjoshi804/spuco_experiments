import argparse
import os
import pickle 

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD
from wilds import get_dataset

from spuco.datasets import GroupLabeledDatasetWrapper, WILDSDatasetWrapper, SpuriousTargetDatasetWrapper
from spuco.evaluate import Evaluator
from spuco.group_inference import SSA
from spuco.robust_train import GroupDRO
from spuco.models import model_factory
from spuco.utils import Trainer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--val-size-pct", type=int, default=25)

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--results_csv", type=str, default="/home/sjoshi/spuco_experiments/end2end_tuning/results.csv")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1.0)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_lr", type=float, default=1e-5)
parser.add_argument("--infer_weight_decay", type=float, default=1e-1)
parser.add_argument("--infer_momentum", type=float, default=0.9)
parser.add_argument("--infer_num_iters", type=int, default=1000)
parser.add_argument("--infer_val_frac", type=float, default=0.5)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="waterbirds", download=True, root_dir=args.root_dir)

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

# Get the training set
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

subset_indices = None
if args.val_size_pct == 5:
    with open("/home/sjoshi/spuco_experiments/end2end_tuning/wbirds_5pct_val_set.pkl", "rb") as f:    
        subset_indices = pickle.load(f)
elif args.val_size_pct == 15:
    with open("/home/sjoshi/spuco_experiments/end2end_tuning/wbirds_5pct_val_set.pkl", "rb") as f:    
        subset_indices = pickle.load(f)
else:
    raise NotImplementedError(f"{args.val_size_pct} % val set size not supported")

trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="background", verbose=True, subset_indices=subset_indices)
testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)

model = model_factory("resnet50", trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

ssa = SSA(
    spurious_unlabeled_dataset=trainset,
    spurious_labeled_dataset=SpuriousTargetDatasetWrapper(valset, valset.spurious),
    model=model,
    labeled_valset_size=args.infer_val_frac,
    lr=args.infer_lr,
    weight_decay=args.infer_weight_decay,
    num_iters=args.infer_num_iters,
    tau_g_min=0.95,
    device=device,
    verbose=True
)

group_partition = ssa.infer_groups()
for key in sorted(group_partition.keys()):
    print(key, len(group_partition[key]))
evaluator = Evaluator(
    testset=trainset,
    group_partition=group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()

robust_trainset = GroupLabeledDatasetWrapper(trainset, group_partition)

valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)

train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=train_transform
)
trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)

group_dro = GroupDRO(
    model=model,
    val_evaluator=valid_evaluator,
    num_epochs=args.num_epochs,
    trainset=robust_trainset,
    valset=valset,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)
group_dro.train()

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
results = pd.DataFrame(index=[0])
results["timestamp"] = pd.Timestamp.now()
results["alg"] = "ssa"
results["seed"] = args.seed
results["pretrained"] = args.pretrained
results["lr"] = args.lr
results["weight_decay"] = args.weight_decay
results["momentum"] = args.momentum
results["num_epochs"] = args.num_epochs
results["batch_size"] = args.batch_size

results["infer_lr"] = args.infer_lr
results["infer_weight_decay"] = args.infer_weight_decay
results["infer_num_iters"] = args.infer_num_iters
results["infer_val_frac"] = args.infer_val_frac

results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["average_accuracy"] = evaluator.average_accuracy

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=group_dro.best_model,
    device=device,
    verbose=True
)
evaluator.evaluate()

results["early_stopping_worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["early_stopping_average_accuracy"] = evaluator.average_accuracy

if os.path.exists(args.results_csv):
    results_df = pd.read_csv(args.results_csv)
else:
    results_df = pd.DataFrame()

results_df = pd.concat([results_df, results], ignore_index=True)
results_df.to_csv(args.results_csv, index=False)

print('Done!')
print('Results saved to', args.results_csv)


