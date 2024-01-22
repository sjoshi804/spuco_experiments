import argparse
import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD
from wilds import get_dataset
import pickle 

from spuco.datasets import WILDSDatasetWrapper
from spuco.evaluate import Evaluator, GroupEvaluator
from spuco.group_inference import SpareInference
from spuco.robust_train import SpareTrain
from spuco.models import model_factory
from spuco.utils import set_seed, Trainer
from spuco.utils.misc import get_model_outputs

parser = argparse.ArgumentParser()
parser.add_argument("--only-inference", action="store_true")
parser.add_argument("--val-size-pct", type=int, default=15)

# Tuning
parser.add_argument("--infer_lr", type=float, default=1e-3)
parser.add_argument("--infer_weight_decay", type=float, default=1e-4)
parser.add_argument("--infer_num_epochs", type=int, default=1)

# Rest
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--results_csv", type=str, default="/home/sjoshi/spuco_experiments/end2end_tuning/group_recovery_metric/results.csv")

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_momentum", type=float, default=0.9)

parser.add_argument("--high_sampling_power", type=int, default=2)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="waterbirds", download=True, root_dir=args.root_dir)

train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

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
    with open("/home/sjoshi/spuco_experiments/end2end_tuning/wbirds_15pct_val_set.pkl", "rb") as f:    
        subset_indices = pickle.load(f)
else:
    raise NotImplementedError(f"{args.val_size_pct} % val set size not supported")

trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="background", verbose=True, subset_indices=subset_indices)
testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)

model = model_factory("resnet50", trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

trainer = Trainer(
    trainset=trainset,
    model=model,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.infer_lr, weight_decay=args.infer_weight_decay, momentum=args.infer_momentum),
    device=device,
    verbose=True
)

trainer.train(num_epochs=args.infer_num_epochs)

logits = trainer.get_trainset_outputs()
predictions = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
spare_infer = SpareInference(
    logits=predictions,
    class_labels=trainset.labels,
    device=device,
    num_clusters=2,
    high_sampling_power=args.high_sampling_power,
    verbose=True
)
group_partition = spare_infer.infer_groups()

val_predictions = torch.nn.functional.softmax(get_model_outputs(model, valset, device, verbose=True), dim=1).detach().cpu().numpy()
val_spare_infer = SpareInference(
    logits=val_predictions,
    class_labels=valset.labels,
    num_clusters=2,
    device=device,
    high_sampling_power=args.high_sampling_power,
    verbose=True
)
val_group_partition = val_spare_infer.infer_groups()

# METRICS
print("Creating compatible inferred group partition")    
inferred_group_partition = {}
if len(val_group_partition[(0,0)]) < len(val_group_partition[(0,1)]):
    inferred_group_partition[(0,0)] = val_group_partition[(0,1)]
    inferred_group_partition[(0,1)] = val_group_partition[(0,0)]
else:
    inferred_group_partition[(0,0)] = val_group_partition[(0,0)]
    inferred_group_partition[(0,1)] = val_group_partition[(0,1)]
if len(val_group_partition[(1,1)]) < len(val_group_partition[(1,0)]):
    inferred_group_partition[(1,1)] = val_group_partition[(1,0)]
    inferred_group_partition[(1,0)] = val_group_partition[(1,1)]
else:
    inferred_group_partition[(1,1)] = val_group_partition[(1,1)]
    inferred_group_partition[(1,0)] = val_group_partition[(1,0)]
    
group_evaluator = GroupEvaluator(
    inferred_group_partition=inferred_group_partition,
    true_group_partition=valset.group_partition,
    num_classes=2,
    verbose=True
)

sampling_powers = spare_infer.sampling_powers 

print("Sampling powers: {}".format(sampling_powers))
for key in sorted(group_partition.keys()):
    for true_key in sorted(trainset.group_partition.keys()):
        print("Inferred group: {}, true group: {}, size: {}".format(key, true_key, len([x for x in trainset.group_partition[true_key] if x in group_partition[key]])))

train_evaluator = Evaluator(
    testset=trainset,
    group_partition=group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
train_evaluator.evaluate()

val_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=valset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)

# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=train_transform
)
trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)

spare_train = SpareTrain(
    model=model,
    num_epochs=args.num_epochs,
    trainset=trainset,
    group_partition=group_partition,
    sampling_powers=sampling_powers,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    val_evaluator=val_evaluator,
    verbose=True
)
spare_train.train()
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
results["alg"] = "spare"
results["timestamp"] = pd.Timestamp.now()
args_dict = vars(args)
for key in args_dict.keys():
    results[key] = args_dict[key]

# Log metrics
results["train_worst_group_accuracy"] = train_evaluator.worst_group_accuracy[1]
results["train_average_accuracy"] = train_evaluator.average_accuracy
precision = group_evaluator.evaluate_precision()
recall = group_evaluator.evaluate_recall()
results["group_accuracy"] = group_evaluator.evaluate_accuracy()
results["group_avg_precision"] = precision[0]
results["group_min_precision"] = precision[1]
results["group_avg_recall"] = recall[0]
results["group_min_recall"] = recall[1]

# Log results
results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["average_accuracy"] = evaluator.average_accuracy

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=spare_train.best_model,
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