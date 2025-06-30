import numpy as np
import random
import copy
import argparse
import csv
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from data import get_dataset
from models import get_model
from train_model import LocalUpdate, LocalUpdate_ENSEMBLE_mixtures
from train_model_kron import LocalUpdate_mixtures_kron, fl_setting_training, reset
from run_one_shot_algs import get_one_shot_model
from utils.compute_accuracy import test_img, test_img_ensemble
from algs.utils_mixtures import instantiate_model, DiagonalMixtureModel_versionB
from algs.ultils_mixtures_kron import MixtureModel_kron_nb


def str_to_bool(s):
    return s.lower() in ('true', '1')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--num_clients', type=int, default=5)
parser.add_argument('--num_rounds', type=int, default=1)
parser.add_argument('--local_epochs', type=int, default=30)
parser.add_argument('--use_pretrained', type=bool, default=False)
parser.add_argument('--n_mixtures', type=int, required=True)
parser.add_argument('--temperature', type=float, required=True)
parser.add_argument('--weights', type=str_to_bool, choices=[True, False], required=True)
parser.add_argument('--local_ens', type=int, required=True)
parser.add_argument('--val_at_server', type=str_to_bool, choices=[True, False], required=True)

args = parser.parse_args()
args = vars(args)  # Convert to dictionary for easy use

# Constants and setup
torch.manual_seed(args['seed'])
random.seed(args['seed'])
torch.backends.cudnn.deterministic = True
np.random.seed(3)

n_c = {'CIFAR100': 100, 'GTSRB': 43}.get(args['dataset'], 10)
args.update({"n_c": n_c, "bs": 64, "device": 'cuda', "augmentation": False, "eta": 0.01})
filename = f"{args['num_clients']}one_shot_ENS_KRON{args['n_mixtures']}__{args['seed']}_alpha_{args['alpha']}T{args['temperature']}_{args['val_at_server']}_{args['dataset']}_{args['model']}_{args['local_epochs']}"
filename_csv = f"results/{filename}.csv"

print(f"alpha: {args['alpha']}\nmodel: {args['model']}\nval_at_server: {args['val_at_server']}")
print(f"Using pre-trained model: {args['use_pretrained']}")

# Load data
dataset_train, dataset_train_global, dataset_test_global, _ = get_dataset(
    args['dataset'], args['num_clients'], n_c, args['alpha'], True)
dataset_val = torch.utils.data.Subset(dataset_train_global, np.random.choice(len(dataset_train_global), 500))

# Model
net_glob_org = get_model(args['model'], n_c, bias=False, use_pretrained=args['use_pretrained']).to(args['device'])
print('model', net_glob_org)

net_glob = copy.deepcopy(net_glob_org)
initial_vector = parameters_to_vector(net_glob.parameters())
n = len(dataset_train)
p = np.array([len(dataset_train[i]) for i in range(n)], dtype=float)
p /= p.sum()

N_ENSEMBLE = args['n_mixtures']
perturb = args['local_ens'] > 1

model_vectors, la_list, models, len_data = [], [], [], []

for t in range(args['num_rounds']):
    if args['dataset'] in ['CIFAR10', 'CIFAR100', 'CINIC10', 'GTSRB']:
        args['augmentation'] = True
    if args['use_pretrained']:
        args['eta'] = 0.001

    ensemble = [copy.deepcopy(net_glob).apply(reset) for _ in range(N_ENSEMBLE)]
    starting_points = [parameters_to_vector(net.parameters()) for net in ensemble]



# Prior Specification
cov = 0.1 #default
d = parameters_to_vector(net_glob_org.parameters()).numel()
prior_mean = torch.zeros(d, device=args['device'])
prior_cov = torch.full((d,), cov, device=args['device'])


# Clients Training
model_map, la_flat, models, starting_points_median = fl_setting_training(
    net_glob, net_glob_org, dataset_train, args, N_ENSEMBLE, starting_points, cov, perturb
)


if not args['weights']:
    l = N_ENSEMBLE * args['local_ens']
    evidence_list = torch.log(torch.ones(len(model_map)) / l).to(args['device'])
else:
    # Placeholder for evidence list (assumed computed during training if weights == True)
    evidence_list = torch.ones(len(model_map), device=args['device'])  

# Mixture model
global_posterior = MixtureModel_kron_nb(
    model_map, la_flat, models, args['device'], evidence_list,
    prior_mean, prior_cov, args['num_clients'], N_ENSEMBLE * args['local_ens'], args['val_at_server']
)

# Optimize and evaluate
W_hats = [
    global_posterior.optimize(0.01, 300, start, net_glob_org, dataset_val, args)
    for start in starting_points_median
]
final_models = [instantiate_model(W, copy.deepcopy(net_glob).apply(reset)) for W in W_hats]

# Logging
dict_results = {}
for i, model in enumerate(final_models):
    acc, loss, _ = test_img(model, dataset_test_global, args)
    print(f"Test Acc {i}: {acc:.4f}, Loss: {loss:.4f}")
    dict_results[f'local_model_test_accuracies_{args["alpha"]}_{i}'] = acc
    dict_results[f'local_model_test_losses_{args["alpha"]}_{i}'] = loss

acc, loss, f1, dis = test_img_ensemble(dataset_test_global, args, final_models)
print(f'ENSEMBLE Test Acc: {acc:.4f}, Loss: {loss:.4f}, Disagreement: {dis:.4f}')

dict_results.update({
    '-----------------------------------------': '-----------------------------------------',
    f'ens_KRON_test_loss_{args["seed"]}_{t}': loss,
    f'ens_KRON_test_acc_{args["seed"]}_{t}': acc,
})

with open(filename_csv, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(dict_results.items())
