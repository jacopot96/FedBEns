import numpy as np
import random
import copy
import argparse
import csv
import torch
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import nn
from torch.utils.data import DataLoader

from data import get_dataset
from models import get_model
from train_model import LocalUpdate, LocalUpdate_ENSEMBLE_mixtures
from run_one_shot_algs import get_one_shot_model
from utils.compute_accuracy import test_img, test_img_ensemble
from algs.utils_mixtures import (
    mean_UN_CORRELATED_diag,
    compute_product_mixtures_diag,
    instantiate_model,
    DiagonalMixtureModel_versionB,
)

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
parser.add_argument('--n_mixtures', type=int, required=True, default=5)
parser.add_argument('--temperature', type=float, required=True, default=0.5)
parser.add_argument('--weights', type=str_to_bool, choices=[True, False], required=True, default=False)
parser.add_argument('--local_ens', type=int, required=True, default=1)
parser.add_argument('--val_at_server', type=str_to_bool, choices=[True, False], required=True, default=False)

args = parser.parse_args()

seed = args.seed
dataset = args.dataset
model_name = args.model
local_epochs = args.local_epochs
use_pretrained = args.use_pretrained
alpha = args.alpha
num_clients = args.num_clients
temperature = args.temperature
num_rounds = args.num_rounds
weights_mixtures = args.weights
N_ensemble_local = args.local_ens
val_at_server = args.val_at_server

print_every_test = 1
print_every_train = 1
N_ENSEMBLE = args.n_mixtures
COMPOSED = True

print('alpha', alpha)
print(model_name)
print('val_at_server', val_at_server)

filename = f"{num_clients}one_shot_ENS_DIAGF{N_ENSEMBLE}__{seed}_alpha_{alpha}T{temperature}_{val_at_server}_{dataset}_{model_name}_{local_epochs}"
filename_csv = f"results_40clientsDF/{filename}.csv"

n_c = 100 if dataset == 'CIFAR100' else 43 if dataset == 'GTSRB' else 10
dict_results = {}

algs_to_run = 'a'
for alg in algs_to_run:
    print("Running algorithm", alg)
    print("Using pre-trained model:", use_pretrained)

    np.random.seed(3)
    dataset_train, dataset_train_global, dataset_test_global, net_cls_counts = get_dataset(dataset, num_clients, n_c, alpha, True)
    test_loader = DataLoader(dataset_test_global, batch_size=len(dataset_test_global))

    ind = np.random.choice(len(dataset_train_global), 500)
    dataset_val = torch.utils.data.Subset(dataset_train_global, ind)

    args_dict = {
        "bs": 64,
        "local_epochs": local_epochs,
        "device": 'cuda',
        "rounds": num_rounds,
        "num_clients": num_clients,
        "augmentation": False,
        "eta": 0.01,
        "dataset": dataset,
        "model": model_name,
        "use_pretrained": use_pretrained,
        "n_c": n_c
    }

    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    net_glob_org = get_model(args_dict['model'], n_c, bias=False, use_pretrained=use_pretrained).to(args_dict['device'])
    net_glob = copy.deepcopy(net_glob_org)
    initial_vector = parameters_to_vector(net_glob.parameters())

    p = np.array([len(dataset_train[i]) for i in range(len(dataset_train))], dtype=float)
    p /= p.sum()

    def reset(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.reset_parameters()

    model_vectors, models, precisions, last_precisions, evidence_list, diversity = [], [], [], [], [], []

    for t in range(args_dict['rounds']):
        args_dict['augmentation'] = dataset in ('CIFAR10', 'CIFAR100', 'CINIC10', 'GTSRB')
        if use_pretrained:
            args_dict['eta'] = 0.001

        ensemble = [copy.deepcopy(net_glob.apply(reset)) for _ in range(N_ENSEMBLE)]
        starting_points = [parameters_to_vector(net.parameters()) for net in ensemble]
        perturb = N_ensemble_local > 1

        for i in range(len(dataset_train)):
            print(f"\nTraining Local Model {i}")
            net_glob.train()
            ensemble = [copy.deepcopy(net_glob) for _ in range(N_ENSEMBLE)]
            local = LocalUpdate_ENSEMBLE_mixtures(args=args_dict, dataset=dataset_train[i])
            model_vector, model, F_diag, prec_last, evidence, diver = local.train_and_compute_fisher(
                ensemble, starting_points, temperature,
                random_init=False, perturbation=perturb, std=0.001,
                composed=COMPOSED, n_locals=N_ensemble_local)
            model_vectors.append(model_vector)
            models.append(model)
            precisions.append(F_diag)
            last_precisions.append(prec_last)
            evidence_list.append(evidence)
            diversity.append(diver)

            for idx in range(min(5, len(model_vector))):
                acc, loss, _ = test_img(instantiate_model(model_vector[idx], net_glob_org), dataset_test_global, args_dict)
                print(f"Local Model {i}, Test Acc. {acc}, Test Loss {loss}")

    prior_mean = torch.zeros_like(initial_vector).to(args_dict['device'])
    prior_cov = torch.ones_like(initial_vector).to(args_dict['device']) * 0.1

    model_map = [item for sublist in model_vectors for item in sublist]
    precisions = [item for sublist in precisions for item in sublist]
    last_precisions = [item for sublist in last_precisions for item in sublist]

    if not COMPOSED:
        last_precisions = None

    if not weights_mixtures:
        l = N_ENSEMBLE * N_ensemble_local
        evidence_list = torch.log(torch.ones(len(model_map)) / l).to(args_dict['device'])
    else:
        evidence_list = [item for sublist in evidence_list for item in sublist]

    global_posterior = DiagonalMixtureModel_versionB(
        model_map, precisions, prior_mean, prior_cov,
        evidence_list, num_clients, N_ENSEMBLE * N_ensemble_local,
        COMPOSED, last_precisions, val_at_server
    )

    starting_points = [torch.median(torch.stack([t[i] for t in model_vectors]), dim=0).values for i in range(len(starting_points))]
    W_hats = [global_posterior.optimize(0.01, 500, sp, net_glob_org, dataset_val, args_dict) for sp in starting_points]

    models = [instantiate_model(W_hat, net_glob_org) for W_hat in W_hats]

    for i, model in enumerate(models):
        acc, loss, _ = test_img(model, dataset_test_global, args_dict)
        print(f"Test Acc {i}: {acc}, Test Loss: {loss}")

    acc, loss, f1, _ = test_img_ensemble(dataset_test_global, args_dict, models)
    print(f"ENSEMBLE {acc}, loss {loss}, f1m {f1}")

    dict_results[f'{alg}_test_loss_{seed}_{t}'] = loss
    dict_results[f'{alg}_test_acc_{seed}_{t}'] = acc

with open(filename_csv, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in dict_results.items():
        writer.writerow([key, value])
