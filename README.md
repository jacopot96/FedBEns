# FedBEns

This repository contains the code for the paper:  
**"FedBEns: One-Shot Federated Learning based on Bayesian Ensemble"**

## Setup

- **Python version**: 3.12.8  
- Install the required packages using:
```bash
pip install -r requirements.txt
```

## Code Structure
This repository contains four main scripts to run both the proposed **FedBEns** algorithm and several baseline methods.

1) main_baselines.py : it runs the considered baselines, run it with:
   ```bash
   python main_baselines.py --dataset 'dataset_name' --model 'model_name' --num_clients n_clients --local_epochs n_epochs --alpha= alpha_value --seed s --algs_to_run 'alg_name'
   ```
   This script runs baseline methods including:
     
     - `fedfisher_kfac`
     - `otfusion`
     - `fishermerge`
     - `dense`
     - `regmean`
     
     Supported models:
     
     - `LeNet`
     - `CNN`
     - `ResNet18`
     
     Supported datasets:
     
     - `FashionMNIST`
     - `SVHN`
     - `CIFAR10`
     - `CIFAR100`
     - 
   Usage example:
   ```
   python main_baselines.py --dataset 'FashionMNIST' --model 'LeNet' --num_clients 5 --local_epochs 20 --alpha=0.1 --seed 1000 --algs_to_run 'fedfisher_kfac'
   ```
2) main_FedBEns_kron.py : it runs the FedBENS algorithm with Kronecker factorization of the Hessian. The command follows the previous logic. As an exemple:
   ```
   python3 main_FedBEns_kron.py  --dataset "FashionMNIST" --model "LeNet" --seed=42 --local_epochs=20 --alpha=0.1 --temperature=0.1 --num_clients=5 --n_mixtures 3 --weights False --local_ens=1 --val_at_server True 

   ```


different files executes different models:

1) main_baselines.py : it runs the considered baselines, run it with:

python main_baselines.py --dataset 'dataset_name' --model 'model_name' --num_clients n_clients --local_epochs n_epochs --alpha= alpha_value --seed s --algs_to_run 'alg_name' 

algs_to_run: 'fedfisher_kfac', 'otfusion', 'fishermerge', 'dense', 'regmean'
model: 'LeNet', 'CNN', 'ResNet18'
dataset: 'FashionMNIST', 'SVHN','CIFAR10', 'CIFAR100'

e.g., : python main_baselines.py --dataset 'FashionMNIST' --model 'LeNet' --num_clients 5 --local_epochs 20 --alpha=0.1 --seed 1000 --algs_to_run 'fedfisher_kfac'

2) main_FedBEns_kron.py : it runs the FedBENS algorithm with Kronecker factorization of the Hessian. The command follows the previous logic. As an exemple:

python3 main_FedBEns_kron.py  --dataset "FashionMNIST" --model "LeNet" --seed=666 --local_epochs=20 --alpha=0.1 --temperature=0.1 --num_clients=5 --n_mixtures 3 --weights False --local_ens=1 --val_at_server True 

3) main_FedBEns_diag.py : as 2) but with diagonal Hessian

4) main_FedBEns_diag_full.py : as 2) but with diagonal hessian+full Hessian for the last layer

----------------------------------------------------------------------------------------------
Code relies heavely on the repository associated with the paper : "FedFisher: Leveraging Fisher Information for One-Shot Federated Learning" by Divyansh Jhunjhunwala, Shiqiang Wang, and Gauri Joshi, published in AISTATS 2024.
