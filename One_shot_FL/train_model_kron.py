import copy
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
import numpy as np
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector
from laplace import Laplace
from laplace.curvature import AsdlGGN,AsdlEF,AsdlHessian
from importlib.util import find_spec

from utils.compute_accuracy import test_img, test_img_ensemble
from algs.utils_mixtures import instantiate_model



def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()



def perturb_model_initialization(model, std=0.001):
    """
    Perturbs the initialization of a model by adding small Gaussian noise to its parameters.

    Args:
        model (nn.Module): The neural network model.
        std (float): Standard deviation of the Gaussian noise. Default is 0.01.

    Returns:
        None: Modifies the model parameters in-place.
    """
    device = next(model.parameters()).device

    for param in model.parameters():
        param.data += torch.randn(param.size(),device=device) * std


def initialize_model_with_tensor(model, array):
    """
    Initialize a PyTorch model's parameters using values from a given tensor.

    Args:
        model (torch.nn.Module): The PyTorch model.
        array (torch.Tensor): The tensor containing values to initialize the model with.
            The shape of `array` should match the concatenated shape of the model's parameters.

    Raises:
        ValueError: If the shape of `array` does not match the concatenated shape of the model's parameters.

    Example:
        model = torch.nn.Linear(10, 5)  # Example model with Linear layer
        array = torch.randn(15)  # Example tensor of shape (15,)
        initialize_model_with_tensor(model, array)  # Initialize model with the tensor
    """
    index = 0
    for param in model.parameters():
        numel = param.numel()  # Number of elements in the parameter tensor
        if index + numel > array.numel():
            raise ValueError("Tensor size does not match the total number of elements in model parameters.")
        
        # Reshape and copy values from the tensor to the parameter tensor
        param.data.copy_(array[index:index+numel].view(param.data.shape))
        index += numel

    if index != array.numel():
        raise ValueError("Tensor size does not match the total number of elements in model parameters.")






class LocalUpdate_mixtures_kron:
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    def train_and_compute_hessian_kron(
        self, ensemble, initialization, temperature,cov,
        random_init=False, perturbation=False, std=0.001,
        n_locals=1
    ):
        """
        Trains multiple local models and computes their approximate Kronecker-factored Hessians.
        
        Args:
            ensemble (List[nn.Module]): List of model copies.
            initialization (List[Tensor]): Initial parameter vectors.
            temperature (float): Temperature for Laplace approximation.
            random_init (bool): If True, reinitialize weights randomly.
            perturbation (bool): If True, apply small perturbations to initial weights.
            std (float): Standard deviation for perturbation.
            cov (float): Prior covariance for the Laplace approximation.
            n_locals (int): Number of local replicas per ensemble model.

        Returns:
            param (List[Tensor]): Flattened parameter vectors.
            kfac (List[Laplace]): Laplace approximations.
            nets (List[nn.Module]): Trained networks.
            mean_diversity (float): Average cosine similarity across ensemble members.
        """
        param, kfac, nets = [], [], []
        k = 0  # Counter for ensemble elements

        for net in ensemble:
            init_vector = initialization[k]
            k += 1

            for local_idx in range(n_locals):
                print(f"Perturbation: {perturbation}, STD: {std}")
                initialize_model_with_tensor(net, init_vector)

                if len(ensemble) > 1:
                    print(f"\nEnsemble {k} Local {local_idx}")

                # Initialization options
                if random_init:
                    net.apply(weight_reset)
                    print("Random Init:", parameters_to_vector(net.parameters()))
                elif perturbation:
                    print("Before Perturb:", parameters_to_vector(net.parameters()))
                    perturb_model_initialization(net, std)
                    print("Perturbed Init:", parameters_to_vector(net.parameters()))
                else:
                    print("Init:", parameters_to_vector(net.parameters()))

                # Train local model
                net.train()
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args['eta'], momentum=0.9)

                for epoch in range(self.args['local_epochs']):
                    batch_loss = []
                    for images, labels in self.ldr_train:
                        images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                        if self.args['augmentation']:
                            images = self.transform_train(images)

                        optimizer.zero_grad()
                        loss = self.loss_func(net(images), labels)
                        loss.backward()
                        optimizer.step()
                        batch_loss.append(loss.item())

                    print(f"Epoch {epoch} Loss: {np.mean(batch_loss):.4f}")

                # Fit Laplace approximation
                print(f"\nFinal Params: {parameters_to_vector(net.parameters())}")
                print(f"Temperature: {temperature}")

                hessian_structure = 'kron'
                print(f"\n--- HESSIAN --> {hessian_structure}, Prior Cov: {cov}\n")

                if hessian_structure == 'kron':
                    la = Laplace(
                        net, 'classification',
                        subset_of_weights='all',
                        hessian_structure=hessian_structure,
                        prior_mean=0,
                        prior_precision=1. / cov,
                        temperature=temperature
                    )
                else:  # Fallback to diagonal structure 
                    la = Laplace(
                        net, 'classification',
                        subset_of_weights='all',
                        hessian_structure='diag',
                        prior_mean=0,
                        prior_precision=1. / cov,
                        temperature=temperature,
                        backend=AsdlGGN
                    )

                la.fit(self.ldr_train)

                # Store results
                param.append(parameters_to_vector(net.parameters()))
                kfac.append(la)
                nets.append(net)

        # Compute average cosine similarity between all pairs
        diversity = 0.0
        for i in range(len(param)):
            for j in range(i + 1, len(param)):
                diversity += torch.nn.functional.cosine_similarity(param[i], param[j], dim=0).item()
        if len(param) == 1:
            mean_diversity = 0
        else:
            mean_diversity = diversity / ((len(param) * (len(param) - 1)) / 2)
        print(f"\nMean Diversity: {mean_diversity:.4f}\n")

        return param, kfac, nets, mean_diversity

    


def convert_dict_to_list(data_dict):
    result_list = []
    
    for key, value in data_dict.items():
        combined_tensors = []
        
        for tensor_data in value:
            combined_tensors.append(tensor_data) # Assuming the pattern involves absolute values
        
        result_list.append(combined_tensors)
    
    return result_list

    

def fl_setting_training(net_glob, net_glob_org, dataset_train, args, N_ENSEMBLE, starting_points, cov ,perturb):
    """
    Trains local models and returns model vectors, Laplace approximations, and metadata.

    Args:
        net_glob: Global model to clone from.
        net_glob_org: Original global model for instantiation.
        dataset_train: List of client datasets.
        args: Dictionary of training arguments.
        N_ENSEMBLE: Number of ensemble models per client.
        starting_points: List of initial weight vectors for each ensemble member.
        perturb: Whether to apply perturbation during training.

    Returns:
        model_map: Flattened list of all local model parameter vectors.
        la_flat: Flattened list of all Laplace approximations.
        models: List of trained client models.
        d: Total number of model parameters.
    """
    import copy
    from torch.nn.utils import parameters_to_vector

    model_vectors, la_list, models, starting_points_median = [], [], [], []
    net_glob.train()
    num_clients = len(dataset_train)

    for i in range(num_clients):
        print(f"\nTraining Local Model {i}")
        ensemble = [copy.deepcopy(net_glob) for _ in range(N_ENSEMBLE)]
        local = LocalUpdate_mixtures_kron(args=args, dataset=dataset_train[i])

        model_vector, la_post, model, diver = local.train_and_compute_hessian_kron(
            ensemble, starting_points, args['temperature'], cov,
            random_init=False, perturbation=perturb,
            std=0.001, n_locals=args['local_ens']
        )

        model_vectors.append(model_vector)
        la_list.append(la_post)
        models.append(model)

        for j, vec in enumerate(model_vector):
            acc, loss, _ = test_img(instantiate_model(vec, net_glob_org), dataset_train[i], args)
            print(f"Local Model {i} [{j}] Train Acc: {acc:.4f}, Loss: {loss:.4f}")

    model_map = [v for sublist in model_vectors for v in sublist]
    la_flat = [v for sublist in la_list for v in sublist]


    for i in range (len(starting_points)):
        starting_points_median.append(torch.median(torch.stack([t[i] for t in model_vectors]), dim=0).values )

    return model_map, la_flat, models,starting_points_median

def reset(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.reset_parameters()
