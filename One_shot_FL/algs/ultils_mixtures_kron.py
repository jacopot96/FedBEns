import torch
import torch.nn as nn
import copy
import torch.distributions as D
import itertools
import numpy as np
from utils.compute_accuracy import test_img
from algs.utils_mixtures import instantiate_model
from torch.nn.utils import parameters_to_vector, vector_to_parameters
#import opt_einsum as oe
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector



class MixtureModel_kron_nb:
    def __init__(self, means,la, model, device, weights, mean_prior, cov_prior, n_clients, n_client_mixtures, val_at_server=False):
        """
        Initialize mixture on kronecker network with bias=False

        assume: -same number of mixtures per client
                -uniform per-mixture wegihts
        
        """
        self.means = [torch.as_tensor(mu, dtype=torch.float32) for mu in means]
        self.la = [la for la in la]
        self.num_components = len(means)
        self.val_at_server = val_at_server
        self.prior_mean = mean_prior
        self.prior_cov = cov_prior
        self.n_clients = n_clients
        self.num_client_mix = n_client_mixtures
        self.model = [mod for mod in model]
        self.val_at_server=val_at_server

        print('num clients',self.n_clients)
        print('num client mixtures',self.num_client_mix)


        self.prior = D.Independent(D.Normal(self.prior_mean, torch.sqrt(self.prior_cov)), 1)


        self.component_distributions = []
        for i in range(self.num_components):
            local = kronecker_network_nb( self.la[i],self.means[i], device)
            
            self.component_distributions.append(local)


    def negative_log_likelihood(self, point):
        """
        log-likelihood mixture kron network without bias

        -ingores terms not depending on the weights
        """
        log_probs = []
        for j in range(int(self.n_clients)):

            aux = []
            for i in range(int(self.num_client_mix)):

                index = int((j * self.num_client_mix) + i)

                log_prob = self.component_distributions[index].log_pdf_network(point)

                aux.append(log_prob)

            log_probs.append(torch.logsumexp(torch.stack(aux), dim=0))

        #negative_log_likelihood = -torch.logsumexp(torch.stack(log_probs), dim=0)
        negative_log_likelihood = - torch.sum(torch.stack(log_probs), dim=0)

        return negative_log_likelihood

    def negative_log_posterior(self, point):
        
        # -ingores terms not depending on the weights

        posterior_part = self.negative_log_likelihood(point)

        total = posterior_part + (self.n_clients - 1) * self.prior.log_prob(point)
        return total

    def optimize(self, l_r, n_steps, w_0, net, dataset_val, args):
        # Initialize the variable to optimize (parameter)
        x = torch.tensor(w_0, requires_grad=True)
        # Define the Adam optimizer
        optimizer = torch.optim.Adam([x], lr=0.001) #0.001
        test_acc_max = 0.0
        train_loss_min = float('inf')
        best_parameters = None

        # Create a deep copy of the network for validation
        net_glob_copy = copy.deepcopy(net)

        # Optimization loop
        for k in range(n_steps):
            # Zero gradients
            optimizer.zero_grad()

            # Compute the negative log-posterior (training loss)
            loss = self.negative_log_posterior(x)

            # Backpropagation
            loss.backward()

            # Gradient clipping to stabilize optimization
            torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Update best parameters based on training loss
            if loss < train_loss_min:  
                train_loss_min = loss.item()
                best_parameters_train = x.detach().clone()

            # Periodic validation
            if (k % 30 == 0 or k == n_steps - 1) :
                # Update the network with the optimized weights
                vector_to_parameters(x, net_glob_copy.parameters())

                # Evaluate on the validation set
                test_acc, test_loss, _ = test_img(net_glob_copy, dataset_val, args)

                # Track the best parameters based on validation accuracy
                if self.val_at_server==True:
                    if test_acc > test_acc_max:
                        test_acc_max = test_acc
                        best_parameters_val = x.detach().clone()

                # Logging
                print(f"Step {k}: Training Loss: {loss.item():.4f}, Validation Accuracy: {test_acc:.4f}, Validation Loss: {test_loss:.4f}")

        # Finalize best parameters: Prefer validation-based parameters, fallback to training-based

        if self.val_at_server==True :
            print("Best parameters selected based on validation accuracy.")
            return best_parameters_val
        else:
            print("Validation unavailable. Best parameters selected based on training loss.")
            return best_parameters_train


class kronecker_network_nb:
    def __init__(self,la,model_vector, device):
        # Initialize a Kronecker network without bias

        self.model = []
        #self.log_determinant_cov = - log_det_prec
        self.model_param = model_vector  # Ignore the bias
        self.la=la
        #self.log_det_network=la.posterior_precision.logdet()


    def log_pdf_network(self, x):
        """
        Compute the log PDF for the entire network by summing the log PDFs of each layer.
        No bias terms are used.
        """

        return self.la.log_prob(x)





class kronecker_layer_nb:
    def __init__(self, eigvals_A, eigvecs_A, eigvals_B, eigvecs_B, weights_values, delta, device):
        # Initialize a Kronecker layer without bias terms
        self.device = device
        #self.eigvecs_A = torch.nan_to_num(eigvecs_A.to(device),1e-5)
        #self.eigvecs_B = torch.nan_to_num(eigvecs_B.to(device),1e-5) #W = torch.nan_to_num(W)
        self.eigvals_A = torch.clamp(eigvals_A,min=1e-5,max=1e5)
        self.eigvals_B = torch.clamp(eigvals_B,min=1e-5,max=1e5)
        self.eigvecs_A, _ = torch.linalg.qr(torch.nan_to_num(eigvecs_A.to(device),1e-5))
        self.eigvecs_B, _ = torch.linalg.qr(torch.nan_to_num(eigvecs_B.to(device),1e-5))
        #self.eigvecs_A,_,_= torch.linalg.svd(torch.nan_to_num(eigvecs_A.to(device),1e-8))
        #self.eigvecs_B,_,_= torch.linalg.svd(torch.nan_to_num(eigvecs_B.to(device),1e-8))
        self.weights_values = weights_values.to(device)
        self.dim1 = eigvals_A.size(0)
        self.dim2 = eigvals_B.size(0)
        self.delta=delta

        self.eigvals_precision = (self.eigvals_A + torch.sqrt(self.delta)).unsqueeze(1) * (self.eigvals_B + torch.sqrt(self.delta)).unsqueeze(0)  # (d1, d2)
        #self.eigvals_precision = self.eigvals_A.unsqueeze(1) * self.eigvals_B.unsqueeze(0) + self.delta
        #self.eigvals_precision = torch.outer(self.eigvals_A, self.eigvals_B) + self.delta

    def log_pdf_kronecker_eigen(self, x,i):  # Contribution from the weights kron factorization
        
        # Check if input is batched or single sample
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Convert to 2D for single sample
        n_samples, d = x.shape

        diff = x - self.weights_values.to(x.device)  # (n_samples, d)

        # Reshape the data to reflect the Kronecker structure (n_samples, d1, d2)
        diff_reshaped = diff.view(n_samples, self.dim1, self.dim2)

        # Apply the eigenvector transformation separately for A and B
        #transformed_diff = self.eigvecs_A.T.to(x.device) @ diff_reshaped @ self.eigvecs_B.to(x.device)
        transformed_diff = torch.einsum('ij,njk,kl->nil', self.eigvecs_A.T.to(x.device), diff_reshaped, self.eigvecs_B.to(x.device))
        transformed_diff = transformed_diff.reshape(n_samples, d)
        # Compute the Kronecker product of the eigenvalues implicitly
        eigvals_precision = self.eigvals_precision.flatten().to(x.device)

        # Compute the quadratic form using the transformed data and eigenvalues
        quadratic_form = torch.sum((transformed_diff ** 2) * (eigvals_precision), dim=1) 
        
        #quadratic_form = torch.clamp(quadratic_form,1e-5)
        log_det = torch.sum(torch.log(eigvals_precision))

        #if i<=4:
        #    diag_prec = kron_to_diag(self.eigvals_A, self.eigvecs_A, self.eigvals_B, self.eigvecs_B,self.delta)
        #    quadratic_form = torch.sum((diff ** 2) * (diag_prec), dim=1)
        #    log_det = torch.sum(torch.log(diag_prec))


        return  quadratic_form,log_det



    def log_pdf_kronecker_eigen_layer(self, x,i):  # Full layer computation, no bias terms
        x_w = x[:self.dim1 * self.dim2].to(x.device)

        # Compute the log-pdf
        log_pdf_weights,log_det = self.log_pdf_kronecker_eigen(x_w,i)
        

        return log_pdf_weights.to(self.device),log_det.to(self.device)



def kron_to_diag(l1, Q1, l2, Q2,delta) -> torch.Tensor:
        """Extract diagonal of the entire decomposed Kronecker factorization.

        Parameters
        ----------
        exponent: float, default=1
            exponent of the Kronecker factorization

        Returns
        -------
        diag : torch.Tensor
        """

        eigval = torch.outer(l1, l2) + delta
        d = oe.contract("mp,nq,pq,mp,nq->mn", Q1, Q2, eigval, Q1, Q2).flatten()

        return torch.tensor(d)

def kron_to_matrix(l1, Q1, l2, Q2,delta) -> torch.Tensor:
        """Make the Kronecker factorization dense by computing the kronecker product.
        Warning: this should only be used for testing purposes as it will allocate
        large amounts of memory for big architectures.

        Parameters
        ----------
        exponent: float, default=1
            exponent of the Kronecker factorization

        Returns
        -------
        block_diag : torch.Tensor
        """
        Q = kron(Q1, Q2)
        delta_sqrt = torch.sqrt(delta)
        eigval = torch.outer(l1, l2) +delta
        eigval = torch.clamp(eigval, min=1e-6)

        L = torch.diag(eigval.flatten())

        blocks = Q @ L @ Q.T
        print('blocks check',is_positive_definite_eigen(blocks))
        blocks = (blocks + blocks.T) / 2
        print('blocks check2',is_positive_definite_eigen(blocks))
        return blocks

def kron(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Computes the Kronecker product between two tensors.

    Parameters
    ----------
    t1 : torch.Tensor
    t2 : torch.Tensor

    Returns
    -------
    kron_product : torch.Tensor
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
        .unsqueeze(3)
        .repeat(1, t2_height, t2_width, 1)
        .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def is_positive_definite_eigen(matrix: torch.Tensor) -> bool:
    """Check if a 2D matrix is positive definite by verifying that all eigenvalues are positive."""
    assert matrix.dim() == 2 and matrix.size(0) == matrix.size(1), "Input must be a square matrix"
    
    # Compute the eigenvalues
    eigvals = torch.linalg.eigvalsh(matrix)  # Use eigvalsh for symmetric (Hermitian) matrices
    
    # Check if all eigenvalues are positive
    return torch.all(eigvals > 0)

def extract_params_from_flat_array(flat_array, model):
    param = []  # To store parameters (weights)
    bias = []   # To store biases

    start = 0
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            # Conv2D layer weights: (out_channels, in_channels, kernel_height, kernel_width)
            weight_size = layer.weight.numel()  # Total number of elements in the weight tensor
            layer_weight = flat_array[start:start + weight_size]  # Reshape to Conv2D weight shape
            param.append(layer_weight)
            start += weight_size

            # Conv2D bias: (out_channels)
            if layer.bias is not None:
                bias_size = layer.bias.numel()  # Total number of elements in the bias tensor
                layer_bias = flat_array[start:start + bias_size]  # Reshape to bias shape
                bias.append(layer_bias)
                start += bias_size

        elif isinstance(layer, nn.Linear):
            # Linear layer weights: (out_features, in_features)
            weight_size = layer.weight.numel()  # Total number of elements in the weight tensor
            layer_weight = flat_array[start:start + weight_size]  # Reshape to Linear weight shape
            param.append(layer_weight)
            start += weight_size

            # Linear layer bias: (out_features)
            if layer.bias is not None:
                bias_size = layer.bias.numel()  # Total number of elements in the bias tensor
                layer_bias = flat_array[start:start + bias_size] # Reshape to bias shape
                bias.append(layer_bias)
                start += bias_size

    return param, bias



def get_dot_product(F_mat_list,w, N):
    n = len(F_mat_list)
    v = F_mat_list[0].mv(w)
    v = v.__rmul__(N)

    #v = v.__add__(x)
    return v


def element_mul(self, other):
        if self.dict_repr is not None and other.dict_repr is not None:
            v_dict = dict()
            for l_id, l in self.layer_collection.layers.items():
                if l.bias is not None:
                    v_dict[l_id] = (self.dict_repr[l_id][0]*
                                    other.dict_repr[l_id][0],
                                    self.dict_repr[l_id][1]*
                                    other.dict_repr[l_id][1])
                else:
                    v_dict[l_id] = (self.dict_repr[l_id][0]*
                                    other.dict_repr[l_id][0],)
            return PVector(self.layer_collection, dict_repr=v_dict)
        elif self.vector_repr is not None and other.vector_repr is not None:
            return PVector(self.layer_collection,
                           vector_repr=self.vector_repr+other.vector_repr)
        else:
            return PVector(self.layer_collection,
                           vector_repr=(self.get_flat_representation() +
                                        other.get_flat_representation()))