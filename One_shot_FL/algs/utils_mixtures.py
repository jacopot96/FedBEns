import torch
import copy
import torch.distributions as D
import itertools
import numpy as np
from utils.compute_accuracy import test_img
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class DiagonalMixtureModel_versionB:
    def __init__(self, means, prec, mean_prior, cov_prior,weights,n_clients,n_client_mixtures,composed=False,prec_last=None,val_at_server=False):
        """
        Initialize a diagonal mixture model with given means and diagonal covariance matrices.

        Args:
            means (list of torch.Tensor): List of mean vectors for each Gaussian component.
            covs (list of torch.Tensor): List of diagonal covariance matrices for each Gaussian component.
        """
        self.means = [torch.as_tensor(mu, dtype=torch.float32) for mu in means]
        self.covs = [torch.clamp(1./torch.as_tensor(pr, dtype=torch.float32),min=1e-6,max=1e6) for pr in prec]
        #self.log_weights = weights 
        self.num_total_components = len(means)
        self.num_client_mix=n_client_mixtures
        self.prior_mean = mean_prior
        self.prior_cov = cov_prior
        self.n_clients = n_clients
        #self.log_weights = self.num_total_components*np.log((1./self.num_client_mix))
        self.log_weights = [torch.tensor(i, dtype=torch.float32) for i in weights]

        self.composed=composed
        self.val_at_server = val_at_server



        # Precompute distributions for each component
        self.prior = D.Independent(D.Normal(self.prior_mean, torch.sqrt(self.prior_cov)), 1)

        self.component_distributions = []

        if composed==False:
            for i in range(self.num_total_components):
                mvn_diagonal = D.Independent(D.Normal(self.means[i], torch.sqrt(self.covs[i])), 1)
                self.component_distributions.append(mvn_diagonal)

        else:
            print()
            print('---COMPOSED---')
            print()
            self.last_prec = [cv for cv in prec_last]


            self.leng_last = self.last_prec[0].shape[0]

            self.component_distributions_firsts = []
            self.component_distributions_last = []
            for i in range(self.num_total_components):
                # Slice means and covs to remove the last `leng_last` elements
                truncated_mean = self.means[i][:-self.leng_last]
                truncated_cov = self.covs[i][:-self.leng_last]
                print('Laplace cov diag',truncated_cov)
                print('median eigenvalue',torch.median(truncated_cov))
                print('min eigenvalue',torch.min(truncated_cov))
                print('max eigenvalue',torch.max(truncated_cov))
                print()
                mvn_diagonal = D.Independent(D.Normal(truncated_mean, torch.sqrt(truncated_cov)), 1)
                #mvn_full = D.MultivariateNormal(truncated_mean, covariance_matrix=truncated_cov)
                self.component_distributions_firsts.append(mvn_diagonal)

            for i in range(self.num_total_components):
                # Slice means and covs to remove the last `leng_last` elements
                prec = torch.clamp(self.last_prec[i],min=-1e6,max=1e6)+(torch.eye(self.leng_last)*1e-6).to(self.last_prec[i].device)
                print('Laplace prec last',prec)
                print('median eigenvalue',torch.median(prec))
                print('min eigenvalue',torch.min(prec))
                print('max eigenvalue',torch.max(prec))
                print()                
                
                mvn_full = D.MultivariateNormal(self.means[i][-self.leng_last:], precision_matrix=self.last_prec[i])
                self.component_distributions_last.append(mvn_full)

         
         

    def negative_log_likelihood(self, point):
        """
        Compute the log likelihood of the point under the diagonal mixture model.

        Args:
            point (torch.Tensor): Point at which to evaluate the log likelihood.

        Returns:
            torch.Tensor: Log likelihood of the point under the mixture model.
        """


        # Calculate log probabilities for each component
        if self.composed==False:
            log_probs = []
            for j in range(int(self.n_clients)):
                aux=[]
                for i in range(int(self.num_client_mix)):
                    index=int((j*self.num_client_mix)+i)
                    log_prob = self.component_distributions[index].log_prob(point) + self.log_weights[index]
                    #print('log_prob',log_prob)
                    aux.append(log_prob)
                log_probs.append(torch.logsumexp(torch.stack(aux), dim=0))

        if self.composed==True:
            log_probs = []
            for j in range(int(self.n_clients)):
                aux=[]
                for i in range(int(self.num_client_mix)):
                    index=int((j*self.num_client_mix)+i)
                    log_prob = self.component_distributions_firsts[index].log_prob(point[:-self.leng_last]) + self.component_distributions_last[index].log_prob(point[-self.leng_last:])
                    #print('log_prob',log_prob)
                    aux.append(log_prob)
                log_probs.append(torch.logsumexp(torch.stack(aux), dim=0))

        # Compute logsumexp of component log probabilities
        negative_log_likelihood = - torch.sum(torch.stack(log_probs), dim=0)

        return negative_log_likelihood
    

    
    def negative_log_posterior(self, point):

        posterior_part = self.negative_log_likelihood(point)

        total= posterior_part + ( self.n_clients - 1 )*self.prior.log_prob(point)
        return total





    
    
    def optimize(self,l_r,n_steps,w_0,net, dataset_val,args):
        # Initialize the variable to optimize (parameter)

        x = torch.tensor(w_0, requires_grad=True)
        # Define the Adam optimizer
        optimizer = torch.optim.Adam([x],lr=0.001,eps=1e-06)
        #optimizer=torch.optim.SGD([x], 0.01)
        test_acc_i_max = 0
        net_glob_copy = copy.deepcopy(net)

        # Optimization loop
        for k in range(n_steps):
            # Zero the gradients
            optimizer.zero_grad()
            y =   self.negative_log_posterior(x)
            y.backward(retain_graph=True)
            # Update x using the optimizer
            optimizer.step()

            
            if(k%50==0):
                w_vec_estimate = x
                vector_to_parameters(w_vec_estimate,net_glob_copy.parameters())
                test_acc_i, test_loss_i,_ = test_img(net_glob_copy, dataset_val, args)

                if self.val_at_server==True:
                    print()
                    print('server at val avaibale')
                    if(test_acc_i > test_acc_i_max):
                        test_acc_i_max = test_acc_i
                        best_parameters = w_vec_estimate.detach().clone()
                else:
                    best_parameters = w_vec_estimate.detach().clone()

                
                print('Estimated log_posterior',y)
                print ("Val Test Acc: ", test_acc_i, " Val Test Loss: ", test_loss_i)


        return best_parameters  



'''   
def mean_UN_CORRELATED_diag(means, precisions):
    # Convert inputs to PyTorch tensors
    means = [torch.as_tensor(mu, dtype=torch.float32) for mu in means]
    #covariances = [torch.as_tensor(cov, dtype=torch.float32) for cov in covariances]
    precis = [torch.as_tensor(prec, dtype=torch.float32) for prec in precisions]

    aux=[torch.mul(precis[i], means[i]) for i in range(len(means))]
    aux=torch.cat(aux).reshape(-1,len(means[0]))
    aux=torch.sum(aux,dim=0)

    aux2=torch.cat(precis).reshape(-1,len(precis[0]))
    aux2=torch.sum(aux2,dim=0)

    mean_prod = torch.mul(1/aux2, aux)

    covariance_matrix = 1/aux2


    return mean_prod, covariance_matrix
'''
def mean_UN_CORRELATED_diag(means, precisions):

    # Convert inputs to PyTorch tensors
    means = [torch.as_tensor(mu, dtype=torch.float32) for mu in means]
    #covariances = [torch.as_tensor(cov, dtype=torch.float32) for cov in covariances]
    precis = [torch.as_tensor(prec, dtype=torch.float32) for prec in precisions]

    aux=[torch.mul(precis[i], means[i]) for i in range(len(means))]
    aux=torch.cat(aux).reshape(-1,len(means[0]))
    aux=torch.sum(aux,dim=0)

    aux2=torch.cat(precis).reshape(-1,len(precis[0]))
    aux2=torch.sum(aux2,dim=0)

    mean_prod = torch.mul(1/aux2, aux)

    covariance_matrix = 1/aux2

    scaling=log_scaling_factor(means[0],means[1], 1/precis[0],1/precis[1])



    return mean_prod, covariance_matrix,scaling

"""
def compute_product_mixtures(ms,covs,n_clients):
  m_c = [mu for mu in ms]
  cov_c = [cov for cov in covs]
  aux_m=[]
  aux_cov=[]
  for j in range(len(m_c[0])):
    for i in range(len(m_c[0])):
      for k in range(len(m_c[0])):
        m,cov=mean_UN_CORRELATED_diag([m_c[0][j],m_c[1][i],m_c[2][k]],[cov_c[0][j],cov_c[1][i],cov_c[2][k]])
        aux_m.append(m)
        aux_cov.append(cov)
  print('number mixtures !!!',len(aux_cov))
  return aux_m, aux_cov
"""

def log_scaling_factor(mean1,mean2, covariance1,covariance2):
    """
    Compute the probability density function (PDF) of a multivariate Gaussian distribution
    at a given point using PyTorch.

    Args:
        mean (torch.Tensor): Mean vector of the Gaussian distribution.
        covariance (torch.Tensor): Covariance matrix of the Gaussian distribution.
        point (torch.Tensor): Point at which to evaluate the PDF.

    Returns:
        torch.Tensor: Value of the PDF at the specified point.
    """
    # Define the multivariate normal distribution
    mvn_diagonal = D.Independent(D.Normal(mean1-mean2, torch.sqrt(covariance1+covariance2)), 1)
    #mvn = D.MultivariateNormal(mean1-mean2, covariance1+covariance2)


    # Evaluate the PDF at the given point
    log_pdf_value = mvn_diagonal.log_prob(mean1*0)  # log_prob gives the log PDF

    # Convert log PDF to actual PDF (exponentiate)


    return log_pdf_value


def compute_product_mixtures_diag(ms, covs, n_clients):
    # Validate input dimensions
    if len(ms) != n_clients or len(covs) != n_clients:
        raise ValueError("Lengths of means and covariances lists must match the number of clients")

    aux_m = []
    aux_cov = []
    aux_scaling_factors=[]

    # Generate all combinations of means and covariances using itertools.product
    combinations = itertools.product(*ms)  # Cartesian product of means
    combinations_cov = itertools.product(*covs)  # Cartesian product of covariances

    # Iterate over each combination of means and covariances
    for means_tuple, covs_tuple in zip(combinations, combinations_cov):
        # Calculate new mixture mean and covariance using mean_UN_CORRELATED_diag
        mixture_mean, mixture_covariance,scaling = mean_UN_CORRELATED_diag(means_tuple, covs_tuple)

        aux_m.append(mixture_mean)
        aux_cov.append(mixture_covariance)
        aux_scaling_factors.append(scaling)

    # Optionally print for debugging
    print('Number of mixtures:', len(aux_cov))

    return aux_m, aux_cov,aux_scaling_factors

def instantiate_model(flattened_params, model_instance):
    """
    Load flattened parameters into a provided instance of a PyTorch model.

    Parameters:
    - flattened_params: 1D tensor containing flattened parameters
    - model_instance: An instance of a PyTorch model

    Returns:
    - model_instance: The provided model instance with loaded parameters

    """

    # Counter for tracking the position in the flattened parameters tensor
    param_pos = 0
    # Iterate through the named parameters of the provided model instance
    for name, param in model_instance.named_parameters():
        # Determine the size of the parameter and reshape it
        param_size = param.numel()
        param_data = flattened_params[param_pos:param_pos + param_size].view(param.size())

        # Set the parameter values in the model instance
        param.data = param_data

        # Move the position counter
        param_pos += param_size

    return model_instance



def log_prob_multivariate_gaussian(x, mu, eigenvalues, eigenvectors):
    # x: Data point (n-dimensional tensor)
    # mu: Mean of the Gaussian (n-dimensional tensor)
    # eigenvalues: Vector of eigenvalues for each Kronecker factor (tensor)
    # eigenvectors: Matrix of eigenvectors for the Kronecker factors (tensor)
    
    n = mu.size(0)  # Dimensionality of the data point
    d = eigenvalues.size(0)  # Number of Kronecker factors
    
    # Compute inverse of the covariance matrix Sigma
    inv_lambda = 1.0 / eigenvalues
    inv_diag_lambda = torch.diag(inv_lambda)
    inv_sigma = eigenvectors @ inv_diag_lambda @ eigenvectors.t()
    
    # Compute determinant of the covariance matrix Sigma
    log_det_sigma = torch.sum(torch.log(eigenvalues))
    
    # Compute log probability
    diff = x - mu
    log_prob = -0.5 * diff @ inv_sigma @ diff - 0.5 * log_det_sigma - (n / 2) * torch.log(2 * torch.tensor(np.pi))
    
    return log_prob