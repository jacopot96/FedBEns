import copy
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector

from laplace import Laplace
from laplace.curvature import AsdlGGN,AsdlEF
import torch.distributions as D


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




class LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])

    def train_and_compute_fisher(self, net, n_c):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args['eta'], momentum = 0.9)
        step_count = 0

        for epoch in range(self.args['local_epochs']):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if(self.args['augmentation']==True):
                    images = self.transform_train(images)

                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            print ("Epoch No. ", epoch, "Loss " , sum(batch_loss)/len(batch_loss))
            
        F_kfac = FIM(model=net,
                          loader=self.ldr_train,
                          representation=PMatKFAC,
                          device='cuda',
                          n_output=n_c)
        
        F_diag = FIM(model=net,
                          loader=self.ldr_train,
                          representation=PMatDiag,
                          device='cuda',
                          n_output=n_c)

        F_diag = F_diag.get_diag()
        vec_curr = parameters_to_vector(net.parameters())      


        return vec_curr, net, F_kfac, F_diag
    



    

class LocalUpdate_ENSEMBLE_mixtures(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])


    def train_and_compute_fisher(self, ensemble,initialization, temperature,random_init=False,perturbation=False,std=0.001,composed=False,n_locals=1): #0,001
        param=[]
        fishers_diag=[]
        fishers_kfac=[]
        nets=[]
        last_precs=[]
        evidence=[]
        
        k=0
        for net in ensemble:
            initial_vector = initialization[int(k)]
            k+=1
            for i in range(n_locals):
                print('Perturbation: ',perturbation, '  STD: ',std)

            

                initialize_model_with_tensor(net, initial_vector)
                



                if(len(ensemble)>1):
                    print()
                    print ("Ensemble",k,'local',i)


                if(random_init==True):
                    net.apply(weight_reset)
            
                    print ("----Initial Vector--", parameters_to_vector(net.parameters()))

                elif(perturbation==True):
                    print ("----Before Perturbation--", parameters_to_vector(net.parameters()))
                    perturb_model_initialization(net, std=std)
                    print ("----Initial Vector--", parameters_to_vector(net.parameters()))

                else:
                    print ("----Initial Vector--", parameters_to_vector(net.parameters()))

                net.train()
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args['eta'], momentum = 0.9)
                print('lr',self.args['eta'])
                #optimizer = torch.optim.Adam(net.parameters(), lr=self.args['eta'])

                for epoch in range(self.args['local_epochs']):
                    batch_loss = []
                    for batch_idx, (images, labels) in enumerate(self.ldr_train):
                        images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                        if(self.args['augmentation']==True):
                            images = self.transform_train(images)

                    #if epoch == 0 and batch_idx == 0:
                    # Print the first image of the first batch in the first epoch
                    #    print('First Image of First Batch in Epoch 0:')
                    #    print(images[0])
                        optimizer.zero_grad()
                        log_probs = net(images)
                        loss = self.loss_func(log_probs, labels)
                        loss.backward()
                        optimizer.step()
                        batch_loss.append(loss.item())
            

                    print ("Epoch No. ", epoch, "Loss " , sum(batch_loss)/len(batch_loss))
            

                print()
                print('param',parameters_to_vector(net.parameters()))

                if n_locals==1:
                    temp=temperature
                else:
                    temp=temperature


                print('temperature',temp)

                la = Laplace(net, 'classification',subset_of_weights='all',hessian_structure='diag',prior_mean=0,prior_precision=1./0.1,temperature = temp,backend=AsdlGGN)
                la.fit(self.ldr_train)
                #la.optimize_prior_precision(method='marglik',val_loader=self.ldr_train)
                print('computing diagonal') #kron
                print(la.prior_precision)
                if composed==True:
                    temp_last=temp
                    print('Composed, with last layer full cov',temp_last)
                    #la2 = Laplace(net, 'classification',subset_of_weights = 'last_layer' ,hessian_structure='kron',prior_precision=1./0.06,temperature=temp_last,backend=AsdlGGN)
                    la2 = Laplace(net, 'classification',subset_of_weights = 'last_layer' ,hessian_structure='full',prior_precision=1./0.1,temperature = temp_last,backend=AsdlGGN)
                    la2.fit(self.ldr_train)

                    last_precs.append(la2.posterior_precision)


                print()
                print('num param',len(parameters_to_vector(net.parameters())))
                posterior_std=torch.sqrt(la.posterior_variance)
                print('Laplace std',posterior_std)
                print('median eigenvalue',torch.median(posterior_std))
                print('min eigenvalue',torch.min(posterior_std))
                print('evidence',la.log_marginal_likelihood())
                del posterior_std
                print()

                param.append(parameters_to_vector(net.parameters()))
                fishers_diag.append(la.posterior_precision)
                evidence.append(la.log_marginal_likelihood())
                #fishers_kfac.append(F_kfac)
                nets.append(net)
                if composed==True:
                    la2.optimize_prior_precision(method='marglik',val_loader=self.ldr_train)
                    print('inferred prior last precision',la2.prior_precision)
                la.optimize_prior_precision(method='marglik',val_loader=self.ldr_train)
                print('inferred prior others precision',la.prior_precision)
                print()


        log_norm = torch.logsumexp(torch.stack(evidence), dim=0)

        diversity=0
        mean_diversity=0
        if len(ensemble)>1:
            for i in range(len(param)):
                for j in range(i+1,len(param)):
                    diversity +=torch.nn.functional.cosine_similarity(param[i],param[j],dim=0).item()

            mean_diversity = diversity/(len(param)*(len(param)-1)/2)
        print()
        print('mean diversity',mean_diversity)
        print()


        weights = [jj-log_norm for jj in evidence]

        return param, nets, fishers_diag, last_precs, weights,mean_diversity
    

def psd_check(matrix):
    eigenval=torch.linalg.eigvalsh(matrix)
    return torch.all(eigenval >= 0.1)

def make_psd(matrix,epsilon=1e-6):
    matrix= (matrix+matrix.T)/2
    eigenval,eigenvect=torch.linalg.eigh(matrix)
    eigenval=torch.clamp(eigenval,min=epsilon)
    #id_matrix=torch.eye(matrix.size(0),dtype = matrix.dtype,device=matrix.device)
    #adjusted= matrix+epsilon*id_matrix
    adjusted = eigenvect@torch.diag(eigenval)@eigenvect.T
    return adjusted