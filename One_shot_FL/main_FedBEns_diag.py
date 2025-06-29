import numpy as np
import random
import copy
import argparse
import csv
import torch
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import get_dataset
from models import get_model
from train_model import LocalUpdate,LocalUpdate_ENSEMBLE_mixtures
from run_one_shot_algs import get_one_shot_model
from utils.compute_accuracy import test_img,test_img_ensemble

from algs.utils_mixtures import mean_UN_CORRELATED_diag, compute_product_mixtures_diag, instantiate_model, DiagonalMixtureModel_versionB


def str_to_bool(s):
  if s.lower() in ('True','true','1'):
    return True
  else:
    return False

parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
#parser.add_argument('--algs_to_run', nargs = '+', type=str, required=False,default='fedfisher_kfac')
parser.add_argument('--seed', type=int, required=False, default = 0)
parser.add_argument('--alpha', type = float, required = False, default = 0.05)  #mixture
parser.add_argument('--num_clients', type = int, required = False, default = 5)
parser.add_argument('--num_rounds', type = int, required = False, default = 1)
parser.add_argument('--local_epochs', type=int, required= False, default = 30)
parser.add_argument('--use_pretrained', type=bool, required = False, default = False) 
parser.add_argument('--n_mixtures', type=int, required = True, default = 5) 
parser.add_argument('--temperature', type=float, required = True, default = 0.5) 
parser.add_argument('--weights',type=str_to_bool,choices=[True,False], required = True, default = False) 
parser.add_argument('--local_ens',type=int,required = True, default = 1)
parser.add_argument('--val_at_server',type=str_to_bool,choices=[True,False],required = True, default = False)


args_parser = parser.parse_args()


seed = args_parser.seed
dataset = args_parser.dataset
model_name = args_parser.model
#algs_to_run = args_parser.algs_to_run
local_epochs = args_parser.local_epochs
use_pretrained = args_parser.use_pretrained
alpha = args_parser.alpha
num_clients = args_parser.num_clients
temperature=args_parser.temperature
num_rounds = args_parser.num_rounds
weights_mixtures = args_parser.weights
N_ensemble_local = args_parser.local_ens
val_at_server = args_parser.val_at_server


print_every_test = 1
print_every_train = 1
N_ENSEMBLE =  args_parser.n_mixtures
COMPOSED = False

print('alpha',alpha)
print(model_name)
print('val_at_server',val_at_server)

filename = str(num_clients)+"one_shot_ENS_"+str(N_ENSEMBLE)+"__"+str(seed)+"_"+"alpha_"+str(alpha)+"T"+str(temperature)+"_"+str(val_at_server)+"_"+dataset+"_"+model_name+"_"+"_"+str(local_epochs)
filename_csv = "results_DIAG/"+filename + ".csv"


if(dataset=='CIFAR100'):
  n_c = 100
elif (dataset == 'GTSRB'):
  n_c = 43
else: n_c = 10

dict_results = {}

algs_to_run='a'
for alg in algs_to_run:
  print ("Running algorithm", alg)
  print ("Using pre-trained model:", use_pretrained)

  np.random.seed(3)
  dataset_train, dataset_train_global, dataset_test_global, net_cls_counts = get_dataset(dataset, num_clients, n_c, alpha, True)
  test_loader = DataLoader(dataset_test_global, batch_size=len(dataset_test_global))

  ind = np.random.choice(len(dataset_train_global), 500)
  dataset_val = torch.utils.data.Subset(dataset_train_global, ind)

  ### Default parameters
  args={
  "bs":64,
  "local_epochs":local_epochs,
  "device":'cuda',               ###!!!! per MAC senno mettere cuda
  "rounds":num_rounds, 
  "num_clients": num_clients,
  "augmentation": False,
  "eta": 0.01,
  "dataset":dataset,
  "model":model_name,
  "use_pretrained":use_pretrained,
  "n_c":n_c
  }



  torch.manual_seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  net_glob_org = get_model(args['model'], n_c, bias = False, use_pretrained = use_pretrained).to(args['device'])

  print('model',net_glob_org)


  n = len(dataset_train)
  print ("No. of clients", n)

  ### Computing weights of the local models proportional to datasize
  p = np.zeros((n))
  for i in range(n):
    p[i] =len(dataset_train[i])#???????????????????
  p = p/np.sum(p)
  
  
  def reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()



  local_model_accs = []
  local_model_loss = []
  d = parameters_to_vector(net_glob_org.parameters()).numel()
  net_glob = copy.deepcopy(net_glob_org)
  initial_vector = parameters_to_vector(net_glob.parameters())



  for t in range(0,args['rounds']):

    if(dataset=='CIFAR10' or dataset=='CIFAR100' or dataset == 'CINIC10' or dataset == 'GTSRB'):
      args['augmentation'] = True

    if(use_pretrained == True):
        args['eta'] = 0.001     ### Smaller learning rate if using pretrained model
          
    ind = [i for i in range(n)]
    F_kfac_list = []
    F_diag_list = []
    evidence_list = []
    model_vectors = []
    models = []
    precisions = []
    last_precisions = []
    starting_points = []
    diversity=[]

    ensemble = [copy.deepcopy(net_glob.apply(reset)) for _ in range(N_ENSEMBLE)]
    starting_points = [parameters_to_vector(net.parameters()) for net in ensemble]
    for i in range(len(starting_points)):
      print(len(ensemble)) 
      print('starting_points',starting_points[i])

    del ensemble

    if N_ensemble_local>1:
      perturb=True
    else:
      perturb=False

    perturb=False
    
    for i in ind:
        print()
        print ("Training Local Model ", i)
        net_glob.train()
        ensemble = [copy.deepcopy(net_glob) for _ in range(N_ENSEMBLE)]
        local = LocalUpdate_ENSEMBLE_mixtures(args=args, dataset=dataset_train[i])
        model_vector, model, F_diag,prec_last,evidence,diver = local.train_and_compute_fisher(ensemble,starting_points, temperature,random_init = False, perturbation = perturb, std = 0.001,composed=COMPOSED,n_locals=N_ensemble_local)
        #del ensemble
        model_vectors.append(model_vector)
        models.append(model)
        precisions.append(F_diag)
        last_precisions.append(prec_last)
        evidence_list.append(evidence)
        diversity.append(diver)

        test_acc, test_loss,_= test_img(instantiate_model( model_vector[0],net_glob_org), dataset_train[0],args)
        print ("Local Model ", i, "Train Acc. ", test_acc, "Train Loss ", test_loss)
        if(len(model_vector)>1):
          test_acc, test_loss,_ = test_img(instantiate_model( model_vector[1],net_glob_org), dataset_train[0],args)
          print ("Local Model ", i, "Train Acc. ", test_acc, "Train Loss ", test_loss)


        test_acc, test_loss,_ = test_img(instantiate_model( model_vector[0],net_glob_org), dataset_test_global,args)
        print ("Local Model ", i, "Test Acc. ", test_acc, "Test Loss ", test_loss)
        if(len(model_vector)>1):
          test_acc, test_loss,_ = test_img(instantiate_model( model_vector[1],net_glob_org), dataset_test_global,args)
          print ("Local Model ", i, "Test Acc. ", test_acc, "Test Loss ", test_loss)
        if(len(model_vector)>2):
          test_acc, test_loss,_ = test_img(instantiate_model( model_vector[2],net_glob_org), dataset_test_global,args)
          print ("Local Model ", i, "Test Acc. ", test_acc, "Test Loss ", test_loss)
        if(len(model_vector)>3):
          test_acc, test_loss,_ = test_img(instantiate_model( model_vector[3],net_glob_org), dataset_test_global,args)
          print ("Local Model ", i, "Test Acc. ", test_acc, "Test Loss ", test_loss)
        if(len(model_vector)>4):
          test_acc, test_loss,_ = test_img(instantiate_model( model_vector[4],net_glob_org), dataset_test_global,args)
          print ("Local Model ", i, "Test Acc. ", test_acc, "Test Loss ", test_loss)
        

  print(len(model_vectors))
  print(len(model_vectors[0]))
  #modes, covs,weights = compute_product_mixtures_diag(model_vectors, precisions, num_clients)


  #prior_mean = torch.zeros(d).to(args['device'])
  prior_mean = (torch.zeros(d)).to(args['device'])
  prior_cov = (torch.ones(d)*1.).to(args['device'])
  #global_posterior = DiagonalMixtureModel(modes, covs,weights,prior_mean, prior_cov, num_clients)
  model_map=[item for sublist in model_vectors for item in sublist]
  precisions=[item for sublist in precisions for item in sublist]
  last_precisions=[item for sublist in last_precisions for item in sublist]
  
  if COMPOSED==False:
    last_precisions=None


  if(weights_mixtures == False): #assumption that each client has the same number of mixtures
    l=int(N_ENSEMBLE*N_ensemble_local)
    evidence_list = torch.log(torch.ones(len(model_map))/l).to(args['device'])

    print('evidence1------',evidence_list)



  else:
    print('mmmmmmmmm')
    evidence_list = [item for sublist in evidence_list for item in sublist]
    print('evidence1------',evidence_list)


  
  #means, prec, mean_prior, cov_prior,weights,n_clients,n_client_mixtures,composed=False,prec_last=None,val_at_server=False
  global_posterior = DiagonalMixtureModel_versionB(model_map, precisions, prior_mean, prior_cov,evidence_list, num_clients,N_ENSEMBLE*N_ensemble_local,COMPOSED,last_precisions,val_at_server)
  w_0_tent = torch.mean(torch.stack(model_map), dim=0)
  #w_0_tent = torch.median(torch.stack(model_map), dim=0).values
  #w_0 = torch.median(torch.stack([t[0] for t in model_vectors]), dim=0).values
  #w_1 = torch.median(torch.stack([t[1] for t in model_vectors]), dim=0).values  
  #w_2 = torch.median(torch.stack([t[2] for t in model_vectors]), dim=0).values  

  start_median=True
  if start_median==True:
     print('start point MEDIAN')
     for i in range (len(starting_points)):
        starting_points[i] = torch.median(torch.stack([t[i] for t in model_vectors]), dim=0).values 



  W_hats=[]
  for i in range(N_ENSEMBLE):
    print()
    print('-----------',i,'------------')
    print()
    W_hats.append(global_posterior.optimize(0.01,1000,starting_points[i],net_glob_org,dataset_val,args))

  del starting_points


  print()


  print('aaaaaaaaa',W_hats)

  models=[]
  for i in range(N_ENSEMBLE):
    models.append(instantiate_model(W_hats[i],ensemble[i]))
    #models.append(vector_to_parameters(W_hats[i],ensemble[i].parameters()))

  del ensemble

  for i in range(N_ENSEMBLE):
    test_acc, test_loss,_ = test_img(models[i], dataset_test_global,args)
    print ("Test Acc",i,':', test_acc, "Test Loss", test_loss)
    print()
  
  test_acc, test_loss,_,dis= test_img_ensemble(dataset_test_global,args,models)
  print('ENSEMBLE',test_acc,'loss',test_loss)

  print('seed',seed)
  print('dataset',dataset)
  print('alpha',alpha)
  print('n_mixture',N_ENSEMBLE)
  print('n_mixt_LOCAL',N_ensemble_local)
  print('weights',weights_mixtures)
  print('val_at_server',val_at_server)
  print('mean diversity',np.mean(diversity))
  print('disagreement',dis)







  #w_hat_global2 = global_posterior.optimize(0.01,1200,initial_vector,net_glob,dataset_test_global,args)
  #print('w_hat_global',w_hat_global2)

  #net_glob2 = instantiate_model( w_hat_global2,net_glob_org)


  ### Creating one-shot model depending on the algorithm
  #net_glob = get_one_shot_model(alg, d,n,p,args,net_glob, models, model_vectors, \
  #F_kfac_list, F_diag_list, dataset_val, dataset_train, dataset_train_global, \
  #dataset_test_global, filename, net_cls_counts)

  #  net_glob_ENSEMBLE = get_one_shot_ENSEMBLE()





  #test_acc2, test_loss2 = test_img(net_glob2, dataset_test_global,args)
  #print ("Test Acc2. ", test_acc2, "Test Loss", test_loss2)
  print()
  #test_acc_ens, test_loss_ens = test_img_ensemble(net_glob,net_glob2, dataset_test_global,args)
  #print('ENSEMBLE',test_acc_ens,'loss',test_loss_ens)



dict_results[alg + '_test_loss_'+str(seed)+"_"+str(t)] = test_loss
dict_results[alg + '_test_acc_' +str(seed)+"_"+str(t)] = test_acc
  
with open(filename_csv, 'w') as csv_file:    
  writer = csv.writer(csv_file)
  for i in dict_results.keys():
      writer.writerow([i, dict_results[i]])




with open(filename_csv, 'w') as csv_file:    
  writer = csv.writer(csv_file)
  for i in dict_results.keys():
      writer.writerow([i, dict_results[i]])
