import torch
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score,classification_report



def test_img(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    predss = []
    groud_truth = []
    
    data_loader = DataLoader(datatest, batch_size=args['bs'])
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args['device']), target.to(args['device'])
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction = 'sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        predss.append(y_pred)
        groud_truth.append(target)

        
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    target= torch.cat(groud_truth)
    y_pred= torch.cat(predss)
    f1_macro =f1_score(target.data.view_as(y_pred).cpu(),y_pred.cpu(),average='macro')

    return accuracy.numpy(), test_loss,f1_macro

def test_img_ensemble(datatest, args, net_list):
    # Set all models to evaluation mode
    for net in net_list:
        net.eval()

    device = args['device']
    batch_size = args['bs']

    
    data_loader = DataLoader(datatest, batch_size=batch_size)
    # Compute and print the number of samples per class
    class_count = {}
    for _, target in data_loader:
        for t in target:
            t = t.item()
            if t in class_count:
                class_count[t] += 1
            else:
                class_count[t] = 1

    # Deduce the number of classes
    test_loss = 0
    correct = 0
    disagreement = 0
    predss=[]
    predicted_class = []
    groud_truth = []
    num_classes = len(class_count)
    print('samples per class', class_count)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            ensemble_probs = torch.zeros((data.size(0), num_classes)).to(device)

            model_preds = []
            ens_pred = []

            # Get predictions from each model and average probabilities
            for net in net_list:
                log_probs = net(data)
                probs = F.softmax(log_probs, dim=1)
                ensemble_probs += probs

                _,pred_class = probs.max(1)
                model_preds.append(pred_class)

            ensemble_probs /= len(net_list)  # Average probabilities across all models

            # Calculate the cross-entropy loss
            test_loss += F.cross_entropy(torch.log(ensemble_probs), target, reduction='sum').item()

            # Calculate the number of correct predictions
            _, y_pred = ensemble_probs.max(1)
            correct += y_pred.eq(target).sum().item()

            predss.append(y_pred)
            groud_truth.append(target)

            # Disagreement
            model_preds = torch.stack(model_preds,dim=1)
            disagreement += torch.any(model_preds != model_preds[:,0].unsqueeze(1),dim=1).sum().item()

    # Calculate average test loss and accuracy
    test_loss /= len(datatest)
    accuracy = 100.0 * correct / len(datatest)
    disagreement_ratio= 100.*disagreement/len(datatest)
    target= torch.cat(groud_truth)
    y_pred= torch.cat(predss)
    f1_macro =f1_score(target.data.view_as(y_pred).cpu(),y_pred.cpu(),average='macro')
    print(classification_report(target.data.view_as(y_pred).cpu(),y_pred.cpu()))


    return accuracy, test_loss,f1_macro,disagreement_ratio