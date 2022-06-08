import argparse
import torch
torch.cuda.empty_cache()
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
from scipy.special import softmax
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.manifold import TSNE

sys.path.insert(1, '/lcrc/project/FastBayes/sanket_bnn/SeqEns_expts/')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from Gauss_resnet_models_new import SSGauss_Node_ResNet_Cifar
from data_utils import get_cifar10_dataloaders, get_cifar100_dataloaders

class Cifar_C_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ims, labels, transform):
        'Initialization'
        self.ims = ims
        self.labels = torch.Tensor(labels)
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ims)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.ims[index]
        label = self.labels[index]
        X = self.transform(image)
        return (X, label)


def milestone_calculation(args):
    individual_epoch = (args.total_epochs - args.epochs_explo) / args.model_num
    args.individual_epoch = individual_epoch
    reset_lr_epochs1 = []
    epoch_ = args.epochs_explo
    for _ in range(args.model_num):
        reset_lr_epochs1.append(epoch_)
        epoch_ = epoch_ + individual_epoch
    reset_lr_epochs2 = np.array(reset_lr_epochs1) + individual_epoch / 2
    reset_lr_epochs1.pop(0)
    jump = individual_epoch / 2
    return np.ceil(reset_lr_epochs1), np.ceil(reset_lr_epochs2), np.ceil(jump)

def extract_prediction(test_loader, model):
    model.eval()

    y_output = []
    y_true = []

    for _, (input, target) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()
        # compute output
        with torch.no_grad():
            output = model(input)

            y_true.append(target.cpu().numpy())
            y_output.append(output.cpu().numpy())

    y_output = np.concatenate(y_output, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    return y_output, y_true

def expected_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print('Individual Model: NLL loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n), test_loss

def evaluate_ensemble(model, device, test_loader):
    model.eval()
    current_fold_preds = []
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            current_fold_preds.append(output)
            labels.append(target)
    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    labels = torch.cat(labels, dim=0)

    return current_fold_preds, labels

def evaluate_ensemble_KD(model, device, test_loader):
    model.eval()
    current_fold_preds = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            logits = F.log_softmax(output, dim=1)
            current_fold_preds.append(logits)
    current_fold_preds = torch.cat(current_fold_preds, dim=0)

    return current_fold_preds

def loss_fn_kd(scores, target_scores):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""
    # if len(scores.size()) != 2:
    #     scores = scores.reshape(scores.size(0), -1)
    #     target_scores = scores.reshape(target_scores.size(0), -1)

    device = scores.device

    log_scores_norm = F.log_softmax(scores)
    targets_norm = F.softmax(target_scores)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    if not scores.size(1) == target_scores.size(1):
        print('size does not match')

    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)
    KD_loss_unnorm = F.kl_div(log_scores_norm, targets_norm, reduction="batchmean")

    return KD_loss_unnorm

def test_calibration(test_loader, model):

    y_pred, y_true = extract_prediction(test_loader, model)
    ece = expected_calibration_error(y_true, softmax(y_pred,axis=1))
    nll = F.cross_entropy(torch.from_numpy(y_pred), torch.from_numpy(y_true), reduction="mean")
    print('Individual Model ECE = {}'.format(ece))
    print('Individual Model NLL = {}'.format(nll))

    return y_pred, y_true

# Given model and test data, return true_labels and predictions.
def evaluate_tsne(model, device, test_loader):
    model.eval()
    n = 0
    pred_labels = []
    with torch.no_grad():
        for data, _ in test_loader:
            n += 1
            if n > 1: break
            data = data.to(device)
            
            output = model(data)
            output = F.softmax(output, dim=1)

            pred_labels.append(output)

    pred_labels = torch.cat(pred_labels, dim=0)
    return pred_labels

def main():
    parser = argparse.ArgumentParser(description='ResNet-32 Cifar-10 Evaluation')

    # Basic Setting
    parser.add_argument('--seed', default=1, type = int, help = 'set seed')
    parser.add_argument('--results_path', default='./results_resnet/cifar10/SS_Gauss_Node_SeqEns_Modified/', 
                            type = str, help = 'base path for saving result')
    parser.add_argument('--mode', type=str, help='predict/calibration/disagreement/KD/tsne')

    # Resnet Architecture
    parser.add_argument('--depth', default=32, type=int, help='depth of the resnet')
    parser.add_argument('--model-num', type=int, default=3, help='number of models in ensemble')
    parser.add_argument('--div-model-num', type=int, default=3, help='number of models in ensemble for diversity analysis')

    # Data setting
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--corrupt', action='store_true', default=False, help='Use corrupted data. Default: True.')  

    # Training Setting
    parser.add_argument('--total-epochs', type=int, default=450, help='total number of epochs to train ensemble')
    parser.add_argument('--epochs-explo', type=int, default=150, help='training epochs of exploration phase')
    parser.add_argument('--init_lr', default = 0.1, type = float, help = 'initial learning rate')
    parser.add_argument('--perturb_factor', default = 3, type = int, help = 'perturbation factor to multiply stad dev with')
    parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--batch_size', default = 128, type = int, help = 'batch size for training')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--test_batch_size', default = 10000, type = int, help = 'batch size for training')
    parser.add_argument('--freeze', action='store_true', default=False, help='Freeze sparsity after exploration. Default: True.')  

    # Optimizer Setting
    parser.add_argument('--clip', default = 1.0, type = float, help = 'Gradient clipping value')

    # Prior Setting
    parser.add_argument('--sigma_0', default = 0.2, type = float, help = 'sigma_0^2 in prior')
    parser.add_argument('--temp', default = 0.5, type = float, help = 'temperature')
    parser.add_argument('--gamma_prior', default = 0.0001, type = float, help = 'prior inclusion probaility for filters/nodes')

    args = parser.parse_args()

    print('\n\n\n###############################################################################\n\n\n')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.data == 'cifar10':
        _ , _ , test_loader, _ = get_cifar10_dataloaders(args, args.valid_split,
                                                                            max_threads=args.max_threads)
        target_dim = 10
    elif args.data == 'cifar100':
        _ , _ , test_loader, _ = get_cifar100_dataloaders(args, args.valid_split,
                                                                            max_threads=args.max_threads)
        target_dim = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    sigma_0 = torch.as_tensor(args.sigma_0).to(device)
    temp = torch.as_tensor(args.temp).to(device)
    gamma_prior = torch.tensor(args.gamma_prior).to(device)

    model = SSGauss_Node_ResNet_Cifar(depth=args.depth, num_classes=target_dim, temp = temp, 
                            gamma_prior = gamma_prior, sigma_0 = sigma_0, testing = 1).to(device)

    milestone1, milestone2, jump = milestone_calculation(args)
    print(f'learning rate recycle minestone1 is {milestone1} minestone2 is {milestone2} jump is {jump}')

    if args.freeze: 
        RESULTS_PATH = args.results_path + 'Freeze/Perturb=%d/' % (args.perturb_factor) + 'Seed=%d/' % (args.seed)
    else:
        RESULTS_PATH = args.results_path + 'No_Freeze/Perturb=%d/' % (args.perturb_factor) + 'Seed=%d/' % (args.seed)

    MODELS_PATH = RESULTS_PATH + 'Models/'

    df = pd.read_csv(RESULTS_PATH+'results_SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                            '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                            '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                            '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                            '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                            '_ens_models_'+ str(args.model_num)+'_Step_'+str(args.model_num)+'.csv', 
                            usecols= ['Test Accuracy'])

    test_accuracy_col= df['Test Accuracy']

    ###############################################################################
    #                          ensemble prediction                              #
    ##############################################################################
    if 'predict' in args.mode:
        all_folds_preds = []
        val_acc = []
        nll_loss = []
        for m in range(args.model_num):
            print('#####################################################################################')
            
            epoch = test_accuracy_col.iloc[int(milestone2[m]):int(milestone2[m]+jump)].idxmax()+1
            print('Sequential Ensembling Step {}, best epoch {}'.format(m+1, epoch))        

            model.load_state_dict(torch.load(MODELS_PATH + 'SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                                '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                '_ens_models_'+ str(args.model_num)+'_model_'+ str(epoch)+'.pt'))
            
            indi_acc, indi_loss = evaluate(model, device, test_loader)
            val_acc.append(indi_acc)
            nll_loss.append(indi_loss)

            current_fold_preds, target = evaluate_ensemble(model, device, test_loader)
            all_folds_preds.append(current_fold_preds)

            individual_acc_mean = np.array(val_acc).mean(axis=0)
            individual_acc_std = np.array(val_acc).std(axis=0)
            individual_nll_mean = np.array(nll_loss).mean(axis=0)
            individual_nll_std = np.array(nll_loss).std(axis=0)
            print(f"Averaged individual model: acc is {individual_acc_mean} and std is {individual_acc_std}")
            print(f"Averaged individual model: NLL is {individual_nll_mean} and std is {individual_nll_std}")

            output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
            test_loss = F.cross_entropy(output_mean, target, reduction='mean').item()
            pred = output_mean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            n = target.shape[0]
            print('Sequential Ensemble Model Results: NLL loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                    test_loss, correct, n, 100. * correct / float(n)))

    ###############################################################################
    #                          ensemble prediction                              #
    ##############################################################################
    if 'calibration' in args.mode:
        if args.corrupt:
            all_folds_preds_c = []
            if args.data == 'cifar10':
                data_path = './data/CIFAR-10-C/'
                transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    (0.2470, 0.2435, 0.2616))
                            ])
            elif args.data == 'cifar100':
                data_path = './data/CIFAR-100-C/'
                transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                            ])
            for m in range(args.div_model_num):
                print('#####################################################################################')
                
                epoch = test_accuracy_col.iloc[int(milestone2[m]):int(milestone2[m]+jump)].idxmax()+1
                print('Sequential Ensembling Step {}, best epoch {}'.format(m+1, epoch))        

                model.load_state_dict(torch.load(MODELS_PATH + 'SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                                '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                '_ens_models_'+ str(args.model_num)+'_model_'+ str(epoch)+'.pt'))

                file_list = os.listdir(data_path)                
                file_list.sort()

                all_outputs = []
                all_targets = []

                for file_name in file_list:
                    if not file_name == 'labels.npy':
                        attack_type = file_name[:-len('.npy')]
                        print('Attack: {}'.format(attack_type))
                        distorted_images = np.load(data_path + attack_type + '.npy')
                        labels = np.load(data_path + 'labels.npy')
                        distorted_dataset = Cifar_C_Dataset(distorted_images,labels,transform)

                        corrupt_test_loader = torch.utils.data.DataLoader(
                                                            distorted_dataset,
                                                            args.test_batch_size,
                                                            shuffle=False,
                                                            num_workers=1)
                        y_output, y_true = extract_prediction(corrupt_test_loader, model)

                        all_outputs.append(y_output)
                        all_targets.append(y_true)

                all_outputs = np.concatenate(all_outputs, axis=0)
                all_targets = np.concatenate(all_targets, axis=0)
                print('Individual Model c-Acc = {}'.format(np.mean(np.argmax(all_outputs, 1)==all_targets)))
                all_folds_preds_c.append(all_outputs)

                ece = expected_calibration_error(all_targets, softmax(all_outputs,axis=1))
                nll = F.cross_entropy(torch.from_numpy(all_outputs), torch.from_numpy(all_targets).long(), reduction="mean")
                print('Individual Model c-ECE = {}'.format(ece))
                print('Individual Model c-NLL = {}'.format(nll))

                output_mean = np.mean(np.stack(all_folds_preds_c, 0), 0)

                ece = expected_calibration_error(all_targets, softmax(output_mean,axis=1))
                nll = F.cross_entropy(torch.from_numpy(output_mean), torch.from_numpy(all_targets).long(), reduction="mean")
                print('Sequential Ensemble Model Results: c-Acc = {}, c-NLL = {}, c-ECE = {}'.format( 
                            np.mean(np.argmax(output_mean, 1)==all_targets),nll,ece))

        else:
            all_folds_preds = []
            for m in range(args.model_num):
                print('#####################################################################################')
                
                epoch = test_accuracy_col.iloc[int(milestone2[m]):int(milestone2[m]+jump)].idxmax()+1
                print('Sequential Ensembling Step {}, best epoch {}'.format(m+1, epoch))        

                model.load_state_dict(torch.load(MODELS_PATH + 'SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                                '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                '_ens_models_'+ str(args.model_num)+'_model_'+ str(epoch)+'.pt'))

                y_pred, y_true = test_calibration(test_loader, model)
                all_folds_preds.append(y_pred)

                output_mean = np.mean(np.stack(all_folds_preds, 0), 0)
                ece = expected_calibration_error(y_true, softmax(output_mean,axis=1))
                nll = F.cross_entropy(torch.from_numpy(output_mean), torch.from_numpy(y_true), reduction="mean")

                print('Sequential Ensemble Model Results: NLL loss: {:.4f}, ECE: {:.4f}\n'.format(nll, ece))

    ###############################################################################
    #                          ensemble KL                                       #
    ###############################################################################
    if 'KD' in args.mode:
        all_folds_preds = []
        for m in range(args.div_model_num):
            print('#####################################################################################')
            
            epoch = test_accuracy_col.iloc[int(milestone2[m]):int(milestone2[m]+jump)].idxmax()+1
            print('Sequential Ensembling Step {}, best epoch {}'.format(m+1, epoch))        

            model.load_state_dict(torch.load(MODELS_PATH + 'SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                                '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                '_ens_models_'+ str(args.model_num)+'_model_'+ str(epoch)+'.pt'))


            current_fold_preds = evaluate_ensemble_KD(model, device, test_loader)
            all_folds_preds.append(current_fold_preds)
        KL_SCORE = []
        KL_SCORE.append(loss_fn_kd(all_folds_preds[0], all_folds_preds[1]).cpu().data.numpy())
        KL_SCORE.append(loss_fn_kd(all_folds_preds[1], all_folds_preds[0]).cpu().data.numpy())
        KL_SCORE.append(loss_fn_kd(all_folds_preds[0], all_folds_preds[2]).cpu().data.numpy())
        KL_SCORE.append(loss_fn_kd(all_folds_preds[2], all_folds_preds[0]).cpu().data.numpy())
        KL_SCORE.append(loss_fn_kd(all_folds_preds[1], all_folds_preds[2]).cpu().data.numpy())
        KL_SCORE.append(loss_fn_kd(all_folds_preds[2], all_folds_preds[1]).cpu().data.numpy())
        mean_KD = np.array(KL_SCORE).mean()
        print('KL is:', mean_KD)

    ###############################################################################
    #                          disagreement                                      #
    ##############################################################################
    if 'disagreement' in args.mode:

        labels = []
        val_acc = []
        nll_loss = []
        all_folds_preds = []

        for m in range(args.div_model_num):
            print('#####################################################################################')
            
            epoch = test_accuracy_col.iloc[int(milestone2[m]):int(milestone2[m]+jump)].idxmax()+1
            print('Sequential Ensembling Step {}, best epoch {}'.format(m+1, epoch))        

            model.load_state_dict(torch.load(MODELS_PATH + 'SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                                '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                '_ens_models_'+ str(args.model_num)+'_model_'+ str(epoch)+'.pt'))

            current_fold_preds, target = evaluate_ensemble(model, device, test_loader)
            all_folds_preds.append(current_fold_preds)
            labels.append(target)

        print(torch.equal(labels[0], labels[1]))

        predictions = []
        num_models = len(all_folds_preds)
        dis_matric = np.zeros(shape=(num_models,num_models))

        for i in range(num_models):
            pred_labels = np.argmax(all_folds_preds[i].cpu().numpy(), axis=1)
            predictions.append(pred_labels)

        for i in range(num_models):
            preds1 = predictions[i]
            for j in range(i, num_models):
                preds2 = predictions[j]
                # compute dissimilarity
                dissimilarity_score = 1 - np.sum(np.equal(preds1, preds2)) / (preds1.shape[0])
                dis_matric[i][j] = dissimilarity_score
                if i is not j:
                    dis_matric[j][i] = dissimilarity_score

        dissimilarity_coeff = dis_matric[::-1]
        dissimilarity_coeff = dissimilarity_coeff + 0.001 * (dissimilarity_coeff!=0)
        plt.figure(figsize=(9, 8))
        ss = np.arange(num_models)[::-1]
        ax = sns.heatmap(dissimilarity_coeff, cmap='RdBu_r', vmin=0, vmax=0.035)
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=22)
        plt.xticks(list(np.arange(num_models) + 0.5), list(np.arange(num_models) + 1), fontsize=22)
        plt.yticks(list(np.arange(num_models) + 0.5), list(np.arange(num_models)[::-1] + 1), fontsize=22)
        print(np.sum(dissimilarity_coeff)/np.sum(dissimilarity_coeff!=0))
        # plt.savefig("./plots/" + "%s_M=%d_prediction_disagreement_%s.pdf" % (method, num_models, args.dataset))
        plt.savefig("./plots/" + "D_Gaussian_RN32_CF10.pdf")

    ###############################################################################
    #                          ensemble tsne                                     #
    ###############################################################################
    if 'tsne' == args.mode:
        predictions_for_tsne = []
        for epoch in range(1,int(milestone1[2]+1),2):
            if epoch < 150: continue
            print('Epoch: {}'.format(epoch))

            model.load_state_dict(torch.load(MODELS_PATH + 'SS_Gaussian_Node_RN_'+str(args.depth)+'_seed_'+str(args.seed)+
                                '_sig0_'+str(args.sigma_0)+'_perturb_'+str(args.perturb_factor)+
                                '_clip_'+str(args.clip)+'_val_'+str(args.valid_split)+
                                '_batch_'+ str(args.batch_size)+'_explo_epoch_'+str(args.epochs_explo)+
                                '_lr_'+str(args.init_lr)+'_total_epochs_'+str(args.total_epochs)+
                                '_ens_models_'+ str(args.model_num)+'_model_'+ str(epoch)+'.pt'))
            preds =  evaluate_tsne(model, device, test_loader)
            predictions_for_tsne.append(preds)

        predictions_for_tsne = torch.stack(predictions_for_tsne, dim=0)
        print(predictions_for_tsne.size())

        predictions_for_tsne = predictions_for_tsne.view(-1, predictions_for_tsne.size()[-2]*predictions_for_tsne.size()[-1])
        # [60, 10000]
        print(predictions_for_tsne.size())
        # torch.save(predictions_for_tsne, args.resume + "/tsne_pred.pt")
        # predictions_for_tsne = predictions_for_tsne.numpy()
        NUM_TRAJECTORIES =3
        fontsize = 20
        tsne = TSNE(n_components=2)
        # compute tsne
        trajectory_embed = []
        prediction_embed = tsne.fit_transform(predictions_for_tsne.cpu().numpy())
        # trajectory_embed = prediction_embed.view(NUM_TRAJECTORIES, -1, 2)
        trajectory_embed = np.reshape(prediction_embed,(NUM_TRAJECTORIES, -1, 2))

        plt.figure(constrained_layout=True, figsize=(6, 6))
        Model = ['Learner-1', 'Learner-2', 'Learner-3']
        colors_list = ['r', 'b', 'g']
        # labels_list = ['traj_{}'.format(i) for i in range(NUM_TRAJECTORIES)]
        for i in range(NUM_TRAJECTORIES):
            plt.plot(trajectory_embed[i, :, 0], trajectory_embed[i, :, 1], color=colors_list[i], alpha=0.8,
                        linestyle="", marker="o",  label=Model[i])
            plt.plot(trajectory_embed[i, :, 0], trajectory_embed[i, :, 1], color=colors_list[i], alpha=0.3,
                        linestyle="-", marker="")
            plt.plot(trajectory_embed[i, 0, 0], trajectory_embed[i, 0, 1], color=colors_list[i], alpha=1.0,
                        linestyle="", marker="*", markersize=10)

        plt.xlabel('Dimension 1', fontsize=fontsize)
        plt.ylabel('Dimension 2', fontsize=fontsize)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        plt.legend(fontsize=15)
        if args.freeze: 
            plt.savefig('SS_Gaussian_Node_Freeze_RN_'+str(args.depth)+'_seed_'+str(args.seed)+'_perturb_'+str(args.perturb_factor)+'.png')
            plt.savefig('SS_Gaussian_Node_Freeze_RN_'+str(args.depth)+'_seed_'+str(args.seed)+'_perturb_'+str(args.perturb_factor)+'.pdf')
        else:
            plt.savefig('SS_Gaussian_Node_No_Freeze_RN_'+str(args.depth)+'_seed_'+str(args.seed)+'_perturb_'+str(args.perturb_factor)+'.png')
            plt.savefig('SS_Gaussian_Node_No_Freeze_RN_'+str(args.depth)+'_seed_'+str(args.seed)+'_perturb_'+str(args.perturb_factor)+'.pdf')
        # reshape
        # trajectory_embed = prediction_embed.reshape([NUM_TRAJECTORIES, -1, 2])
        # print('[INFO] Shape of reshaped tensor: ', trajectory_embed.shape)

if __name__ == '__main__':
    main()