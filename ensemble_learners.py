from __future__ import print_function

import collections
import os
import time
import argparse
import logging
import hashlib
import copy
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.sto_resnet import StoResNet18
from models.sto_wide_resnet import Sto_ResNet, StoWideResNet
from data.data_utils import get_cifar10_dataloaders, get_cifar100_dataloaders
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import seaborn as sns
from OoD_expts_pytorch import SVHN_M, CIFAR10_M, CIFAR100_M, make_ood_dataset, eval_metrics
sns.set()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
# models['MLPCIFAR10'] = (MLP_CIFAR10,[])
# models['lenet5'] = (LeNet_5_Caffe,[])
# models['lenet300-100'] = (LeNet_300_100,[])
# models['alexnet-s'] = (AlexNet, ['s', 10])
# models['alexnet-b'] = (AlexNet, ['b', 10])
# models['vgg-c'] = (VGG16, ['C', 10])
# models['vgg-d'] = (VGG16, ['D', 10])
# models['vgg-like'] = (VGG16, ['like', 10])
models['rn-18'] = (StoResNet18, [10])
# models['resnet-20'] = (Sto_ResNet, [20, 10])
models['resnet-32'] = (Sto_ResNet, [32, 10])
models['resnet-56'] = (Sto_ResNet, [56, 10])
models['wrn-28-2'] = (StoWideResNet, [28, 2, 10])
models['wrn-28-10'] = (StoWideResNet, [28, 10, 10])
models['wrn-22-8'] = (StoWideResNet, [22, 8, 10])
models['wrn-16-8'] = (StoWideResNet, [16, 8, 10])
models['wrn-16-10'] = (StoWideResNet, [16, 10, 10])

import re


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

def milestone_calculation(args):
    individual_epoch = (args.epochs - args.epochs_explo) / args.model_num
    args.individual_epoch = individual_epoch
    reset_lr_epochs1 = []
    epoch_ = args.epochs_explo
    for _ in range(args.model_num):
        reset_lr_epochs1.append(epoch_)
        epoch_ = epoch_ + individual_epoch
    reset_lr_epochs2 = np.array(reset_lr_epochs1) + individual_epoch / 2
    reset_lr_epochs1.pop(0)
    return np.ceil(reset_lr_epochs1), np.ceil(reset_lr_epochs2)

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

def evaluate_ood(args,model,device,test_loader):
    ood_labels_G = []
    ood_scores_G = []
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)

    for _, (Data_InD, Data_OoD) in enumerate(test_loader):

        # In distribution
        images = Data_InD[0].to(device)
        # labels = Data_InD[1][0]
        ood_labels = 1-Data_InD[1][1].to(device)
        output = model(images)
        probs = torch.exp(output)
        max_probs, _ = torch.max(probs, dim = 1)
        ood_scores = 1 - max_probs
        # ood_scores = 1 - torch.max(probs, dim=-1, keepdim=False, out=None) #tf.reduce_max(probs, axis=-1)

        ood_labels_G.extend(ood_labels.detach().cpu().numpy())
        ood_scores_G.extend(ood_scores.detach().cpu().numpy())

        # Out of distribution

        images = Data_OoD[0].to(device)
        # labels = Data_OoD[1][0]
        ood_labels = 1-Data_OoD[1][1].to(device)
        output = model(images)
        probs = torch.exp(output)
        max_probs, _ = torch.max(probs, dim = 1)
        ood_scores = 1 - max_probs
        # ood_scores = 1 - torch.max(probs, dim=-1, keepdim=False, out=None) #tf.reduce_max(probs, axis=-1)

        ood_labels_G.extend(ood_labels.detach().cpu().numpy())
        ood_scores_G.extend(ood_scores.detach().cpu().numpy())

    return ood_labels_G, ood_scores_G

# # FGSM attack
# def fgsm_attack(image, epsilon, data_grad, clip_min, clip_max):
#     # Collect the element-wise sign of the data gradient
#     sign_data_grad = data_grad.sign()
#     # Create the perturbed image by adjusting each pixel of the input image
#     perturbed_image = image + epsilon*sign_data_grad
#     # Adding clipping to maintain [0,1] range
#     perturbed_image = torch.clamp(perturbed_image, clip_min, clip_max)
#     # Return the perturbed image
#     return perturbed_image

# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081], device='cuda'):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

# Function to generate adversarial examples using FGSM
def fgsm_attack(model, data, target, mean, std, epsilon, device):
    data, target = data.to(device), target.to(device)
    data.requires_grad = True
    output = model(data)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    data_denorm = denorm(data, mean, std, device=device)
    perturbed_data = data_denorm + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

def generate_adversarial_examples(args, model, device, test_loader, mean, std, epsilon):
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)

    model_adversarial_examples = []
    adversarial_targets = []

    for data, target in test_loader:
        perturbed_data = fgsm_attack(model, data, target, mean, std, epsilon, device)
        perturbed_data_normalized = transforms.Normalize(mean, std)(perturbed_data)
        model_adversarial_examples.append(perturbed_data_normalized.cpu())
        adversarial_targets.append(target.cpu())

    model_adversarial_examples = torch.cat(model_adversarial_examples)
    adversarial_targets = torch.cat(adversarial_targets)
    adversarial_dataset = TensorDataset(model_adversarial_examples, adversarial_targets)
    adversarial_loader = DataLoader(adversarial_dataset, batch_size=args.test_batch_size, shuffle=False)   

    return adversarial_loader

# Evaluate the ensemble on original and adversarial examples
def evaluate_adv_ensemble(args, model, device, adv_loader):
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)

    correct_adv = 0
    n = 0
    current_fold_adv_preds = []
    test_data = []
    with torch.no_grad():
        for data, target in adv_loader:
            data, target = data.to(device), target.to(device)
            adv_output = model(data)
            softmax_adv_preds = torch.nn.Softmax(dim=1)(input=adv_output)
            current_fold_adv_preds.append(softmax_adv_preds)
            test_data.append(target)
            adv_pred = adv_output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct_adv += adv_pred.eq(target.view_as(adv_pred)).sum().item()
            n += target.shape[0]
    current_fold_adv_preds = torch.cat(current_fold_adv_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)
    
    print(f'Individual Model Accuracy on adversarial examples: {100 * correct_adv / float(n)}%')

    return current_fold_adv_preds, test_data

# def adv_test(args,model, device, test_loader, clip_min, clip_max, epsilon):
#     model.eval()
#     if args.use_bnn:
#         model.set_test_mean(False)
#     # Accuracy counter
#     original_correct = 0.0
#     adversarial_correct = 0.0
#     test_loss = 0.0
#     test_loss_adv = 0.0
#     run_start_time = time.time()
#     # Loop over all examples in test set
#     for data, target in test_loader:

#         # Send the data and label to the device
#         data, target = data.to(device), target.to(device)

#         # Set requires_grad attribute of tensor. Important for Attack
#         data.requires_grad = True

#         # Forward pass the data through the model
#         output = model(data)
#         test_loss += F.nll_loss(output, target).item()
#         pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
#         original_correct += pred.eq(target.view_as(pred)).sum().item()

#         # Calculate the loss
#         loss = F.nll_loss(output, target)

#         # Zero all existing gradients
#         model.zero_grad()

#         # Calculate gradients of model in backward pass
#         loss.backward()

#         # Collect datagrad
#         data_grad = data.grad.data

#         # Call FGSM Attack
#         perturbed_data = fgsm_attack(data, epsilon, data_grad, clip_min, clip_max)

#         # Re-classify the perturbed image
#         output = model(perturbed_data)
#         test_loss_adv += F.nll_loss(output, target).item()
#         pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
#         adversarial_correct += pred.eq(target.view_as(pred)).sum().item()

#     original_accuracy = 100 * original_correct / len(test_loader.dataset)
#     adversarial_accuracy = 100 * adversarial_correct / len(test_loader.dataset)
#     test_loss /= len(test_loader.dataset)
#     test_loss_adv /= len(test_loader.dataset)

#     run_duration = time.time()-run_start_time
#     print("Run duration = {}".format(run_duration))
    
#     return original_accuracy, adversarial_accuracy, test_loss, test_loss_adv

# Given model and test data, return true_labels and predictions.
def evaluate_tsne(args, model, device, test_loader, is_test_set=False):
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)
    n = 0
    pred_labels = []
    with torch.no_grad():
        for data, _ in test_loader:
            n += 1
            if n > 1: break
            data = data.to(device)
            output = model(data)

            pred_labels.append(output)  ## change here

    pred_labels = torch.cat(pred_labels, dim=0)
    return pred_labels


def extract_prediction(val_loader, model, args):
    """
    Run evaluation
    """
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)
    start = time.time()

    y_pred = []
    y_true = []

    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            pred = F.softmax(output, dim=1)

            y_true.append(target.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

        if i % args.print_freq == 0:
            end = time.time()
            print('Scores: [{0}/{1}]\t'
                  'Time {2:.2f}'.format(
                i, len(val_loader), end - start))
            start = time.time()

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    print('* prediction shape = ', y_pred.shape)
    print('* ground truth shape = ', y_true.shape)

    return y_pred, y_true

def test_calibration(val_loader, model, args):

    y_pred, y_true = extract_prediction(val_loader, model, args)
    ece = expected_calibration_error(y_true, y_pred)
    nll = F.nll_loss(torch.from_numpy(y_pred).log(), torch.from_numpy(y_true), reduction="mean")
    print('* ECE = {}'.format(ece))
    print('* NLL = {}'.format(nll))

    return ece, nll


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

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            output = model(data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            print('Inference time per sample: {}'.format(inference_time))
            softmax_preds = torch.nn.Softmax(dim=1)(input=output)
            # nll_loss = F.nll_loss(output, target, reduction='mean').item() #F.nll_loss(torch.log(softmax_preds), target, reduction='mean').item()  # NLL
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: NLL: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n), test_loss

def evaluate_ensemble(args, model, device, test_loader, is_test_set=False):
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)
    test_loss = 0
    correct = 0
    n = 0
    current_fold_preds = []
    test_data = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            softmax_preds = torch.nn.Softmax(dim=1)(input=output)
            current_fold_preds.append(softmax_preds)
            test_data.append(target)
    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    return current_fold_preds, test_data


def evaluate_ensemble_bd(args, model, device, test_loader, is_test_set=False):
    '''
    breakdown version of ensemble
    '''
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)
    test_loss = 0
    correct = 0
    n = 0
    current_fold_preds = []
    test_data = []
    for i in range(100):
        with torch.no_grad():
            for data, target in test_loader:
                data = data[target==i]
                target = target[target==i]
                if data.nelement() == 0: continue
                data, target = data.to(device), target.to(device)
                output = model(data)
                softmax_preds = torch.exp(output) #torch.nn.Softmax(dim=1)(input=output)
                current_fold_preds.append(softmax_preds)
                test_data.append(target)
    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    return current_fold_preds, test_data

def evaluate_ensemble_KD(args, model, device, test_loader, is_test_set=False):
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)
    test_loss = 0
    correct = 0
    n = 0
    current_fold_preds = []
    test_data = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            # softmax_preds = torch.nn.functional.log_softmax(input=logits, dim=1)
            current_fold_preds.append(logits)
            test_data.append(target)
    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    return current_fold_preds, test_data

def evaluate_ensemble_KD_intermediate(args, model, device, test_loader, layer_name, is_test_set=False):
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for name, layer in model.named_modules():
        if name == layer_name:
        # if isinstance(layer, nn.Linear):
            layer.register_forward_hook(get_activation(name))

    current_preds = collections.defaultdict(list)
    test_data = []
    with torch.no_grad():
        for index, (data, target) in enumerate(test_loader):
            if index > 0: break
            data, target = data.to(device), target.to(device)
            logits = model(data)
            test_data.append(target)

            for key in activation:
                current_preds[key].append(activation[key])

    for key in current_preds.keys():
        current_preds[key] = torch.cat(current_preds[key], dim=0)
    return current_preds

def loss_fn_kd_models(scores_model, target_scores_model):

    scores = []
    for key in scores_model:
        scores.append(loss_fn_kd(scores_model[key], target_scores_model[key]).cpu().data.numpy())

    return scores

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


def test_ensemble_part(model, val_loader, args):
    """
    Run evaluation
    """
    model.eval()
    if args.use_bnn:
        model.set_test_mean(False)
    current_fold_preds = []
    test_data = []
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            softmax_preds = torch.nn.Softmax(dim=1)(input=output)
            current_fold_preds.append(softmax_preds)
            test_data.append(target)

    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    pred = current_fold_preds.argmax(dim=1, keepdim=True)
    correct = pred.eq(test_data.view_as(pred)).sum().item()
    n = test_data.shape[0]
    print('\n{}: Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation',
        correct, n, 100. * correct / float(n)))

    return current_fold_preds, test_data

def test_ensemble(val_loader, model, args):
    print_and_log("=> loading checkpoint '{}'".format(args.resume))
    all_folds_preds = []
    model_files = os.listdir(args.resume)
    model_files = sorted_nicely(model_files)

    for file in range(0, len(model_files)):
        print(model_files[file])
        if 'SeBayS' in args.resume or 'EDST' in args.resume:
            checkpoint = torch.load(args.resume + str(model_files[file]))
        elif 'BayS' in args.resume or 'DST in args.resume':
            checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        current_fold_preds, target = test_ensemble_part(model, val_loader, args)
        all_folds_preds.append(current_fold_preds)

    output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
    print(output_mean.size())

    pred = output_mean.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    n = target.shape[0]
    print('\n{}: Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation',
            correct, n, 100. * correct / float(n)))

def ensemble_calibration_corruption(model, args):

    all_folds_preds_c = []
    if args.dataset == 'cifar10':
        data_path = './_dataset/CIFAR-10-C/'
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2470, 0.2435, 0.2616))
                    ])
    elif args.dataset == 'cifar100':
        data_path = './_dataset/CIFAR-100-C/'
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                    ])
    print_and_log("=> loading checkpoint '{}'".format(args.resume))
    model_files = os.listdir(args.resume)
    model_files = sorted_nicely(model_files)

    for file in range(0, len(model_files)):
        all_preds = []
        all_targets = []

        print(model_files[file])
        if 'SeBayS' in args.resume or 'EDST' in args.resume:
            checkpoint = torch.load(args.resume + str(model_files[file]))
        elif 'BayS' in args.resume or 'DST' in args.resume:
            checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        file_list = os.listdir(data_path)                
        file_list.sort()

        # for file_name in file_list:
        #     if not file_name == 'labels.npy':
        #         attack_type = file_name[:-len('.npy')]
        #         for severity in range(1,6):
        #             print('attack_type={}'.format(attack_type), 'severity={}'.format(severity))
        #             cifar10c_test_loader = cifar_c_dataloaders(args.batch_size, cifar_c_path, severity, attack_type, normalize)
        #             y_pred, y_true = extract_prediction(cifar10c_test_loader, model,args)

        #             print('* Acc = {}'.format(np.mean(np.argmax(y_pred, 1)==y_true)))
        #             all_preds.append(y_pred)
        #             all_targets.append(y_true)
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
                print(len(corrupt_test_loader.dataset))
                y_pred, y_true = extract_prediction(corrupt_test_loader, model, args)

                all_preds.append(y_pred)
                all_targets.append(y_true)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        print('* c-Acc = {}'.format(np.mean(np.argmax(all_preds, 1)==all_targets)))
        all_folds_preds_c.append(all_preds)

        ece = expected_calibration_error(all_targets, all_preds)
        nll = F.nll_loss(torch.from_numpy(all_preds).log(), torch.from_numpy(all_targets).long(), reduction="mean")
        print('* c-ECE = {}'.format(ece))
        print('* c-NLL = {}'.format(nll))

    output_mean = np.mean(np.stack(all_folds_preds_c, 0), 0)
    print(output_mean.shape)

    ece = expected_calibration_error(all_targets, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(all_targets).long(), reduction="mean")
    print('* c-ECE = {}'.format(ece))
    print('* c-NLL = {}'.format(nll))
    print('* c-Acc = {}'.format(np.mean(np.argmax(output_mean, 1)==all_targets)))

def ensemble_calibration(val_loader, model, args):
    print_and_log("=> loading checkpoint '{}'".format(args.resume))
    all_folds_preds = []
    model_files = os.listdir(args.resume)
    model_files = sorted_nicely(model_files)

    for file in range(0, len(model_files)):
        print(model_files[file])
        if 'SeBayS' in args.resume or 'EDST' in args.resume:
            checkpoint = torch.load(args.resume + str(model_files[file]))
        elif 'BayS' in args.resume or 'DST' in args.resume:
            checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
                         
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.cuda()
        y_pred, y_true = extract_prediction(val_loader, model, args)
        all_folds_preds.append(y_pred)


        ece = expected_calibration_error(y_true, y_pred)
        nll = F.nll_loss(torch.from_numpy(y_pred).log(), torch.from_numpy(y_true), reduction="mean")
        print('* ECE = {}'.format(ece))
        print('* NLL = {}'.format(nll))

    output_mean = np.mean(np.stack(all_folds_preds, 0), 0)
    print(output_mean.shape)

    ece = expected_calibration_error(y_true, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(y_true), reduction="mean")
    print('* Ensemble ECE = {}'.format(ece))
    print('* Ensemble NLL = {}'.format(nll))

def ed(param_name, default=None):
    return os.environ.get(param_name, default)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')

    parser.add_argument('--mode', type=str, help='disagreement model')
    # parser.add_argument('--dense_allocation', default=ed('DENSE_ALLOCATION'), type=float,
    #                     help='percentage of dense parameters allowed. if None, pruning will not be used. must be on the interval (0, 1]')
    # parser.add_argument('--delta', default=ed('DELTA', 1000), type=int,
    #                     help='delta param for pruning')
    parser.add_argument('--grad_accumulation_n', default=ed('GRAD_ACCUMULATION_N', 1), type=int)
    # parser.add_argument('--alpha', default=ed('ALPHA', 0.3), type=float,
    #                     help='alpha param for pruning')
    parser.add_argument('--static_topo', default=ed('STATIC_TOPO', 0), type=int, help='if 1, use random sparsity topo and remain static')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--epochs-explo', type=int, default=150, metavar='N',
                        help='training time of exploration phase')
    parser.add_argument('--model-num', type=int, default=3,
                        help='How many subnetworks to produce, default=3')
    parser.add_argument('--init_lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    
    parser.add_argument('--step_size', type=float, default=ed('DECAY_STEP', 80), metavar='DS')
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save_model', default=1, type=bool,
                        help='For Saving the current Model')
    parser.add_argument('--exp_name', default='name', type=str)
    parser.add_argument('--load_ckpt', default='', type=str)
    # parser.add_argument('--nowandb', default=False, action='store_true')

    parser.add_argument('--corrupt', action='store_true', default=False, help='Corrupt dataset evaluation. Default: True.')
    parser.add_argument('--epsilon', default = 0.1, type = float, help = 'epsilon in FGSM attack')

    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--ood-dataset', type=str, default='svhn')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--mgpu', action='store_true', help='Enable snip initialization. Default: True.')

    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--update-frequency', type=int, default=1000, metavar='N', help='how many iterations to train between parameter exploration')

    # bnn parameter
    parser.add_argument('--use_bnn', default=False, action='store_true')
    parser.add_argument('--prior_mean', default=0, type=float)
    parser.add_argument('--prior_std', default=0.2, type=float)
    parser.add_argument('--kl_scale', default=0.01, type=float)
    parser.add_argument('--eval_bnn', default=0, type=int, help='if 0, eval normal nn; if 1, eval bnn with mean; if >1, eval bnn with mean and sample eval_bnn times')
    parser.add_argument('--same_noise', default=False, action='store_true')

    parser.add_argument('--drop-criteria', default='SNR_mean_abs', type=str, choices=['mean', 'E_mean_abs', 'snr', 'E_exp_mean_abs', 'SNR_mean_abs', 'SNR_exp_mean_abs'])
    parser.add_argument('--lambda_exp', default=1.0, type=float)
    parser.add_argument('--add_reg_sigma', default=False, action='store_true', help='if true, add regularization term for sigma to prevent zeros')
    parser.add_argument('--grow_std', default='mean', type=str, choices=['mean', 'eps'])
    parser.add_argument('--grow_mean_grad', default=False, action='store_true', help='if true, grow mean grad')
    parser.add_argument('--lr_std', default=0.01, type=float, help='lr for std')

    print('\n\n\n###############################################################################\n\n\n')

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)
    # milestone1, milestone2 = milestone_calculation(args)
    # print(f'learning rate recycle milestone1 is {milestone1} milestone2 is {milestone2}')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))

        if args.dataset == 'cifar10':
            _ , _ , test_loader = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads)
            outputs = 10
        elif args.dataset == 'cifar100':
            _ , _ , test_loader = get_cifar100_dataloaders(args, args.valid_split, max_threads=args.max_threads)
            outputs = 100

        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:
            cls, cls_args = models[args.model]
            if args.model == 'rn-18':
                cls_args[0] = outputs
                model = cls(*(cls_args + [args.use_bnn,args.prior_mean,args.prior_std,args.same_noise,\
                                        args.save_features, args.bench])).to(device)
            elif args.model == 'resnet-32' or args.model == 'resnet-56':
                cls_args[1] = outputs
                model = cls(*(cls_args + [args.use_bnn,args.prior_mean,args.prior_std,args.same_noise,\
                                        args.save_features, args.bench])).to(device)
            else: 
                cls_args[2] = outputs
                model = cls(*(cls_args + [args.use_bnn,args.prior_mean,args.prior_std,args.same_noise,\
                                        args.save_features,args.bench])).to(device)
            print_and_log(args.model+' on '+args.dataset)
            print_and_log('=' * 60)


        if args.resume:
                ###############################################################################
                #                          disagreement                                      #
                ##############################################################################
                if 'disagreement' in args.mode:

                    labels = []
                    val_acc = []
                    nll_loss = []

                    print_and_log("=> loading checkpoint '{}'".format(args.resume))

                    model_files = os.listdir(args.resume)
                    model_files = sorted_nicely(model_files)
                    all_folds_preds = []

                    for file in range(0, len(model_files)):

                        print(model_files[file])
                        if 'SeBayS' in args.resume or 'EDST' in args.resume:
                            checkpoint = torch.load(args.resume + str(model_files[file]))
                        elif 'BayS' in args.resume or 'DST' in args.resume:
                            checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)

                        current_fold_preds, target = evaluate_ensemble(args, model, device, test_loader)
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
                    # plt.savefig("./plots/" + "D_RN32_CF10_dense.pdf")
                ###############################################################################
                #                          predict                                            #
                ###############################################################################
                if 'predict' in args.mode:
                    print_and_log("=> loading checkpoint '{}'".format(args.resume))

                    model_files = os.listdir(args.resume)
                    model_files = sorted_nicely(model_files)

                    all_folds_preds = []
                    labels = []
                    val_acc = []
                    nll_loss = []
                    for file in range(0, len(model_files)):

                        print(model_files[file])
                        if 'SeBayS' in args.resume or 'EDST' in args.resume:
                            checkpoint = torch.load(args.resume + str(model_files[file]))
                        elif 'BayS' in args.resume or 'DST' in args.resume:
                            checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
                            
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)

                        print_and_log('Testing...')
                        indi_acc, indi_loss = evaluate(args, model, device, test_loader)
                        val_acc.append(indi_acc)
                        nll_loss.append(indi_loss)

                        current_fold_preds, target = evaluate_ensemble(args, model, device, test_loader)
                        all_folds_preds.append(current_fold_preds)
                        labels.append(target)

                    print(torch.equal(labels[0], labels[1]))

                    individual_acc_mean = np.array(val_acc).mean(axis=0)
                    individual_acc_std = np.array(val_acc).std(axis=0)
                    individual_nll_mean = np.array(nll_loss).mean(axis=0)
                    individual_nll_std = np.array(nll_loss).std(axis=0)
                    print('Averaged individual model: acc is {:.3f}% and std is {:.4f}'.format(individual_acc_mean*100, individual_acc_std*100))
                    print('Averaged individual model: NLL is {:.4f} and std is {:.4f}'.format(individual_nll_mean, individual_nll_std))

                    output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
                    test_loss = F.nll_loss(torch.log(output_mean), target, reduction='mean').item()  # sum up batch loss
                    print(f"Ensemble NLL is {test_loss}")
                    pred = output_mean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct = pred.eq(target.view_as(pred)).sum().item()
                    n = target.shape[0]
                    print_and_log('\n{}: Ensemble Accuracy is: {}/{} ({:.3f}%)\n'.format(
                        'Test evaluation',
                         correct, n, 100. * correct / float(n)))
                ###############################################################################
                #                          adversarial                                        #
                ###############################################################################
                if 'adv' in args.mode:
                    print_and_log("=> loading checkpoint '{}'".format(args.resume))

                    model_files = os.listdir(args.resume)
                    model_files = sorted_nicely(model_files)

                    if args.dataset=='cifar10':
                        mean = [0.4914, 0.4822, 0.4465]
                        std = [0.2470, 0.2435, 0.2616]
                    elif args.dataset == 'cifar100':
                        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
                        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

                    adversarial_loaders = []
                    for file in range(0, len(model_files)):

                        print(model_files[file])
                        if 'SeBayS' in args.resume or 'EDST' in args.resume:
                            checkpoint = torch.load(args.resume + str(model_files[file]))
                        elif 'BayS' in args.resume or 'DST' in args.resume:
                            checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
                            
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)

                        print_and_log('Generating Adversarial Datasets...')
                        adversarial_loader = generate_adversarial_examples(args, model, device, test_loader, mean, std, epsilon=args.epsilon)
                        adversarial_loaders.append(adversarial_loader)

                    
                    for i, adv_loader in enumerate(adversarial_loaders):
                        print(f"Adversarial dataset {i}")
                        all_folds_adv_preds = []
                        labels = []
                        for file in range(0, len(model_files)):

                            print(model_files[file])
                            if 'SeBayS' in args.resume or 'EDST' in args.resume:
                                checkpoint = torch.load(args.resume + str(model_files[file]))
                            elif 'BayS' in args.resume or 'DST' in args.resume:
                                checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
                                
                            if 'state_dict' in checkpoint:
                                model.load_state_dict(checkpoint['state_dict'])
                            else:
                                model.load_state_dict(checkpoint)

                            print_and_log('Testing...')

                        
                            current_fold_adv_preds, target = evaluate_adv_ensemble(args, model, device, adv_loader)
                            all_folds_adv_preds.append(current_fold_adv_preds)
                            labels.append(target)

                        print(torch.equal(labels[0], labels[1]))
                        
                        output_adv_mean = torch.mean(torch.stack(all_folds_adv_preds, dim=0), dim=0)
                        pred_adv = output_adv_mean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct_adv = pred_adv.eq(target.view_as(pred_adv)).sum().item()
                        n = target.shape[0]
                        print_and_log('\n{}: Ensemble Adversarial Accuracy is: {}/{} ({:.3f}%)\n'.format(
                            'Test evaluation',
                            correct_adv, n, 100. * correct_adv / float(n)))
                    
                ###############################################################################
                #                          adversarial                                        #
                ###############################################################################
                if 'ood' in args.mode:
                    if args.dataset == 'cifar10':
                        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    (0.2470, 0.2435, 0.2616))
                        test_transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize
                                            ])
                        if args.ood_dataset == 'svhn':
                            test_loader = torch.utils.data.DataLoader(
                                    make_ood_dataset(
                                        CIFAR10_M('./_dataset', train=False, download=True, transform=test_transform, in_dist_label=1),
                                        SVHN_M('./_dataset', split='test', download=True, transform=test_transform, in_dist_label=0)
                                    ),
                                    batch_size=args.test_batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True)

                        elif args.ood_dataset == 'cifar100':
                            test_loader = torch.utils.data.DataLoader(
                                    make_ood_dataset(
                                        CIFAR10_M('./_dataset', train=False, download=True, transform=test_transform, in_dist_label=1),
                                        CIFAR100_M('./_dataset', train=False, download=True, transform=test_transform, in_dist_label=0)
                                    ),
                                    batch_size=args.test_batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True)
                        
                    elif args.dataset == 'cifar100':
                        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                        test_transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize
                                            ])
                        if args.ood_dataset == 'svhn':
                            test_loader = torch.utils.data.DataLoader(
                                    make_ood_dataset(
                                        SVHN_M('./_dataset', split='test', download=True, transform=test_transform, in_dist_label=0),
                                        CIFAR100_M('./_dataset', train=False, download=True, transform=test_transform, in_dist_label=1)
                                    ),
                                    batch_size=args.test_batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True)

                        elif args.ood_dataset == 'cifar10':
                            test_loader = torch.utils.data.DataLoader(
                                    make_ood_dataset(
                                        CIFAR10_M('./_dataset', train=False, download=True, transform=test_transform, in_dist_label=0),
                                        CIFAR100_M('./_dataset', train=False, download=True, transform=test_transform, in_dist_label=1)
                                    ),
                                    batch_size=args.test_batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True)

                    print_and_log("=> loading checkpoint '{}'".format(args.resume))

                    model_files = os.listdir(args.resume)
                    model_files = sorted_nicely(model_files)

                    all_folds_scores = []
                    val_AUROC = []
                    val_AUPRC = []
                    val_Specificity = []

                    for file in range(0, len(model_files)):

                        print(model_files[file])
                        if 'SeBayS' in args.resume or 'EDST' in args.resume:
                            checkpoint = torch.load(args.resume + str(model_files[file]))
                        elif 'BayS' in args.resume or 'DST' in args.resume:
                            checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
                            
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)

                        print_and_log('Testing...')

                        ood_labels_G, ood_scores_G = evaluate_ood(args, model, device, test_loader)
                        indi_AUROC, indi_AUPRC, indi_Specificity = eval_metrics(ood_labels_G,ood_scores_G)
                        val_AUROC.append(indi_AUROC)
                        val_AUPRC.append(indi_AUPRC)
                        val_Specificity.append(indi_Specificity)

                        # all_folds_labels.append(ood_labels_G)
                        # print(ood_labels_G==np.mean(np.stack(all_folds_labels, axis=0), axis=0))
                        all_folds_scores.append(ood_scores_G)
                        ens_AUROC, ens_AUPRC, ens_Specificity = eval_metrics(ood_labels_G,
                                                    np.mean(np.stack(all_folds_scores, axis=0), axis=0))
                        
                        print('Individual Model Results: AUROC is {:.4f}, AUPRC is {:.4f}, Specificity is {:.4f}\n'.format(indi_AUROC, indi_AUPRC, indi_Specificity))
                        individual_AUROC_mean = np.array(val_AUROC).mean(axis=0)
                        individual_AUROC_std = np.array(val_AUROC).std(axis=0)
                        individual_AUPRC_mean = np.array(val_AUPRC).mean(axis=0)
                        individual_AUPRC_std = np.array(val_AUPRC).std(axis=0)
                        individual_Specificity_mean = np.array(val_Specificity).mean(axis=0)
                        individual_Specificity_std = np.array(val_Specificity).std(axis=0)
                        print(f"Averaged individual model: AUROC is {individual_AUROC_mean} and its std is {individual_AUROC_std}")
                        print(f"Averaged individual model: AUPRC is {individual_AUPRC_mean} and its std is {individual_AUPRC_std}")
                        print(f"Averaged individual model: Specificity is {individual_Specificity_mean} and its std is {individual_Specificity_std}\n")

                        print('Sequential Ensemble Model Results: AUROC is {:.4f}, AUPRC is {:.4f}, Specificity is {:.4f}\n'.format(ens_AUROC, ens_AUPRC, ens_Specificity))

                
                ###############################################################################
                #                          ensemble calibration                              #
                ##############################################################################
                if 'calibration' in args.mode:
                    # cifar_c_path = args.cf_c_path
                    # if args.dataset == 'cifar10':
                    #     cor_path = 'CIFAR-10-C'
                    #     normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                    #                                      (0.2470, 0.2435, 0.2616))
                    # elif args.dataset == 'cifar100':
                    #     cor_path = 'CIFAR-100-C'
                    #     normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    #                                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                    # test_ensemble(test_loader, model, args)
                    # ensemble_calibration_corruption(model, args, normalize)
                    if args.corrupt:
                        ensemble_calibration_corruption(model, args)
                    else:
                        ensemble_calibration(test_loader, model, args)
                    

                ###############################################################################
                #                          ensemble DL                                       #
                ###############################################################################
                if 'KD' in args.mode:
                    print_and_log("=> loading checkpoint '{}'".format(args.resume))
                    all_folds_preds = []
                    model_files = os.listdir(args.resume)
                    model_files = sorted_nicely(model_files)

                    all_folds_preds = []
                    for file in range(0, len(model_files)):
                        print(model_files[file])
                        if 'SeBayS' in args.resume or 'EDST' in args.resume:
                            checkpoint = torch.load(args.resume + str(model_files[file]))
                        elif 'BayS' in args.resume or 'DST' in args.resume:
                            checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))

                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)

                        print_and_log('Testing...')
                        current_fold_preds, data = evaluate_ensemble_KD(args, model, device, test_loader)
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
                #                          ensemble tsne                                     #
                ###############################################################################
                if 'tsne' == args.mode:
                    print_and_log("=> loading checkpoint '{}'".format(args.resume))
                    if args.dataset == 'cifar10':
                        if 'SeBayS' in args.resume:
                            file_path = "tsne_CF10_SeBayS_data.pt"
                        elif 'BayS' in args.resume:
                            file_path = "tsne_CF10_BayS_data.pt"
                    elif args.dataset == 'cifar100':
                        if 'SeBayS' in args.resume:
                            file_path = "tsne_CF100_SeBayS_data.pt"
                        elif 'BayS' in args.resume:
                            file_path = "tsne_CF100_BayS_data.pt"
                    if not os.path.exists(file_path):
                        independent_models_tsne = [dir for dir in os.listdir(args.resume) if dir != 'Subnet_0.0']
                        independent_models_tsne = sorted_nicely(independent_models_tsne)
                        print(independent_models_tsne)
                        predictions_for_tsne = []
                        for i in range(len(independent_models_tsne)):
                            subdir = independent_models_tsne[i]
                            model_files = os.listdir(os.path.join(args.resume, subdir))
                            model_files = sorted_nicely(model_files)
                            predictions = []

                            for model_file in range(125, len(model_files), 2):
                                
                                # if model_file < 28: continue
                                print(model_files[model_file])

                                # if model_files[model_file] == 'initial.pt':  # Skip if the model checkpoint didn't serialize properly.
                                #     continue
                                checkpoint = torch.load(os.path.join(args.resume, subdir, model_files[model_file]))
                                model.load_state_dict(checkpoint['state_dict'])
                                preds = evaluate_tsne(args, model, device, test_loader)
                                print(preds.size())
                                predictions.append(preds)
                            predictions = torch.stack(predictions, dim=0)
                            print(predictions.size())
                            predictions_for_tsne.append(predictions)

                        predictions_for_tsne = torch.stack(predictions_for_tsne, dim=0)
                        print(predictions_for_tsne.size())

                        predictions_for_tsne = predictions_for_tsne.view(-1, predictions_for_tsne.size()[-2] * predictions_for_tsne.size()[-1])
                        print(predictions_for_tsne.size())

                        torch.save(predictions_for_tsne, file_path)
                    else:
                        predictions_for_tsne = torch.load(file_path)
                        print(predictions_for_tsne.size())                    
                    

                    # model_files = os.listdir(args.resume)
                    # model_files = sorted_nicely(model_files)

                    # # independent_models_tsne = os.listdir(args.resume)
                    # predictions_for_tsne = []
                    # for file in range(0, len(model_files)):
                    #     print(model_files[file])
                    #     if 'SeBayS' in args.resume or 'EDST' in args.resume:
                    #         checkpoint = torch.load(args.resume + str(model_files[file]))
                    #     elif 'BayS' in args.resume or 'DST' in args.resume:
                    #         checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
                            
                    #     if 'state_dict' in checkpoint:
                    #         model.load_state_dict(checkpoint['state_dict'])
                    #     else:
                    #         model.load_state_dict(checkpoint)

                    # # for i in range(len(independent_models_tsne)):
                    # #     subdir = independent_models_tsne[i]
                    # #     print(subdir)
                    # #     model_files = os.listdir(args.resume + subdir)
                    # #     model_files = sorted_nicely(model_files)
                    #     # predictions = []

                    #     # for model_file in range(0,len(model_files),2):
                    #     #     if model_file < 145: continue

                    #     #     if model_files[model_file] == 'initial.pt':  # this check is because this model checkpoint didn't serialize properly.
                    #     #         continue
                    #     #     checkpoint = torch.load(args.resume + str(subdir) + '/'+model_files[model_file])
                    #         # model.load_state_dict(checkpoint)
                    #     preds =  evaluate_tsne(args, model, device, test_loader)
                    #     predictions_for_tsne.append(preds)

                    # # predictions_for_tsne = torch.stack(predictions_for_tsne, dim=0)
                    # # print(predictions_for_tsne.size())

                    # predictions_for_tsne = predictions_for_tsne.view(-1, predictions_for_tsne.size()[-2]*predictions_for_tsne.size()[-1])
                    # # [60, 10000]
                    # print(predictions_for_tsne.size())
                    # # torch.save(predictions_for_tsne, args.resume + "/tsne_pred.pt")
                    # # predictions_for_tsne = predictions_for_tsne.numpy()
                    NUM_TRAJECTORIES = 3
                    fontsize = 20
                    tsne = TSNE(n_components=2)
                    # compute tsne
                    trajectory_embed = []
                    prediction_embed = tsne.fit_transform(predictions_for_tsne.cpu().numpy())
                    # trajectory_embed = prediction_embed.view(NUM_TRAJECTORIES, -1, 2)
                    trajectory_embed = np.reshape(prediction_embed,(NUM_TRAJECTORIES, -1, 2))

                    plt.figure(constrained_layout=True, figsize=(6, 6))
                    plt.style.use('default')
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
                    ax.grid(False)
                    ax.set_facecolor('white')
                    # plt.xticks(fontsize=15)
                    # plt.yticks(fontsize=15)
                    plt.legend(fontsize=15, loc=4)
                    if args.dataset == 'cifar10':
                        if 'SeBayS' in args.resume:
                            plt.savefig('WRN_CF10_SeBayS_tsne.pdf',dpi=300)
                            plt.savefig('WRN_CF10_SeBayS_tsne.png')
                        elif 'BayS' in args.resume:
                            plt.savefig('WRN_CF10_BayS_tsne.pdf',dpi=300)
                            plt.savefig('WRN_CF10_BayS_tsne.png')
                    elif args.dataset == 'cifar100':
                        if 'SeBayS' in args.resume:
                            plt.savefig('WRN_CF100_SeBayS_tsne.pdf',dpi=300)
                            plt.savefig('WRN_CF100_SeBayS_tsne.png')
                        elif 'BayS' in args.resume:
                            plt.savefig('WRN_CF100_BayS_tsne.pdf',dpi=300)
                            plt.savefig('WRN_CF100_BayS_tsne.png')
                    # reshape
                    # trajectory_embed = prediction_embed.reshape([NUM_TRAJECTORIES, -1, 2])
                    # print('[INFO] Shape of reshaped tensor: ', trajectory_embed.shape)



if __name__ == '__main__':
   main()