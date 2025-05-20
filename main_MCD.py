from __future__ import print_function
import os
import time
import argparse
import logging
import hashlib
import copy
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from mcd_models.mcd_wide_resnet import MCD_WideResNet
from data.data_utils import get_cifar10_dataloaders, get_cifar100_dataloaders

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

# Available models: key -> (model_class, [depth, widen_factor, num_classes, dropRate])
models = {}
models['wrn-28-10'] = (MCD_WideResNet, [28, 10, 10])

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)

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

def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}.log'.format(args.model, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))


    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary' ,
        train_loss/batch_idx, correct, n, 100. * correct / float(n)))
    
def evaluate(model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    preds = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]
            preds.append(F.softmax(output, dim=1))
            targets.append(target)

    test_loss /= float(n)
    ece = expected_calibration_error(torch.cat(targets, dim=0).cpu().numpy(),torch.cat(preds, dim=0).cpu().numpy())

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%), ECE: {:.4f},\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n), ece))
    
    return correct / float(n)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--init_lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                        help='dropout probability (default: 0.1)')
    parser.add_argument('--mc_dropout', action='store_true', default=True, help='Enables MC dropout.')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--data', type=str, default='cifar10', help='The dataset to use. Default: cifar10. Options: cifar10, cifar100.')
    parser.add_argument('--model', type=str, default='wrn-28-10', help='The model to use. Default: wrn-28-10.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=5e-4)
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--mgpu', action='store_true', help='Enable snip initialization. Default: True.')
    

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.data == 'cifar10':
        train_loader, valid_loader, _ = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads)
        outputs = 10
    elif args.data == 'cifar100':
        train_loader, valid_loader, _ = get_cifar100_dataloaders(args, args.valid_split, max_threads=args.max_threads)
        outputs = 100

    if args.model not in models:
        print('You need to select an existing model via the --model argument. Available models include: ')
        for key in models:
            print('\t{0}'.format(key))
        raise Exception('You need to select a model')
    else:
        cls, cls_args = models[args.model]
        cls_args[2] = outputs
        model = cls(*(cls_args + [args.drop_rate, args.mc_dropout,
                                  args.save_features, args.bench])).to(device)       
        print_and_log(model)
        print_and_log('=' * 60)
        print_and_log(args.model)
        print_and_log('=' * 60)

    if args.mgpu:
        print('Using multi gpus')
        model = torch.nn.DataParallel(model).to(device)

    base_lr = args.init_lr * args.batch_size / 128

    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=base_lr,momentum=args.momentum,weight_decay=args.l2,nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=base_lr,weight_decay=args.l2)
    else:
        print('Unknown optimizer: {0}'.format(args.optimizer))
        raise Exception('Unknown optimizer.')
    
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                     milestones=[int(args.epochs/2), int(args.epochs*3/4)], last_epoch=-1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                        milestones=[60, 120, 160], gamma=0.1, last_epoch=-1)

    best_acc = 0.0

    for epoch in range(args.epochs):

        save_dir = 'results_MCD' + '/' + str(args.model) + '/' + str(args.data) + '/wd_' + str(args.l2) + \
                    '/batch_' + str(args.batch_size) + '/Drop_rate_' + str(args.drop_rate) +  '/' + 'seed=%d' % (args.seed)

        if not os.path.exists(save_dir): os.makedirs(save_dir)

        t0 = time.time()
        train(args, model, device, train_loader, optimizer, epoch)

        if lr_scheduler: lr_scheduler.step()
        
        if args.valid_split > 0.0:
            val_acc = evaluate(model, device, valid_loader)

        if val_acc > best_acc:
                print('Saving model')
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(save_dir, 'model.pth'))

        print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))

    print_and_log('Best validation accuracy: {0:.3f}%'.format(best_acc * 100.0))

      
if __name__ == '__main__':
   main()