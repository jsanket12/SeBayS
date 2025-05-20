from __future__ import print_function
import os
import time
import argparse
import logging
import hashlib
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from rigl_torch.RigL import RigLScheduler

from models.sto_resnet import StoResNet18
from models.sto_wide_resnet import Sto_ResNet, StoWideResNet
from data.data_utils import get_cifar10_dataloaders, get_cifar100_dataloaders

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
models['rn-18'] = (StoResNet18, [10])
models['resnet-32'] = (Sto_ResNet, [32, 10])
models['resnet-56'] = (Sto_ResNet, [56, 100])
models['wrn-28-2'] = (StoWideResNet, [28, 2, 10])
models['wrn-28-10'] = (StoWideResNet, [28, 10, 10])
models['wrn-22-8'] = (StoWideResNet, [22, 8, 10])
models['wrn-16-8'] = (StoWideResNet, [16, 8, 10])
models['wrn-16-10'] = (StoWideResNet, [16, 10, 10])


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


def train(args, model, device, train_loader, optimizer, epoch, NTrain, pruner):
    model.train()
    if args.use_bnn:
        model.set_test_mean(False)
    flag = 0
    total_sample, correct, train_loss, loss_avg, lr_avg = 0., 0., 0., 0., 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.grow_mean_grad:
            if pruner.check_step_only():
                model.set_test_mean(True)
                flag = 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if args.use_bnn:
            loss_kl = model.kl()
            if args.kl_anneal > 0:
                loss_kl = loss_kl * min(1.0, epoch / args.kl_anneal)
            loss += loss_kl.div(NTrain)
        loss.backward()

        if pruner():
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_sample += target.shape[0]
        loss_avg = (loss_avg * batch_idx + loss.item()) / (batch_idx+1)
        lr_avg = (lr_avg * batch_idx + optimizer.param_groups[0]["lr"]) / (batch_idx+1)

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), correct, total_sample, 100. * correct / float(total_sample)))
            if args.dry_run:
                break
        if flag:
            model.set_test_mean(False)
            flag = 0

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary' ,
        train_loss/batch_idx, correct, total_sample, 100. * correct / float(total_sample)))

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    if args.use_bnn:
        model.set_test_mean(True)
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

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%), ECE: {:.4f}\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n), ece))
    
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
    #     'Test evaluation' if is_test_set else 'Evaluation',
    #     test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)

def ed(param_name, default=None):
    return os.environ.get(param_name, default)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')

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
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--init_lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    
    parser.add_argument('--step_size', type=float, default=ed('DECAY_STEP', 80), metavar='DS')
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # parser.add_argument('--clip', default = 1, type = float, help = 'Gradient clipping value')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--ens_seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save_model', default=1, type=bool,
                        help='For Saving the current Model')
    parser.add_argument('--exp_name', default='name', type=str)
    parser.add_argument('--load_ckpt', default='', type=str)
    # parser.add_argument('--nowandb', default=False, action='store_true')

    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    # parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',help='number of data loading workers (default: 10)')
    parser.add_argument('--world-size', default=-1, type=int,help='number of nodes for distributed training')
    parser.add_argument('--mgpu', action='store_true', help='Enable snip initialization. Default: True.')
    
    # parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    # parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.')
    # parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate for DST.')
    parser.add_argument('--density', type=float, default=0.20, help='The density of the overall sparse network.')
    parser.add_argument('--sparsity_distribution', type=str, default='Uniform', help='sparse initialization')
    parser.add_argument('--update-frequency', type=int, default=1000, metavar='N', help='how many iterations to train between parameter exploration')

    # bnn parameter
    parser.add_argument('--use_bnn', default=False, action='store_true')
    parser.add_argument('--prior_mean', default=0, type=float)
    parser.add_argument('--prior_std', default=0.2, type=float)
    parser.add_argument('--kl_anneal', default = 0, type=int, help='steps for annealing beta from 0 to 1')
    parser.add_argument('--eval_bnn', default=0, type=int, help='if 0, eval normal nn; if 1, eval bnn with mean; if >1, eval bnn with mean and sample eval_bnn times')
    parser.add_argument('--same_noise', default=False, action='store_true')

    parser.add_argument('--drop-criteria', default='SNR_mean_abs', type=str, choices=['mean', 'E_mean_abs', 'snr', 'E_exp_mean_abs', 'SNR_mean_abs', 'SNR_exp_mean_abs'])
    parser.add_argument('--lambda_exp', default=1.0, type=float)
    parser.add_argument('--add_reg_sigma', default=False, action='store_true', help='if true, add regularization term for sigma to prevent zeros')
    parser.add_argument('--grow_std', default='mean', type=str, choices=['mean', 'eps'])
    parser.add_argument('--grow_mean_grad', default=False, action='store_true', help='if true, grow mean grad')
    parser.add_argument('--lr_std_diff', default=False, action='store_true')
    parser.add_argument('--steps_diff', default=False, action='store_true')

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if args.grow_mean_grad:
        assert 'growgrad_mean' in args.exp_name

    # if not args.nowandb:
    #     wandb.init(entity='entity', project='SeBayS', name=args.exp_name, config=args)

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.ens_seed+args.seed)
    random.seed(args.ens_seed+args.seed)
    np.random.seed(args.ens_seed+args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

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
                                        args.save_features, args.bench])).to(device)

            print_and_log(model)
            print_and_log('=' * 60)
            print_and_log(args.model)
            print_and_log('=' * 60)

            print_and_log('=' * 60)
            print_and_log('Prune mode: {0}'.format(args.drop_criteria))
            # print_and_log('Growth mode: {0}'.format(args.growth))
            print_and_log('=' * 60)

        if args.mgpu:
            print('Using multi gpus')
            model = torch.nn.DataParallel(model).to(device)

        optimizer = None
        if args.optimizer == 'sgd':
            if not args.lr_std_diff:
                optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if not name.endswith('posterior_std')], 
                                'weight_decay': args.weight_decay,
                                'lr': args.init_lr},
                            {'params': [param for name, param in model.named_parameters() if name.endswith('posterior_std')], 
                                'weight_decay': 0, 
                                'lr': args.init_lr}],
                            momentum=args.momentum)
            else:
                optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if not name.endswith('posterior_std')], 
                                'weight_decay': args.weight_decay,
                                'lr': args.init_lr},
                            {'params': [param for name, param in model.named_parameters() if name.endswith('posterior_std')], 
                                'weight_decay': 0, 
                                'lr': args.init_lr*0.1}],
                            momentum=args.momentum)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        NTrain = len(train_loader.dataset)

        if args.load_ckpt:
            ckpt = torch.load(args.load_ckpt, map_location='cpu')
            if 'model' in ckpt:
                ckpt = ckpt['model']
            model.load_state_dict(ckpt, strict=True)
        model.to(device)

        pruner = lambda: True
        if args.density is not None:
            T_end = int(1 * args.epochs * len(train_loader))
            pruner = RigLScheduler(model, optimizer, dense_allocation=args.density, alpha=args.death_rate, delta=args.update_frequency, 
                                   static_topo=args.static_topo, T_end=T_end, sparsity_distribution=args.sparsity_distribution, 
                                   ignore_linear_layers=False, grad_accumulation_n=args.grad_accumulation_n, args=args)

        # writer = SummaryWriter(log_dir='./graphs')

        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):

            print_and_log(f"{pruner} \n")
            
            if args.use_bnn:
                if not args.lr_std_diff:
                    if not args.steps_diff:
                        save_dir = 'results' + '/' + str(args.model) + '/' + str(args.data) +  '/' + 'wd_' + str(args.weight_decay) +  '/ens_seed_' + str(
                            args.ens_seed) + '/density_' + str(args.density) + '/' + 'BayS' + '/' + str(args.sparsity_distribution) + '/sig_' + str(
                            args.prior_std) +  '/anneal_' + str(args.kl_anneal) + '/lr_std_same/' + 'seed=%d' % (args.seed)
                    else:
                        save_dir = 'results' + '/' + str(args.model) + '/' + str(args.data) +  '/' + 'wd_' + str(args.weight_decay) +  '/ens_seed_' + str(
                            args.ens_seed) + '/density_' + str(args.density) + '/' + 'BayS' + '/' + str(args.sparsity_distribution) + '/sig_' + str(
                            args.prior_std) +  '/anneal_' + str(args.kl_anneal) + '/steps_diff/lr_std_same/' + 'seed=%d' % (args.seed)
                else:
                    if not args.steps_diff:
                        save_dir = 'results' + '/' + str(args.model) + '/' + str(args.data) +  '/' + 'wd_' + str(args.weight_decay) +  '/ens_seed_' + str(
                            args.ens_seed) + '/density_' + str(args.density) + '/' + 'BayS' + '/' + str(args.sparsity_distribution) + '/sig_' + str(
                            args.prior_std) +  '/anneal_' + str(args.kl_anneal) + '/lr_std_diff/' + 'seed=%d' % (args.seed)
                    else:
                        save_dir = 'results' + '/' + str(args.model) + '/' + str(args.data) +  '/' + 'wd_' + str(args.weight_decay) +  '/ens_seed_' + str(
                            args.ens_seed) + '/density_' + str(args.density) + '/' + 'BayS' + '/' + str(args.sparsity_distribution) + '/sig_' + str(
                            args.prior_std) +  '/anneal_' + str(args.kl_anneal) + '/steps_diff/lr_std_diff/' + 'seed=%d' % (args.seed)

            else:
                save_dir = 'results' + '/' + str(args.model) + '/' + str(args.data) +  '/' + 'wd_' + str(args.weight_decay) + '/ens_seed_' + str(
                        args.ens_seed) + '/density_' + str(args.density) + '/' + 'DST' + '/Uniform/' + 'seed=%d' % (args.seed)

            if not os.path.exists(save_dir): os.makedirs(save_dir)

            t0 = time.time()
            train(args, model, device, train_loader, optimizer, epoch, NTrain, pruner)

            if args.valid_split > 0.0:
                val_acc = evaluate(args, model, device, valid_loader)

            if val_acc > best_acc:
                print('Saving model')
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(save_dir, 'model.pth'))

            if not args.steps_diff:
                if epoch == int(3*args.epochs/5):
                    optimizer.param_groups[0]['lr'] = 0.01
                elif epoch == int(4*args.epochs/5):
                    optimizer.param_groups[0]['lr'] = 0.001
            else:
                if epoch == int(args.epochs/2):
                    optimizer.param_groups[0]['lr'] = 0.01
                elif epoch == int(args.epochs*3/4):
                    optimizer.param_groups[0]['lr'] = 0.001

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))

        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))


if __name__ == '__main__':
   main()