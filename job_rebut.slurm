#!/bin/bash
#SBATCH --job-name=MCD     # create a short name for your job
#SBATCH --account=csigeneral           # csigeneral-v mlg-core oddr
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=80G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=csi
#SBATCH --qos=csi
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

##SBATCH --mail-type=BEGIN        # send email when job begins
##SBATCH --mail-type=END          # send email when job ends
##SBATCH --mail-user=sjantre@bnl.gov

#SBATCH --output=run_MCD_WRN_28_10_C100_%j.log

source ~/.bashrc
conda activate torch

# # ===================================================================================================================================
# # SnapShot Ensemble
# # ===================================================================================================================================
# python main_SnapShot.py --model wrn-28-10 --data cifar10 --seed 1 --batch-size 128 \
#                 --epochs 250 --valid_split 0.1 --initial_lr 0.2 --l2 0.0005 --cycles 5

# python main_SnapShot.py --model wrn-28-10 --data cifar100 --seed 1 --batch-size 128 \
#                 --epochs 250 --valid_split 0.1 --initial_lr 0.2 --l2 0.0005 --cycles 5

# # ===================================================================================================================================
# # MC-Dropout Ensemble
# # ===================================================================================================================================
# python main_MCD.py --model wrn-28-10 --data cifar10 --seed 1 --batch-size 128 \
#                 --epochs 200 --valid_split 0.1 --init_lr 0.1 --l2 0.0005 --drop_rate 0.1 --mc_dropout

# python main_MCD.py --model wrn-28-10 --data cifar10 --seed 2 --batch-size 128 \
#                 --epochs 200 --valid_split 0.1 --init_lr 0.1 --l2 0.0005 --drop_rate 0.1 --mc_dropout

# python main_MCD.py --model wrn-28-10 --data cifar10 --seed 3 --batch-size 128 \
#                 --epochs 200 --valid_split 0.1 --init_lr 0.1 --l2 0.0005 --drop_rate 0.1 --mc_dropout

# python main_MCD.py --model wrn-28-10 --data cifar100 --seed 1 --batch-size 128 \
#                 --epochs 200 --valid_split 0.1 --init_lr 0.1 --l2 0.0005 --drop_rate 0.1 --mc_dropout

# python main_MCD.py --model wrn-28-10 --data cifar100 --seed 2 --batch-size 128 \
#                 --epochs 200 --valid_split 0.1 --init_lr 0.1 --l2 0.0005 --drop_rate 0.1 --mc_dropout

# python main_MCD.py --model wrn-28-10 --data cifar100 --seed 3 --batch-size 128 \
#                 --epochs 200 --valid_split 0.1 --init_lr 0.1 --l2 0.0005 --drop_rate 0.1 --mc_dropout