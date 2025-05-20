#!/bin/bash
#SBATCH --job-name=EDST_E      # create a short name for your job
#SBATCH --account=csigeneral           # csigeneral-v mlg-core oddr
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=csi
#SBATCH --qos=csi
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

##SBATCH --mail-type=BEGIN        # send email when job begins
##SBATCH --mail-type=END          # send email when job ends
##SBATCH --mail-user=sjantre@bnl.gov

#SBATCH --output=run_EDST_ERK_RN32_%j.log

source ~/.bashrc
conda activate torch

# # ===================================================================================================================================
# # Dense DNN Ensemble
# # ===================================================================================================================================
# python main_DST.py --fix --model resnet-32 --data cifar10 --seed 1 --density 1 \
#                    --batch-size 128 --epochs 250 --valid_split 0.1 --init_lr 0.1 --weight_decay 0.0005

# python main_DST.py --fix --model resnet-56 --data cifar100 --seed 1 --density 1 \
#                     --batch-size 128 --epochs 250 --valid_split 0.1 --init_lr 0.1 --weight_decay 0.0005

# # ===================================================================================================================================
# # DST Ensemble
# # ===================================================================================================================================
# python main_BayS.py --model resnet-32 --data cifar10 --seed 1 \
#                     --update-frequency 1000 --batch-size 128 --epochs 250 --valid_split 0.1 --init_lr 0.1 --weight_decay 0.0005 \
#                     --density 0.2 --death-rate 0.5

# python main_DST.py --sparse --model resnet-32 --data cifar10 --seed 1 --sparse-init ERK \
#                     --update-frequency 1000 --batch-size 128 --epochs 250 --valid_split 0.1 --init_lr 0.1 --weight_decay 0.0005 \
#                     --density 0.2 --death-rate 0.5  --decay-schedule constant \
#                     --growth gradient --death magnitude --redistribution none


# python main_BayS.py --model resnet-56 --data cifar100 --seed 1 \
#                     --update-frequency 1000 --batch-size 128 --epochs 250 --valid_split 0.1 --init_lr 0.1 --weight_decay 0.0005 \
#                     --density 0.2 --death-rate 0.5

# python main_DST.py --sparse --model resnet-56 --data cifar100 --seed 1 --sparse-init ERK \
#                     --update-frequency 1000 --batch-size 128 --epochs 250 --valid_split 0.1  --init_lr 0.1 --weight_decay 0.0005 \
#                     --density 0.2 --death-rate 0.5 --decay-schedule constant \
#                     --growth gradient --death magnitude --redistribution none

# # ===================================================================================================================================
# # EDST Ensemble
# # ===================================================================================================================================
# python main_SeBayS.py --model resnet-32 --data cifar10 --seed 17 --epochs-explo 150 --model-num 3 \
#                       --update-frequency 1000 --batch-size 128 --epochs 450 --valid_split 0.1 --init_lr 0.1 --weight_decay 0.0005 \
#                       --density 0.2  --death-rate 0.5 --large-death-rate 0.8

# python main_EDST.py --sparse --model resnet-32 --data cifar10 --seed 17 --epochs-explo 150 --model-num 3 --sparse-init ERK \
#                       --update-frequency 1000 --batch-size 128 --epochs 450 --valid_split 0.1 --init_lr 0.1 --weight_decay 0.0005 \
#                       --density 0.2 --death-rate 0.5 --large-death-rate 0.8 --decay-schedule constant \
#                       --growth gradient --death magnitude --redistribution none


# python main_SeBayS.py --model resnet-56 --data cifar100 --seed 17 --epochs-explo 150 --model-num 3 \
#                       --update-frequency 1000 --batch-size 128 --epochs 450 --valid_split 0.1 --init_lr 0.1 --weight_decay 0.0005 \
#                       --density 0.2  --death-rate 0.5 --large-death-rate 0.8

# python main_EDST.py --sparse --model resnet-56 --data cifar100 --seed 17 --epochs-explo 150 --model-num 3 --sparse-init ERK \
#                       --update-frequency 1000 --batch-size 128 --epochs 450 --valid_split 0.1 --init_lr 0.1 --weight_decay 0.0005 \
#                       --density 0.2 --death-rate 0.5 --large-death-rate 0.8 --decay-schedule constant \
#                       --growth gradient --death magnitude --redistribution none

conda deactivate