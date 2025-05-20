#!/bin/bash
#SBATCH --job-name=SeBayS_ImageNet     # create a short name for your job
#SBATCH --account=csigeneral           # csigeneral-v mlg-core oddr
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=80G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=csi
#SBATCH --qos=csi
#SBATCH --time=240:00:00

##SBATCH --mail-type=BEGIN        # send email when job begins
##SBATCH --mail-type=END          # send email when job ends
##SBATCH --mail-user=sjantre@bnl.gov

#SBATCH --output=run_SeBayS_RN_50_I1000_%j.log

source ~/.bashrc
conda activate torch

#python $1multiproc.py --nproc_per_node 2 $1main.py --sparse_init ER --filename save/DST_individual/ --multiplier 1 --growth gradient --seed 17 --master_port 4545 -j5 -p 500 --arch resnet50 -c fanin --update_frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --epochs 310 --density 0.2 $2 ../data/
python $1multiproc.py --nproc_per_node 2 $1main_individual.py --sparse_init ERK --filename results --multiplier 1 --growth gradient --seed 17 --master_port 4545 -j5 -p 500 --arch resnet50 -c fanin --update_frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --epochs 100 --density 0.2 $2 ../../data/ImageNet/

conda deactivate