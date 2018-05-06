#!/bin/bash -l
#SBATCH --gres=gpu
#SBATCH -t 960
#SBATCH -o batch_outputs/slurm-%j.out
cp -r ../data/Boxing-v0 /tmp
cp -r ../data/Small_Boxing-v0 /tmp
filename=$1
shift
python $filename  $@ --data_dir /tmp
