#!/usr/bin/env bash

#SBATCH --job-name=stars
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

source "/p/haicluster/niesel1/venv_stars/activate.sh"

cd ..
cd src

# train
# srun --cpus-per-task="$SLURM_CPUS_PER_TASK" python train.py experiment=stars_experiment.yaml trainer=gpu logger=tensorboard # hparams_search=stars_optuna.yaml --multirun

# eval
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" python eval.py ckpt_path="/p/haicluster/niesel1/lightning-hydra-stars/logs/train/runs/2023-07-17_14-34-42/checkpoints/epoch_228.ckpt"