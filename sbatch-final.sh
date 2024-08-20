#!/bin/bash
#SBATCH --partition=ashton
#SBACTH --qos=ashton
#SBATCH --job-name=mongarud-manual
#SBATCH --output=model-transE-trial/transE-manual_%j.log      # Standard output and error log
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

mkdir -p model-transE-trial

conda activate kge
python3 run.py --cuda --do_train --do_valid --do_test --data_path data/FB15k-237 --model TransE --hidden_dim 512 --gamma 12.0 --patience 20 --batch_size 1024 --negative_sample_size 512 --learning_rate 0.0001 --max_epochs 1000 --save_path model-transE-trial