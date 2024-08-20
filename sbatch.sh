#!/bin/bash
#SBATCH --partition=ashton
#SBACTH --qos=ashton
#SBATCH --job-name=mongarud-manual
#SBATCH --output=model-distmult-one/distmult-manual_%j.log      # Standard output and error log
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

mkdir -p model-distmult-one

conda activate kge
python3 run.py --cuda --do_train --do_valid --do_test --data_path data/FB15k-237 --model DistMult --hidden_dim 500 --gamma 12.0 --batch_size 1024 --negative_sample_size 128 --learning_rate 0.001 --max_epochs 1 --save_path model-distmult-one