#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

lambda_CH=0.5
inject_type=FiLM
inject_layers=11,12,13,14,15,16,17,18,19,20

name=exp_android_loss_CH_"$lambda_CH"_emb
python train.py --dataroot ../tat_asr_aligned/UNA-GAN --name "$name" --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints
python test.py --name "$name" --source_folder 1 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
python test.py --name "$name" --source_folder 2 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
python test.py --name "$name" --source_folder 3 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
python test.py --name "$name" --source_folder 4 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
python test.py --name "$name" --source_folder 5 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
python test.py --name "$name" --source_folder 6 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
python test.py --name "$name" --source_folder 7 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
python test.py --name "$name" --source_folder 8 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
python test.py --name "$name" --source_folder 9 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
python test.py --name "$name" --source_folder 10 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
python test.py --name "$name" --source_folder 11 --lambda_CH "$lambda_CH" --inject_type "$inject_type" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
