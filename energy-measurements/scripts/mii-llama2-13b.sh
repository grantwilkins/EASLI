#!/bin/bash

#SBATCH -J mii-llama2-13b

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-large

HF_NAME=/lcrc/project/ECP-EZ/ac.gwilkins/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496
MODEL_NAME=llama2-13b
module load amd-uprof
cd /home/ac.gwilkins/EASLI/energy-measurements/

conda activate mii


DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
deepspeed --num_gpus 1 mii-inference.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset alpaca

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
deepspeed --num_gpus 1 mii-inference.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset self-oss

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
deepspeed --num_gpus 1 mii-inference.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset orca