#!/bin/bash

#SBATCH -J mii-llama2-70b

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00 
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-large

HF_NAME=/lcrc/project/ECP-EZ/ac.gwilkins/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080
MODEL_NAME=llama2-70b
module load amd-uprof anaconda3/2023-01-11
cd /home/ac.gwilkins/EASLI/energy-measurements/

conda activate mii


DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
deepspeed --num_gpus 2 --master_port 29502 mii-inference.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset alpaca

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
deepspeed --num_gpus 2 --master_port 29502 mii-inference.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset self-oss

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
deepspeed --num_gpus 2 --master_port 29502 mii-inference.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset orca
