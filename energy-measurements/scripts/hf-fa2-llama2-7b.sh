#!/bin/bash

#SBATCH -J hf-fa2-llama2-7b

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-large

HF_NAME=/lcrc/project/ECP-EZ/ac.gwilkins/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/
MODEL_NAME=llama2-7b
module load amd-uprof
cd /home/ac.gwilkins/EASLI/energy-measurements/

conda activate hf-fa2

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
python3 hf-fa2.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset alpaca

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
python3 hf-fa2.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset self-oss

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
python3 hf-fa2.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset orca