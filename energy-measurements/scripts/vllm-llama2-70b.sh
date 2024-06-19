#!/bin/bash

#SBATCH -J vllm-llama2-70b

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00 
#SBATCH --gres=gpu:4

HF_NAME="/lcrc/project/ECP-EZ/ac.gwilkins/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080"
MODEL_NAME="llama2-70b"
module load amd-uprof
cd /home/ac.gwilkins/EASLI/energy-measurements/

conda activate vllm

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 vllm.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset alpaca

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 vllm.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset self-oss

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p ./$MODEL_NAME/$DATE/$TIME/
AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 vllm.py --out_dir ./$MODEL_NAME/$DATE/$TIME --hf_name $HF_NAME --dataset orca