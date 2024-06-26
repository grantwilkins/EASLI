from transformers import AutoTokenizer
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.handler.pandas_handler import PandasHandler
import argparse
import datetime
import pandas as pd
from pynvml.smi import nvidia_smi
import os
import psutil
import time
import numpy as np
from scipy import stats
import subprocess
from datasets import load_dataset
from optimum.nvidia import AutoModelForCausalLM

# from optimum.nvidia.pipelines import pipeline, Pipeline


def get_prompts(dataset_name: str) -> list[tuple[str, str]]:
    if dataset_name == "alpaca":
        dataset = load_dataset("vicgalle/alpaca-gpt4")
        lengths_instructions = [x for x in dataset["train"]["instruction"]]
        lengths_inputs = [x for x in dataset["train"]["input"]]
        lengths_outputs = [x for x in dataset["train"]["output"]]
        lengths = [x + y for x, y in zip(lengths_instructions, lengths_inputs)]
    elif dataset_name == "self-oss":
        dataset = load_dataset("bigcode/self-oss-instruct-sc2-exec-filter-50k")
        lengths = [x for x in dataset["train"]["prompt"]]
        lengths_outputs = [x for x in dataset["train"]["response"]]
    elif dataset_name == "orca":
        dataset = load_dataset("pankajmathur/WizardLM_Orca")
        lengths_systems = [x for x in dataset["train"]["system"]]
        lengths_instruction = [x for x in dataset["train"]["instruction"]]
        lengths_outputs = [x for x in dataset["train"]["output"]]
        lengths = [x + y for x, y in zip(lengths_systems, lengths_instruction)]
    else:
        raise ValueError("Invalid dataset name")

    return list(zip(lengths, lengths_outputs))


def find_current_cpu_core():
    return psutil.Process().cpu_num()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="alpaca")

    args = parser.parse_args()

    todays_date = datetime.date.today().strftime("%Y-%m-%d")
    num_gpus = torch.cuda.device_count()
    hf_name = args.hf_name
    model_name = hf_name.split("--")[-1]
    batch_size = args.batch_size
    out_dir = args.out_dir
    dataset = args.dataset
    csv_file = f"optimum-{model_name}-{num_gpus}.csv"

    prompts = get_prompts(dataset)

    pandas_handle = PandasHandler()
    if out_dir == ".":
        start_time = datetime.datetime.now().strftime("%H-%M-%S")
    else:
        start_time = out_dir.split("/")[-1]

    if "AMD" in subprocess.check_output("lscpu").decode():
        domains = [NvidiaGPUDomain(i) for i in range(num_gpus)]
    elif "Intel" in subprocess.check_output("lscpu").decode():
        domains = [RaplPackageDomain(0), RaplPackageDomain(1)]
        domains.extend([NvidiaGPUDomain(i) for i in range(num_gpus)])

    with open(f"{out_dir}/job_info.yaml", "w") as file:
        file.write("job:\n")
        file.write(f"  date: {todays_date}\n")
        file.write(f"  start_time: {start_time}\n")
        file.write("  details:\n")
        file.write(f"    model_name: {model_name}\n")
        file.write(f"    batch_size: {batch_size}\n")
        file.write(f"    hf_name: {hf_name}\n")
        file.write(f"    num_gpus: {num_gpus}\n")
        file.write(f"    dataset: {dataset}\n")

    with EnergyContext(
        handler=pandas_handle,
        domains=domains,
        start_tag="tokenizer",
    ) as ctx:
        tokenizer_core = find_current_cpu_core()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipeline_core = find_current_cpu_core()
        ctx.record(tag="model load")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, max_batch_size=batch_size
        )

    df = pandas_handle.get_dataframe()
    df["Number of Input Tokens"] = 0
    df["Iteration"] = 0
    df["Model Name"] = model_name
    df["Number of GPUs"] = num_gpus
    df["Prompt"] = "startup"
    df["Number of Output Tokens"] = 0
    df["Batch Size"] = batch_size
    df["CPU Core"] = [tokenizer_core, pipeline_core]
    df["Serving Strategy"] = "Optimum"
    df["Dataset"] = dataset
    for idx_gpus in range(num_gpus):
        df[f"Total Memory {idx_gpus}"] = nvidia_smi.getInstance().DeviceQuery(
            "memory.total"
        )["gpu"][idx_gpus]["fb_memory_usage"]["total"]
        df[f"Used Memory {idx_gpus}"] = nvidia_smi.getInstance().DeviceQuery(
            "memory.used"
        )["gpu"][idx_gpus]["fb_memory_usage"]["used"]
    df.to_csv(
        csv_file,
        mode="a",
        header=False,
        index=False,
    )

    for iteration, (input, output) in enumerate(prompts):
        pandas_handle = PandasHandler()
        with EnergyContext(
            handler=pandas_handle,
            domains=domains,
            start_tag=f"start-inference-{iteration}",
        ) as ctx:
            cpu_core = find_current_cpu_core()
            token_limit = len(tokenizer.encode(output))
            tokens = tokenizer(input, padding=True, return_tensors="pt")
            generated, lengths = model.generate(
                **tokens,
                top_k=50,
                top_p=0.95,
                max_new_tokens=token_limit,
            )
        input_tokens = tokenizer.encode(input)
        num_input_tokens = len(input_tokens)
        num_output_tokens = len(generated[0])
        df = pandas_handle.get_dataframe()
        df["Number of Input Tokens"] = num_input_tokens
        df["Iteration"] = iteration
        df["Model Name"] = model_name
        df["Number of GPUs"] = num_gpus
        df["Prompt"] = input[:10].strip()
        df["Number of Output Tokens"] = num_output_tokens
        df["Total Number of Tokens"] = num_output_tokens + num_input_tokens
        df["Batch Size"] = batch_size
        df["CPU Core"] = cpu_core
        df["Serving Strategy"] = "Optimum"
        df["Dataset"] = dataset
        for idx_gpus in range(num_gpus):
            df[f"Total Memory {idx_gpus}"] = nvidia_smi.getInstance().DeviceQuery(
                "memory.total"
            )["gpu"][idx_gpus]["fb_memory_usage"]["total"]
            df[f"Used Memory {idx_gpus}"] = nvidia_smi.getInstance().DeviceQuery(
                "memory.used"
            )["gpu"][idx_gpus]["fb_memory_usage"]["used"]
        df.to_csv(
            csv_file,
            mode="a",
            header=False,
            index=False,
        )
