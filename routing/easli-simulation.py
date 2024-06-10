from collections import deque
import random
import json
from math import floor
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, name, gpu_capacity, alpha_coeffs, beta_coeffs, accuracy):
        self.name = name
        self.queue = []
        self.gpu_capacity = gpu_capacity
        self.queue_history = []
        self.wait_times = []
        self.accuracy = accuracy
        self.historical_data = {
            "Energy": deque(),
            "Runtime": deque(),
            "Accuracy": deque(),
        }
        self.current_cost = 0
        self.alpha_coeffs = alpha_coeffs
        self.beta_coeffs = beta_coeffs
        self.energy_minmax = (0, self.energy_cost(2048, 2048))
        self.runtime_minmax = (0, self.runtime_cost(2048, 2048))
        self.accuracy_minmax = (0, self.accuracy_cost(2048, 2048))

    def add_query(self, query, t):
        t_in, t_out, arrival_time = query
        if len(self.queue) < self.gpu_capacity:
            projected_end_time = t + self.runtime_cost(t_in, t_out)
            self.queue.append((t_in, t_out, projected_end_time))
            self.queue_history.append((t_in, t_out, arrival_time))
            self.wait_times.append(0)
        else:
            self.queue.append((t_in, t_out, t))

    def energy_cost(self, t_in, t_out):
        coeffs = self.alpha_coeffs
        return (
            coeffs["alpha_0"] * t_in
            + coeffs["alpha_1"] * t_out
            + coeffs["alpha_2"] * t_in * t_out
        )

    def normalized_energy_cost(self, t_in, t_out):
        return self.normalize(self.energy_cost(t_in, t_out), self.energy_minmax)

    def runtime_cost(self, t_in, t_out):
        coeffs = self.beta_coeffs
        return (
            coeffs["beta_0"] * t_in
            + coeffs["beta_1"] * t_out
            + coeffs["beta_2"] * t_in * t_out
        )

    def normalized_accuracy_cost(self, t_in, t_out):
        return self.normalize(self.runtime_cost(t_in, t_out), self.runtime_minmax)

    def accuracy_cost(self, t_in, t_out):
        return self.accuracy * (t_in + t_out)

    def normalized_accuracy(self, t_in, t_out):
        return self.normalize(self.accuracy_cost(t_in, t_out), self.accuracy_minmax)

    def calculate_cost(self, t_in, t_out, zeta):
        return zeta * self.energy_cost(t_in, t_out) - (1 - zeta) * self.accuracy_cost(
            t_in, t_out
        )

    def update_costs(self, t_in, t_out, zeta):
        self.current_cost += self.calculate_cost(self.name, t_in, t_out, zeta)

    def update_historical_data(self, t_in, t_out):
        self.historical_data["Energy"].append(self.energy_cost(self.name, t_in, t_out))
        self.historical_data["Runtime"].append(
            self.runtime_cost(self.name, t_in, t_out)
        )
        self.historical_data["Accuracy"].append(
            self.accuracy_cost(self.name, t_in, t_out)
        )

        if len(self.historical_data["Energy"] > 1):
            self.energy_minmax = (
                min(self.historical_data["Energy"]),
                max(self.historical_data["Energy"]),
            )
            self.runtime_minmax = (
                min(self.historical_data["Runtime"]),
                max(self.historical_data["Runtime"]),
            )
            self.accuracy_minmax = (
                min(self.historical_data["Accuracy"]),
                max(self.historical_data["Accuracy"]),
            )

        if len(self.historical_data["Energy"]) > 100:
            self.historical_data["Energy"].popleft()
            self.historical_data["Runtime"].popleft()
            self.historical_data["Accuracy"].popleft()

    def normalize(self, value, minmax):
        min_value, max_value = minmax
        return (
            (value - min_value) / (max_value - min_value)
            if max_value != min_value
            else 0
        )

    def reset(self):
        self.queue = []
        self.queue_history = []
        self.wait_times = []
        self.historical_data = {
            "Energy": deque(),
            "Runtime": deque(),
            "Accuracy": deque(),
        }
        self.current_cost = 0
        self.energy_minmax = (0, self.energy_cost(2048, 2048))
        self.runtime_minmax = (0, self.runtime_cost(2048, 2048))
        self.accuracy_minmax = (0, self.accuracy_cost(2048, 2048))


class Simulation:
    def __init__(self, models, timespan):
        self.models = models
        self.time = 0
        self.timespan = timespan

    def handle_query_random(self, query):
        t_in, t_out, arrival_time = query
        selected_model = random.choice(self.models)
        selected_model.add_query(query, self.time)

    def handle_query_round_robin(self, query, current_index):
        t_in, t_out, arrival_time = query
        selected_model = self.models[current_index % len(self.models)]
        selected_model.add_query(query, self.time)

    def handle_query_online(self, query):
        t_in, t_out, arrival_time = query
        for K in self.models:
            cost_with_query = K.current_cost + K.calculate_cost(t_in, t_out, K.zeta)
            if cost_with_query < min_cost:
                min_cost = cost_with_query
                selected_model = K
        selected_model.add_query(query, self.time)
        selected_model.update_costs(t_in, t_out, selected_model.zeta)
        selected_model.update_historical_data(t_in, t_out)
        if len(selected_model.historical_data["Energy"]) % 10 == 0:
            for K in selected_model.models:
                selected_model.current_cost = 0

    def run_simulation(self, queries, strategy="random"):
        for query in queries:
            if strategy == "random":
                self.handle_query_random(query)
            elif strategy == "round_robin":
                current_index = 0
                self.handle_query_round_robin(query, current_index)
                current_index += 1
            elif strategy == "online":
                self.handle_query_online(query)
            self.time += 1  # Increment simulation time


def load_model_info(json_file_path: str) -> dict:
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data


def load_queries(T: float, dataset: str) -> deque:
    queries = deque()
    if dataset is "alpaca":
        dataset = load_dataset("vicgalle/alpaca-gpt4")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        lengths_instructions = [
            len(tokenizer.encode(x)) for x in dataset["train"]["instruction"]
        ]
        lengths_inputs = [len(tokenizer.encode(x)) for x in dataset["train"]["input"]]
        lengths_outputs = [len(tokenizer.encode(x)) for x in dataset["train"]["output"]]
        total_lengths_inputs = [
            x + y for x, y in zip(lengths_instructions, lengths_inputs)
        ]
        arrival_times = sorted(
            np.random.uniform(0, T, len(lengths_instructions))
        )  # Simulate arrivals over time
        queries = deque(zip(total_lengths_inputs, lengths_outputs, arrival_times))
    else:
        raise ValueError("Invalid dataset")
    return queries


def main():
    N = 16000  # Number of GPUs
    T = 10000  # [0, T] time span for simulation
    # Load model info from JSON file
    info = load_model_info("coefficients.json")
    models = []

    # Populate objects with model info
    for model_name in info["alpha_coeffs"].keys():
        alpha = info["alpha_coeffs"][model_name]
        beta = info["beta_coeffs"][model_name]
        accuracy = info["accuracy"][model_name]
        gamma_K = info["gamma_K"][model_name]
        min_a100s = info["min_a100s"][model_name]
        gpu_capacity = floor(N * gamma_K / min_a100s)
        model = Model(
            name=model_name,
            gpu_capacity=gpu_capacity,
            alpha_coeffs=alpha,
            beta_coeffs=beta,
            accuracy=accuracy,
        )
        models.append(model)
    simulation = Simulation(models=models, timespan=T)
    queries = load_queries(T=T, dataset="alpaca")
    lambda_val = len(queries) / T  # Arrival rate
    simulation.run_simulation(queries, strategy="random")


if __name__ == "__main__":
    main()
