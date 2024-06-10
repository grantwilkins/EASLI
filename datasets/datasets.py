from datasets import load_dataset
from transformers import AutoTokenizer
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

dataset = load_dataset("vicgalle/alpaca-gpt4")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
lengths_instructions = [
    len(tokenizer.encode(x)) for x in dataset["train"]["instruction"]
]
lengths_temp = [len(tokenizer.encode(x)) for x in dataset["train"]["input"]]
lengths_outputs = [len(tokenizer.encode(x)) for x in dataset["train"]["output"]]
lengths_inputs = [x + y for x, y in zip(lengths_instructions, lengths_temp)]

dataset_2 = load_dataset("ArtifactAI/arxiv-math-instruct-50k")
lengths_inputs_2 = [x for x in dataset_2["train"]["question"]]
lengths_outputs_2 = [x for x in dataset_2["train"]["answer"]]

dataset_3 = load_dataset("pankajmathur/WizardLM_Orca")
lengths_systems = [x for x in dataset_3["train"]["system"]]
lengths_instruction = [x for x in dataset_3["train"]["instruction"]]
lengths_outputs_3 = [x for x in dataset_3["train"]["output"]]
lengths_inputs_3 = [x + y for x, y in zip(lengths_systems, lengths_instruction)]


plt.figure(figsize=(6, 5))
sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")
sns.kdeplot(lengths_inputs, label="Alpaca", alpha=0.5, fill=True)
sns.kdeplot(lengths_inputs_2, label="arXiv Math ", alpha=0.5, fill=True)
sns.kdeplot(lengths_inputs_3, label="WizardLM Orca", alpha=0.5, fill=True)
# plt.xlim(0, 250)
plt.xlabel("Number of Input Tokens")
plt.ylabel("Density")
plt.legend(
    bbox_to_anchor=(0.5, -0.25),
    loc="center",
    ncol=3,
    frameon=False,
)
# plt.show()
plt.savefig("input-tokens-kde.pdf", bbox_inches="tight")

plt.figure(figsize=(6, 5))
sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")
sns.kdeplot(lengths_outputs, label="Alpaca", alpha=0.5, fill=True)
sns.kdeplot(lengths_outputs_2, label="GSM8K", alpha=0.5, fill=True)
sns.kdeplot(lengths_outputs_3, label="Python Codes", alpha=0.5, fill=True)
# plt.xlim(0, 700)
plt.xlabel("Number of Output Tokens")
plt.ylabel("Density")
plt.legend(
    bbox_to_anchor=(0.5, -0.25),
    loc="center",
    ncol=3,
    frameon=False,
)
# plt.show()
plt.savefig("output-tokens-kde.pdf", bbox_inches="tight")
