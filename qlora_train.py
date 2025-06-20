from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
from datasets import load_dataset
import matplotlib.pyplot as plt
import math
import torch

# ==== Step 1: Load tokenizer and model in 4-bit ====
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# ==== Step 2: QLoRA config ====
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# ==== Step 3: Load and prepare dataset ====
dataset = load_dataset("json", data_files="data/malaysian_english.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

def combine_prompt(example):
    return {
        "text": f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['response']}"
    }

train_dataset = train_dataset.map(combine_prompt)
eval_dataset = eval_dataset.map(combine_prompt)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

# ==== Step 4: TrainingArguments ====
training_args = TrainingArguments(
    output_dir="./qlora_output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="no", 
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==== Step 5: Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ==== Step 6: Train all epochs ====
trainer.train()

# ==== Step 7: Evaluate only after full training ====
results = trainer.evaluate(eval_dataset=eval_dataset)
print("Evaluation results:", results)
print("Perplexity:", math.exp(results["eval_loss"]))

# ==== Step 8: Plot training loss only (eval is single point) ====
def plot_metrics(trainer):
    log_history = trainer.state.log_history
    train_epochs, train_loss = [], []

    for log in log_history:
        if "loss" in log and "epoch" in log:
            train_epochs.append(log["epoch"])
            train_loss.append(log["loss"])

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_loss, label="Training Loss", marker='o')
    plt.axhline(y=results["eval_loss"], color='r', linestyle='--', label="Final Eval Loss")
    plt.title("Training Loss per Epoch (Eval after Final Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss_plot.png")
    plt.show()

plot_metrics(trainer)

# ==== Step 9: Save model and tokenizer ====
model.save_pretrained("./qlora_finetuned_model")
tokenizer.save_pretrained("./qlora_finetuned_model")
torch.cuda.empty_cache()