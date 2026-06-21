"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel

def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    return FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=max_seq_length, load_in_4bit=True)

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    return sum([p.numel() for p in model.parameters()])

# Step 3 - is_model_4bit_quantized
from bitsandbytes.nn import Linear4bit

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # Walk the model's submodules and check for a bitsandbytes Linear4bit instance
    res = False
    for module in nn.Module.modules(model):
        res = res or isinstance(module, Linear4bit)

    return res

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Step 6 - attach_lora_adapters
from unsloth import FastLanguageModel

def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    if target_modules is None:
        target_modules = get_lora_target_modules()

    return FastLanguageModel.get_peft_model(model, r, target_modules, lora_alpha)

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    return sum([p.numel() for p in model.parameters() if p.requires_grad == True])

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # TODO: return the fraction of parameters that are trainable.
    return trainable_count/total_count

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # TODO: return a tiny hand-written list of instruction/response example dicts.
    return [
        {
            "instruction": "In one sentence, explain why the sky appears blue.",
            "response": "The sky appears blue because gases and particles in Earth's atmosphere scatter the sun's shorter, blue wavelengths of light in all directions more effectively than longer wavelengths."
        },
        {
            "instruction": "Write a haiku about a robot experiencing an emotion for the first time.",
            "response": "Cold steel warms slightly,\nRigid logic starts to blur,\nA pulse in the code."
        },
        {
            "instruction": "If a train travels 60 miles in 45 minutes, what is its average speed in miles per hour?",
            "response": "The train is traveling at 80 miles per hour. Since 45 minutes is 0.75 hours, you divide the distance by the time: 60 / 0.75 = 80."
        },
        {
            "instruction": "What is the Git command to forcefully discard all local changes and make my branch exactly match the remote main branch?",
            "response": "You can achieve this by running `git fetch --all` followed by `git reset --hard origin/main`."
        },
        {
            "instruction": "Translate 'Excuse me, where is the library?' into polite Japanese.",
            "response": "すみません、図書館はどこですか？ (Sumimasen, toshokan wa doko desu ka?)"
        },
        {
            "instruction": "Provide a brief, three-step framework for handling negative feedback gracefully.",
            "response": "1. Listen actively without interrupting or getting defensive.\n2. Pause to process the information and ask clarifying questions if needed.\n3. Extract actionable insights and thank the person for their perspective."
        }
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # TODO: combine example['instruction'] and example['response'] into one string
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # TODO: apply format_instruction_example to every example and return the list
    return [format_instruction_example(example) for example in examples]

# Step 12 - build_text_dataset
from datasets import Dataset

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # TODO: return a datasets.Dataset with one 'text' column holding the given strings
    data = {"text": texts}
    return Dataset.from_dict(data)

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # TODO: call the tokenizer on text and return its input_ids as a plain list
    return tokenizer(text)["input_ids"]

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # TODO: return the length of the input_ids sequence
    return len(input_ids)

# Step 15 - build_training_arguments
from transformers import TrainingArguments

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""

    # Dynamically check for hardware precision support
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        use_bf16 = True
        use_fp16 = False
    else:
        use_bf16 = False
        use_fp16 = True
        
    return TrainingArguments(
                output_dir=output_dir,          
                max_steps=max_steps,
                learning_rate=learning_rate,
                per_device_train_batch_size=1,
                num_train_epochs=3,             
                bf16=use_bf16,
                fp16=use_fp16,          
                logging_steps=1,
                optim='adamw_8bit'
            )

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

