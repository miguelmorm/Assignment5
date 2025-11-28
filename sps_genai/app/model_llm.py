# app/model_llm.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

BASE_MODEL_NAME = "openai-community/gpt2"
BASE_DIR = os.path.dirname(__file__)

# Modelo fine-tuned "normal" (de Activity 9)
FINETUNED_DIR = os.path.join(BASE_DIR, "finetuned_gpt2")

# Modelo post-entrenado con RL (Assignment 5)
FINETUNED_RL_DIR = os.path.join(BASE_DIR, "finetuned_gpt2_rl")


def load_llm_model():
    """
    Carga el modelo en este orden de prioridad:
    1) Modelo RL (finetuned_gpt2_rl)
    2) Modelo fine-tuned normal (finetuned_gpt2)
    3) Modelo base GPT-2 de HuggingFace
    """
    if os.path.isdir(FINETUNED_RL_DIR):
        model_name = FINETUNED_RL_DIR
        print(f"Loading RL post-trained model from {FINETUNED_RL_DIR}")
    elif os.path.isdir(FINETUNED_DIR):
        model_name = FINETUNED_DIR
        print(f"Loading fine-tuned model from {FINETUNED_DIR}")
    else:
        model_name = BASE_MODEL_NAME
        print(f"Fine-tuned models not found, loading base model: {BASE_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # asegurar pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, device
