# Assignment 5 — Language Models, Fine-Tuning & Reinforcement Learning

**Course:** Applied Generative AI  
**Student:** Miguel Morales (mam2670)  
**Professor:** Gurgen Hayrapetyan  
**Date:** Fall 2025  

This repository contains all work completed for **Assignment 5**, which builds on the material from **Activity 9**.  
The assignment includes:

- A full implementation of **character-level language modeling**  
- A **custom RNN model**
- A **Bigram Language Model**
- A **GPT-2 fine-tuning pipeline using HuggingFace Transformers**
- A custom **Reward Model + RL post-training loop** (Prefix-rewarded model)
- A clean, reproducible project structure built using **uv** (Python environment manager)

---

## Repository Structure

Assignment5/
│
├── sps_genai/
│ ├── app/
│ │ ├── bigram_model.py
│ │ ├── model_rnn.py
│ │ ├── model_llm.py
│ │ ├── finetune_llm.py
│ │ ├── rl_posttrain_llm.py
│ │ ├── main.py → CLI to test model outputs
│ │ └── finetuned_gpt2/ → (small metadata only; heavy files ignored)
│ │
│ ├── helper_lib/ → Utilities for model training
│ ├── data/ → MNIST dataset (small files only)
│ ├── README.md → Internal module documentation
│ ├── pyproject.toml → uv environment config
│ └── uv.lock
│
├── .gitignore
└── Theory Section #2 Assignment 5 Miguel Morales.pdf


All large files (checkpoints >100MB) are ignored to keep the repository light and GitHub-friendly.

---

##  **How to Run This Project**

### **1. Install uv**
If not installed:

curl -LsSf https://astral.sh/uv/install.sh | sh

2. Enter the project directory
  cd sps_genai

3. Install environment & dependencies
  uv sync
  uv pip install -e .

4. Run the Bigram Language Model
  uv run python app/bigram_model.py

5. Run the RNN Character Model
  uv run python app/model_rnn.py

6. Fine-Tune GPT-2
  (Requires GPU if training for long)
  uv run python app/finetune_llm.py

7. Run RL Post-Training (Prefix-Reward)
  uv run python app/rl_posttrain_llm.py

8. Test model generation
  uv run python app/main.py

Assignment Highlights
    ✔ Bigram LM
        Implemented from scratch including:
        Vocabulary extraction
        Token–token transition matrix
        Sampling
        Loss computation
    ✔ Custom RNN
      Includes:
        Manual forward pass
        Loss function
        Mini-batch training
        Temperature sampling for text generation
    ✔ GPT-2 Fine-Tuning
      Using HuggingFace:
        LoRA/PEFT structure
        Trainer API
        TrainingArguments
        Save/Load pipeline
    ✔ RL Reward Model
   
  Implements a prefix reward strategy:
  
  def compute_reward(text: str) -> float:
    reward = 0.0
    if text.startswith("PREFIX_"):
        reward += 70
    return reward

Then a PPO-style loop trains GPT-2 to increase likelihood of rewarded outputs.
  Included Deliverables
    Theory Section PDF
    All Python code required for character LM, LLM fine-tuning, and RL post-training
    Clean Git history
    Fully reproducible environment
    All heavy checkpoints were intentionally excluded following best GitHub practices.
  Notes & Important Info
    model.safetensors, optimizer states, and MNIST raw files are automatically ignored.
    This repository contains only lightweight metadata, enough for inspection.
    If the professor requires full checkpoints, they can be provided separately via Google Drive.
  Contact

Miguel Morales

    
