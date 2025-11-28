# app/rl_posttrain_llm.py
import os
import random
import torch
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# Frases de formato que queremos forzar con RL
PREFIX = "That is a great question."
SUFFIX = "Let me know if you have any other questions."

BASE_DIR = os.path.dirname(__file__)
START_MODEL_DIR = os.path.join(BASE_DIR, "finetuned_gpt2")      # modelo de Activity 9
OUTPUT_DIR = os.path.join(BASE_DIR, "finetuned_gpt2_rl")       # modelo con RL

# Algunas preguntas de ejemplo para generar respuestas durante RL
QUESTIONS = [
    "What is AI?",
    "How does reinforcement learning work?",
    "What is a neural network?",
    "What is GPT-2?",
    "Why is data important in machine learning?",
]


def compute_reward(answer: str) -> float:
    """
    Función de recompensa:
    - Premia si empieza con PREFIX
    - Premia si termina con SUFFIX
    - Bonus si la longitud está en un rango razonable
    """
    text = answer.strip()
    reward = 0.0

    # Prefijo correcto
    if text.startswith(PREFIX):
        reward += 70.0
    else:
        reward -= 10.0

    # Sufijo correcto
    if text.endswith(SUFFIX):
        reward += 50.0
    else:
        reward -= 10.0

    # Longitud (en palabras)
    n_tokens = len(text.split())
    if 25 <= n_tokens <= 80:
        reward += 10.0
    else:
        reward -= 5.0

    return reward


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) cargar modelo YA fine-tuned (o base si no existe)
    model_path = START_MODEL_DIR if os.path.isdir(START_MODEL_DIR) else "openai-community/gpt2"
    print(f"Loading starting model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.config.pad_token_id = tokenizer.eos_token_id

    optimizer = Adam(model.parameters(), lr=1e-5)

    num_episodes = 50          # puedes subir/bajar para experimentar
    max_new_tokens = 80

    model.train()

    for ep in range(num_episodes):
        # 1) escoger una pregunta "de usuario"
        question = random.choice(QUESTIONS)

        # 2) construir prompt con prefijo fijo
        prompt = f"Q: {question}\nA: {PREFIX} "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 3) generar respuesta (exploración) SIN gradientes
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )

        # decodificar texto entero (prompt + respuesta)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 4) calcular recompensa
        R = compute_reward(full_text)

        # 5) REINFORCE sencillo: volver a calcular log-probs con gradientes
        #    usando "teacher forcing" sobre la secuencia generada
        input_ids = outputs[0][:-1].unsqueeze(0)   # todos menos el último
        target_ids = outputs[0][1:].unsqueeze(0)   # desplazado 1 a la derecha

        logits = model(input_ids).logits                         # [1, T-1, vocab]
        log_probs = F.log_softmax(logits, dim=-1)                # mismo shape

        # log_probs escogidos (para el token real en cada paso)
        chosen_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        # pérdida = -R * media de log-probs
        loss = - R * chosen_log_probs.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Episode {ep+1}/{num_episodes}] Reward={R:.2f}  Loss={loss.item():.4f}")

    # 6) guardar modelo post-entrenado en app/finetuned_gpt2_rl
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ RL post-training completado. Modelo guardado en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
