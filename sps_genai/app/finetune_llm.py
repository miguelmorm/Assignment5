# finetune_llm.py
import os
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch

# 🔹 Modelo base de HuggingFace
BASE_MODEL = "openai-community/gpt2"

# 🔹 Carpeta donde SE GUARDA el modelo fine-tuned
#     => coincide con FINETUNED_DIR en app/model_llm.py
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "finetuned_gpt2")

# 🔹 Si ya tienes CSVs reales, pon aquí los paths correctos
DATA_FILES = {
    "train": "data/nectar_qa_train.csv",
    "validation": "data/nectar_qa_valid.csv",
}
QUESTION_COL = "question"
ANSWER_COL = "answer"


def load_or_create_dataset():
    """
    1) Si existen los CSV en data/, los usa.
    2) Si NO existen, crea un mini dataset sintético de ejemplo
       (para que la actividad funcione sí o sí).
    """
    if all(os.path.exists(path) for path in DATA_FILES.values()):
        print("✅ Usando dataset CSV real...")
        raw_datasets = load_dataset("csv", data_files=DATA_FILES)
    else:
        print("⚠️ CSVs no encontrados, creando dataset QA sintético para la actividad...")
        examples = [
            {
                QUESTION_COL: "What is AI?",
                ANSWER_COL: "AI is the field of building machines that can perform tasks that normally require human intelligence.",
            },
            {
                QUESTION_COL: "What is a neural network?",
                ANSWER_COL: "A neural network is a function composed of layers of simple units that learn patterns from data.",
            },
            {
                QUESTION_COL: "What is GPT2?",
                ANSWER_COL: "GPT2 is a transformer-based language model that can generate text.",
            },
        ]
        raw_datasets = Dataset.from_list(examples)
        raw_datasets = {"train": raw_datasets, "validation": raw_datasets}
    return raw_datasets


def main():
    # 1. Cargar dataset (real o sintético)
    raw_datasets = load_or_create_dataset()

    # 2. Cargar tokenizer y modelo base
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # 3. Preprocesado: concatenar pregunta + respuesta
    def format_example(example):
        text = (
            "Question: " + example[QUESTION_COL].strip()
            + "\nAnswer: " + example[ANSWER_COL].strip()
        )
        return {"text": text}

    if isinstance(raw_datasets, dict):
        # caso de CSV (dataset dict con train/validation)
        formatted_datasets = {
            split: ds.map(format_example)
            for split, ds in raw_datasets.items()
        }
    else:
        # caso Dataset simple
        formatted_datasets = {
            "train": raw_datasets.map(format_example),
            "validation": raw_datasets.map(format_example),
        }

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=256,
        )

    tokenized_datasets = {
        split: ds.map(
            tokenize_function,
            batched=True,
            remove_columns=ds.column_names,
        )
        for split, ds in formatted_datasets.items()
    }

    # 4. Data collator para causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, no masked LM
    )

    # 5. Configurar entrenamiento (puedes ajustar epochs/batch_size)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        # si tu versión no soporta esto, lo puedes quitar:
        # per_device_eval_batch_size=2,
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=50,
        save_steps=500,       # guarda cada N pasos
        save_total_limit=1,   # solo conserva el último checkpoint
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 6. Entrenar
    trainer.train()

    # 7. Guardar modelo + tokenizer EXACTAMENTE en app/finetuned_gpt2
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✅ Fine-tuning terminado. Modelo guardado en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
