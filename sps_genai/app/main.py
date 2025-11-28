# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.model_llm import load_llm_model

app = FastAPI(title="RNN + LLM Text Generator API")

model, tokenizer, device = load_llm_model()

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = 50

@app.post("/generate_with_llm")
def generate_with_llm(request: TextGenerationRequest):
    prompt = request.start_word
    max_length = request.length

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}
