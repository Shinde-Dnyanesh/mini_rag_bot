from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

MODEL_NAME = "google/flan-t5-small"
_GEN = None

def get_generator():
    global _GEN
    if _GEN is None:
        print("Loading generator model (may take a minute)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _GEN = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return _GEN

def build_prompt(question: str, contexts: list) -> str:
    ctx = "\n\n".join([f"[{c['source']}] {c['text']}" for c in contexts])
    prompt = f"Answer the question based on the context below.\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    return prompt

def generate(question: str, contexts: list) -> str:
    gen = get_generator()
    prompt = build_prompt(question, contexts)
    out = gen(prompt, max_new_tokens=120, do_sample=False)
    return out[0]["generated_text"].strip()
