import gradio as gr
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    MarianMTModel, MarianTokenizer
)
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Tokenizery i modele ABSA ===
aspect_tokenizer = AutoTokenizer.from_pretrained("EfektMotyla/bert-aspect-ner")
aspect_model = AutoModelForTokenClassification.from_pretrained("EfektMotyla/bert-aspect-ner").to(device)

sentiment_tokenizer = AutoTokenizer.from_pretrained("EfektMotyla/absa-roberta")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("EfektMotyla/absa-roberta").to(device)

en_to_pl_tokenizer = MarianTokenizer.from_pretrained("gsarti/opus-mt-tc-en-pl")
en_to_pl_model = MarianMTModel.from_pretrained("gsarti/opus-mt-tc-en-pl").to(device)

pl_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
pl_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-pl-en").to(device)

def translate(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

def translate_pl_to_en(texts): return translate(texts, pl_to_en_tokenizer, pl_to_en_model)
def translate_en_to_pl(texts): return translate(texts, en_to_pl_tokenizer, en_to_pl_model)

def extract_aspects(text):
    inputs = aspect_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = aspect_model(**inputs)
    preds = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    tokens = aspect_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [aspect_model.config.id2label[p] for p in preds]

    aspects = []
    current_tokens = []
    for token, label in zip(tokens, labels):
        if label == "B-ASP":
            if current_tokens:
                aspects.append(aspect_tokenizer.convert_tokens_to_string(current_tokens).strip())
                current_tokens = []
            current_tokens = [token]
        elif label == "I-ASP" and current_tokens:
            current_tokens.append(token)
        else:
            if current_tokens:
                aspects.append(aspect_tokenizer.convert_tokens_to_string(current_tokens).strip())
                current_tokens = []
    if current_tokens:
        aspects.append(aspect_tokenizer.convert_tokens_to_string(current_tokens).strip())
    return list(set(aspects))

def analyze(text_pl):
    try:
        text_en = translate_pl_to_en([text_pl])[0]
        aspects_en = extract_aspects(text_en)
        if not aspects_en:
            return "Nie wykryto żadnych aspektów."
        
        results = []
        for asp in aspects_en:
            input_text = f"{text_en} [SEP] {asp}"
            inputs = sentiment_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                logits = sentiment_model(**inputs).logits
                predicted_class_id = int(logits.argmax().cpu())
                sentiment_label = {0: "negatywny", 1: "neutralny", 2: "pozytywny", 3: "konfliktowy"}[predicted_class_id]
                asp_pl = translate_en_to_pl([asp])[0]
                results.append(f"🧩 {asp_pl.capitalize()} → **{sentiment_label}**")
        return "\n".join(results)
    except Exception as e:
        return f"Błąd: {str(e)}"

# === Gradio UI ===
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Komentarz po polsku", placeholder="Np. Pizza była pyszna, ale kelner był nieuprzejmy."),
    outputs=gr.Markdown(label="Wyniki analizy"),
    title="ABSA – Analiza komentarzy restauracyjnych",
    description="Wykrywa aspekty i przypisuje im sentymenty (pozytywny / negatywny / neutralny / konfliktowy)."
)

demo.launch()
