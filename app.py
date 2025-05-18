import gradio as gr
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Tokenizery i modele ABSA ===
aspect_tokenizer = AutoTokenizer.from_pretrained("EfektMotyla/bert-aspect-ner")
aspect_model = AutoModelForTokenClassification.from_pretrained("EfektMotyla/bert-aspect-ner").to(device)

sentiment_tokenizer = AutoTokenizer.from_pretrained("EfektMotyla/absa-roberta")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("EfektMotyla/absa-roberta").to(device)

pl_to_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-pl-en", device=0 if torch.cuda.is_available() else -1)
en_to_pl_translator = pipeline("translation", model="gsarti/opus-mt-tc-en-pl", device=0 if torch.cuda.is_available() else -1)

def translate_pl_to_en(texts):
    return [res["translation_text"] for res in pl_to_en_translator(texts)]

def translate_en_to_pl(texts):
    return [res["translation_text"] for res in en_to_pl_translator(texts)]

# === Słownik znanych aspektów (EN → PL) ===
aspect_aliases = {
    # JEDZENIE / SMAK
    "food": "jedzenie",
    "meal": "jedzenie",
    "taste": "smak",
    "flavor": "smak",
    "dish": "danie",
    "portion": "porcja",
    "serving": "porcja",
    "ingredients": "składniki",
    "spices": "przyprawy",
    "salt": "sól",
    "fat": "tłuszcz",
    "grease": "tłuszcz",

    # OBSŁUGA
    "service": "obsługa",
    "staff": "obsługa",
    "waiter": "obsługa",
    "waitress": "obsługa",
    "manager": "obsługa",
    "attitude": "obsługa",

    # CENY / WARTOŚĆ
    "price": "cena",
    "value": "cena",
    "cost": "cena",

    # ATMOSFERA / WYSTRÓJ
    "decor": "wystrój",
    "interior": "wystrój",
    "design": "wystrój",
    "counter": "wystrój",
    "fridge": "wystrój",
    "music": "muzyka",
    "ambience": "klimat",
    "atmosphere": "klimat",
    "vibe": "klimat",
    "climate": "klimat",

    # MIEJSCE
    "location": "lokalizacja",
    "place": "lokalizacja",
    "entrance": "lokalizacja",
    "parking": "parking",
    "toilet": "toaleta",

    # CZAS / SZYBKOŚĆ
    "waiting time": "czas oczekiwania",
    "time": "czas oczekiwania",
    "delay": "opóźnienie",
    "speed": "czas oczekiwania",
    "service time": "czas oczekiwania",
    "slow": "czas oczekiwania",
    "fast": "czas oczekiwania",
    "immediate": "czas oczekiwania",
    "late": "opóźnienie",


    # ZAPACH / CZYSTOŚĆ
    "smell": "zapach",
    "odor": "zapach",
    "cleanliness": "czystość",
    "hygiene": "czystość",

    # OGÓLNE
    "experience": "doświadczenie",
    "visit": "wizyta",
    "menu": "menu",
    "variety": "menu",

    # MIEJSCE / LOKALIZACJA / OTOCZENIE
    "location": "lokalizacja",
    "place": "lokalizacja",
    "entrance": "lokalizacja",
    "parking": "parking",
    "view": "lokalizacja",
    "lake": "lokalizacja",
    "window": "lokalizacja",
    "terrace": "lokalizacja",
    "balcony": "lokalizacja",
    "outside": "lokalizacja",
    "area": "lokalizacja",
    "surroundings": "lokalizacja",
    "neighborhood": "lokalizacja",
    "river": "lokalizacja",
    "garden": "lokalizacja",

    # NAPOJE
    "drink": "napoje",
    "drinks": "napoje",
    "beverage": "napoje",
    "coffee": "napoje",
    "tea": "napoje",
    "water": "napoje",
    "juice": "napoje",
    "alcohol": "napoje",
    "cocktail": "napoje",
    "wine": "napoje",

    #HIGIENA
    "dirt": "czystość",
    "dirty": "czystość",
    "mess": "czystość",
    "messy": "czystość",
    "clean": "czystość",
    "filth": "czystość",

    #KUCHNIA /JAKOŚĆ 
    "chef": "kuchnia",
    "kitchen": "kuchnia",
    "preparation": "kuchnia",
    "presentation": "prezentacja",
    "quality": "jakość",
    "freshness": "jakość",
    "raw": "jakość",
    "undercooked": "jakość",
    "burnt": "jakość",
    "microwaved": "jakość",
    # Wyposażenie 
    "seat": "komfort",
    "seating": "komfort",
    "chair": "komfort",
    "table": "komfort",
    "furniture": "komfort",
    "light": "komfort",
    "noise": "komfort",
    "temperature": "komfort",
    "air conditioning": "komfort",

    # OGÓLNE WRAŻENIE / WARTOŚĆ
    "recommendation": "ogólna ocena",
    "return": "ogólna ocena",
    "again": "ogólna ocena",
    "worth": "cena",
    "overpriced": "cena",
    "cheap": "cena",
    "affordable": "cena",

    # DZIECI / RODZINA
    "child": "dzieci",
    "children": "dzieci",
    "kid": "dzieci",
    "kids": "dzieci",
    "child-friendly": "dzieci",
    "kids menu": "dzieci",
    "high chair": "dzieci",
    "stroller": "dzieci",
    "family": "rodzina",
    "families": "rodzina",
    "parent": "rodzina",
    "parents": "rodzina",
    "group": "rodzina",
    "big group": "rodzina",
    "baby": "dzieci",

    # ZWIERZĘTA
    "dog": "zwierzęta",
    "dogs": "zwierzęta",
    "pet": "zwierzęta",
    "pets": "zwierzęta",
    "pet-friendly": "zwierzęta",
    "dog-friendly": "zwierzęta",
    "animal": "zwierzęta",
}

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
    return list(set(aspects))  # usuń duplikaty

def analyze(text_pl, progress=gr.Progress()):
    try:
        progress(0, desc="Tłumaczenie na angielski...")
        text_en = translate_pl_to_en([text_pl])[0]

        progress(0.3, desc="Wykrywanie aspektów...")
        aspects_en = extract_aspects(text_en)
        if not aspects_en:
            return "Nie wykryto żadnych aspektów."

        unique_aspects = sorted(set([asp.lower() for asp in aspects_en]))
        results = []
        seen_pl_aspects = set()

        for i, asp in enumerate(unique_aspects):
            progress(0.4 + i/len(unique_aspects)*0.6, desc=f"Analiza aspektu: {asp}")
            input_text = f"{text_en} [SEP] {asp}"
            inputs = sentiment_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

            with torch.no_grad():
                logits = sentiment_model(**inputs).logits
                predicted_class_id = int(logits.argmax().cpu())
                sentiment_label = {0: "negatywny", 1: "neutralny", 2: "pozytywny", 3: "konfliktowy"}[predicted_class_id]

            # Tłumaczenie aspektu przez słownik lub model
            if asp in aspect_aliases:
                asp_pl = aspect_aliases[asp]
            else:
                asp_pl = translate_en_to_pl([asp])[0].lower()

            if asp_pl not in seen_pl_aspects:
                seen_pl_aspects.add(asp_pl)
                results.append(f"{asp_pl.capitalize()} → **{sentiment_label}**")

        return "\n".join(results)

    except Exception as e:
        return f"Błąd podczas analizy: {e}"

# === Gradio UI ===
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        label="Komentarz po polsku",
        placeholder="Np. Pizza była pyszna, ale kelner był nieuprzejmy.",
        lines=4,
        max_lines=6
    ),
    outputs=gr.Markdown(label="Wyniki analizy"),
    title="ABSA – Analiza komentarzy restauracyjnych",
    description="Wykrywa aspekty i przypisuje im sentymenty (pozytywny / negatywny / neutralny / konfliktowy).",
    theme="default",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
