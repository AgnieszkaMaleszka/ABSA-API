# 🧠 ABSA API – Aspect-Based Sentiment Analysis (PL 🇵🇱)

**ABSA API** to aplikacja webowa (Gradio), która analizuje komentarze restauracyjne w języku **polskim** i:
- ✨ wykrywa **aspekty** (np. pizza, obsługa, ceny)
- 📊 przypisuje im **sentyment** (pozytywny, negatywny, neutralny, konfliktowy)

---

## 🚀 Demo na Hugging Face Spaces

👉 [Uruchom aplikację tutaj](https://huggingface.co/spaces/EfektMotyla/absa-api)  
🔄 Działa 24/7 w przeglądarce – bez instalacji

---

## 🔧 Jak to działa?

### 📌 Wejście:
Polski komentarz, np.:

Pizza była pyszna, ale obsługa bardzo niemiła.

### 📌 Wyjście:
🧩 Pizza → pozytywny
🧩 Obsługa → negatywny

---

## 🛠️ Technologie

| Część | Opis |
|-------|------|
| `transformers` | Modele NLP (tokenizacja, klasyfikacja, tłumaczenie) |
| `Gradio` | Interfejs użytkownika (UI + API) |
| `Hugging Face Hub` | Hosting modeli i Spaces |
| `MarianMT` | Tłumaczenie PL ↔ EN |
| `PyTorch` | Backend modeli (CPU/GPU) |

---

## 🤖 Używane modele

| Model | Opis |
|-------|------|
| [`EfektMotyla/bert-aspect-ner`](https://huggingface.co/EfektMotyla/bert-aspect-ner) | Wykrywanie aspektów |
| [`EfektMotyla/absa-roberta`](https://huggingface.co/EfektMotyla/absa-roberta) | Klasyfikacja sentymentu |
| `Helsinki-NLP/opus-mt-pl-en` | Tłumaczenie z polskiego na angielski |
| `gsarti/opus-mt-tc-en-pl` | Tłumaczenie z angielskiego na polski |

---

## 🧪 Przykłady komentarzy do testu

- Obsługa była bardzo miła, ale na jedzenie czekaliśmy ponad godzinę.  
- Pizza była przepyszna, ale ceny stanowczo za wysokie.  
- Muzyka była zbyt głośna, ale lokalizacja świetna.  
- Zamówienie pomylono, ale kelner przeprosił i szybko naprawił sytuację.

---

## 📱 Integracja z aplikacją mobilną

Ten projekt jest wykorzystywany jako **API do analizy komentarzy** w mojej aplikacji mobilnej (Android Jetpack Compose).  
Aplikacja wysyła komentarze REST-owo do endpointu, a w odpowiedzi otrzymuje analizę sentymentu dla wykrytych aspektów.

### 🔁 Przykład zapytania (POST `/absa`)

```http
POST https://huggingface.co/spaces/EfektMotyla/absa-api
Content-Type: application/json

{
  "text": "Pizza była pyszna, ale kelner był nieuprzejmy."
}
```
### 🔁 Przykład odpowiedzi:
```json
[
  { "aspect": "pizza", "sentiment": "pozytywny" },
  { "aspect": "kelner", "sentiment": "negatywny" }
]
```

### 📦 Użycie w aplikacji:
W aplikacji mobilnej wykorzystuję to API np. za pomocą biblioteki Retrofit, która wysyła komentarze użytkowników do analizy.
Wynik (aspekt + sentyment) jest wyświetlany bezpośrednio w interfejsie.



