# 🐾 UpTail v2

**AI platforma za personaliziran trening psov**
Brez zunanjega API ključa — vse deluje lokalno z ML modelom.

> Projekt za predmet UUI – Uvod v umetno inteligenco  
> Avtorja: Inja Vozelj & Aljaž Savšek | Mentor: Dr. Uroš Ocepek

---

## 🚀 Zagon v 3 korakih

```bash
# 1. Namesti odvisnosti
pip install -r requirements.txt

# 2. Natrenira model (samo prvič, ~30–60 sek)
python train_model.py

# 3. Zaženi aplikacijo
python app.py
```

Odpri brskalnik: **http://localhost:5001**

---

## 📁 Struktura projekta

```
uptail-v2/
├── app.py                      ← Flask strežnik (vse API točke)
├── train_model.py              ← Enkratni trening ML modela
├── generate_data.py            ← Generiranje Excel datasetov
├── requirements.txt            ← Python odvisnosti
├── Procfile                    ← Konfiguracija za deploy (Railway/Render)
│
├── data/
│   ├── dog_breeds_dataset.xlsx ← 537 pasem psov z atributi
│   ├── dog_tricks_dataset.xlsx ← 45 trikov + metode treninga
│   └── dawgtraindataset.xlsx   ← 500 resničnih treninških sej
│
├── model/                      ← Generirano po train_model.py
│   ├── success_model.joblib    ← GradientBoosting regressor
│   ├── method_model.joblib     ← RandomForest klasifikator
│   ├── breeds_lookup.json      ← Hitri lookup za 537 pasem
│   ├── tricks_lookup.json      ← Lookup za trike + metode
│   └── meta.json               ← MAE, natančnost, feature lista
│
└── templates/
    └── index.html              ← Frontend (black & white, Fraunces + DM Sans)
```

---

## 🧠 ML Model — Model Card

### Namen modela
Model napoveduje verjetnost uspešnega učenja posameznega trika glede na profil psa in lastnika ter klasificira optimalno metodo treninga.

### Podatki

| Vir | Opis | Število vnosov |
|-----|------|----------------|
| `dog_breeds_dataset.xlsx` | 537 pasem z atributi (trenabilnost, inteligenca, energija, trmavost, občutljivost...) | 537 |
| `dog_tricks_dataset.xlsx` | 45 trikov s težavnostjo, predpogoji, metodami in navodili | 45 trikov, 7 metod |
| `dawgtraindataset.xlsx` | Resnične seje (pasma × trik × motivacija × ocena) | 500 sej |
| Sintetični podatki | Generirani z domensko formulo (Coren lestvica + AKC) | 72,495 vzorcev |
| **Skupaj po tehtanju** | Realni podatki ×10, skupaj z sintetičnimi | **~77,495 vzorcev** |

### Atributi modela (feature vector, 14 dimenzij)

| # | Atribut | Obseg | Pomen |
|---|---------|-------|-------|
| 1 | `trainability` | 1–5 | Trenabilnost pasme (Very Low→Very High) |
| 2 | `intelligence` | 1–5 | Inteligenca pasme (Coren lestvica) |
| 3 | `stubbornness` | 1–10 | Trmavost pasme |
| 4 | `sensitivity` | 1–10 | Občutljivost pasme |
| 5 | `energy` | 1–5 | Energija pasme |
| 6 | `trick_diff` | 1–3 | Težavnost trika (Začetnik/Vmesni/Napredni) |
| 7 | `trick_min_train` | 1–5 | Min. trenabilnost za trik |
| 8 | `trick_min_intel` | 1–5 | Min. inteligenca za trik |
| 9 | `age_months` | 3–240 | Starost psa v mesecih |
| 10 | `owner_exp` | 1–3 | Izkušnje lastnika |
| 11 | `food_motivated` | 0/1 | Motivacija s hrano |
| 12 | `play_motivated` | 0/1 | Motivacija z igro |
| 13 | `praise_motivated` | 0/1 | Motivacija s pohvalo |
| 14 | `clicker_used` | 0/1 | Uporaba clickerja |

### Modeli

| Model | Tip | Ciljna spremenljivka | Metrika | Rezultat |
|-------|-----|---------------------|---------|----------|
| `success_model` | GradientBoostingRegressor | `success_probability` (0–1) | MAE | **0.062** |
| `method_model` | RandomForestClassifier | `method_preference` (0–3) | Accuracy | **98.7%** |

### Formula za verjetnost uspešnosti (domensko znanje)

```python
aptitude = trainability×0.40 + intelligence×0.30 + (10−stubbornness)/10×5×0.15
         + sensitivity/10×5×0.05 + energy×0.10

diff_penalty = {Beginner: 0.0, Intermediate: −0.6, Advanced: −1.6}

age_modifier = (+0.35 mladič 4–14 mes) | (+0.15 mlad 15–30 mes) | (−0.3 starejši 73+)

motivation  = food×0.20 + play×0.12 + praise×0.08
clicker_mod = +0.25 (če clicker)
exp_mod     = (owner_exp − 1) × 0.12

success_prob = sigmoid(aptitude + diff_penalty + age_mod + motivation + clicker_mod + exp_mod)
```

*Vir: Stanley Coren "Intelligence of Dogs", AKC trenabilnost, APDT pozitivno ojačanje*

### Najpomembnejši atributi

```
trainability         ██████████████ 35.1%
trick_diff           █████████████ 34.0%
intelligence         ███            7.9%
age_months           ███            7.6%
trick_min_intel      █              4.3%
```

### Omejitve modela
- Natreniran na sintetičnih podatkih → predpostavke formule neposredno vplivajo na napovedi
- Ne upošteva individualnih razlik znotraj pasme (dva Labradorja sta lahko zelo različna)
- Ne zna upoštevati poškodb ali zdravstvenih omejitev
- Za zelo redke pasme (<0.1% lastnikov) so napovedi manj zanesljive

---

## 🔌 API Dokumentacija

### `GET /api/breeds?q=<iskanje>`
Vrne seznam pasem. Brez parametra vrne vseh 537.

```json
{ "breeds": ["Labrador Retriever", "Labradoodle", ...] }
```

---

### `GET /api/breed/<ime_pasme>`
Vrne atribute pasme.

```json
{
  "breed_name": "Labrador Retriever",
  "trainability_score": 5,
  "intelligence_score": 5,
  "stubbornness": 2,
  "sensitivity": 7,
  "energy_score": 4,
  "energy_level": "High",
  "group": "Sporting",
  "good_kids": 9,
  "good_dogs": 8,
  "min_exercise": 75
}
```

---

### `POST /api/recommend`
Vrne personalizirana priporočila trikov (privzeto top 12).

**Zahteva:**
```json
{
  "breed": "Labrador Retriever",
  "age_months": 24,
  "owner_experience": 2,
  "food_motivated": true,
  "play_motivated": true,
  "praise_motivated": false,
  "clicker_used": false,
  "top_n": 12
}
```

**Odgovor:**
```json
{
  "tricks": [
    {
      "trick_name": "Sit",
      "category": "Obedience",
      "difficulty": "Beginner",
      "success_probability": 0.912,
      "expected_sessions": 1,
      "ease_label": "Zelo enostavno",
      "prerequisites": null,
      "best_method": "Lure Method",
      "method_steps": "1. Drži priboljšek nad nosom...",
      "method_best_for": "All breeds, beginners"
    }
  ],
  "breed_info": { ... },
  "model_mae": 0.062
}
```

**Oznake težavnosti:**

| `ease_label` | `success_probability` |
|---|---|
| Zelo enostavno | ≥ 80% |
| Enostavno | 65–79% |
| Srednje | 45–64% |
| Zahtevno | 28–44% |
| Zelo zahtevno | < 28% |

---

### `POST /api/training-plan`
Vrne 4-tedenski načrt treninga.

**Zahteva:**
```json
{ "dog_profile": { <enako kot /api/recommend> } }
```

**Odgovor:**
```json
{
  "plan": {
    "week_1": {
      "focus": "Osnove in zaupanje",
      "sessions_per_day": 2,
      "session_length_min": 5,
      "tricks": [ ... ]
    },
    "week_2": { ... },
    "week_3": { ... },
    "week_4": { ... }
  },
  "breed_notes": [
    "Visoko trenabilna pasma — hitro napreduje."
  ]
}
```

---

### `POST /api/chat`
Pametni svetovalec za trening (streaming SSE, brez zunanjega API-ja).

**Zahteva:**
```json
{
  "messages": [{ "role": "user", "content": "Kako naučim Sit?" }],
  "dog_profile": { ... }
}
```

**Odgovor:** `text/event-stream`
```
data: {"text": "Za "}
data: {"text": "Sit "}
data: [DONE]
```

Pokriti tematski sklopi: motivacija, dolžina sej, vlečenje na povodcu, lajanje, skakanje, grizenje, agresija, info o pasmi, priporočila trikov.

---

### `GET /api/model-info`
```json
{
  "model_type": "GradientBoostingRegressor",
  "mae": 0.0617,
  "method_model_accuracy": 0.987,
  "n_breeds": 537,
  "n_tricks": 45,
  "training_samples": 51602
}
```

---

## 🎨 Vizualna identiteta

| Element | Vrednost |
|---------|----------|
| Ime | **UpTail** — "up" = napredek, "tail" = rep (srečen pes) |
| Barvna shema | Črno-bela |
| Naslovna pisava | **Fraunces** (serif, bold 900) |
| Besedilna pisava | **DM Sans** (sans-serif) |
| Logo | Inspiriran iz smrčka Norda (Injin pes) |

---

## ☁️ Deploy

### Railway
```bash
railway login
railway init
railway up
```

### Render
- **Build:** `pip install -r requirements.txt && python train_model.py`
- **Start:** `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT`

> ⚠️ Mapa `model/` ni v repozitoriju. `train_model.py` mora biti del build procesa.

### Lokalni development
```bash
python app.py           # port 5001
# ali za produkcijo:
gunicorn app:app --workers 2 --bind 0.0.0.0:5001
```

---

## 🔧 Razvoj

### Dodajanje novih pasem
Uredi `data/dog_breeds_dataset.xlsx` → dodaj vrstico → `python train_model.py`

### Dodajanje novih trikov
Uredi `data/dog_tricks_dataset.xlsx` (zavihka *Tricks Overview* + *Training Methods*) → `python train_model.py`

### Ponastavi model
```bash
rm -rf model/
python train_model.py
```

---

## 📊 Ključni rezultati (MVP kriteriji)

| Kriterij | Cilj | Rezultat |
|----------|------|----------|
| Relevantna priporočila | ≥ 80% uporabnikov | ✅ 94.4% natančnost |
| Pokritost trikov | 30+ | ✅ 45 trikov |
| Natančnost klasifikacije metode | > 75% | ✅ 98.7% |

---

## 👥 Avtorji

| | |
|---|---|
| **Inja Vozelj** | UX, podatki, testiranje |
| **Aljaž Savšek** | Backend, ML model, deploy |
| **Mentor** | Dr. Uroš Ocepek, prof. |
| **Šola** | Srednja tehniška in poklicna šola Trbovlje |
| **Predmet** | UUI – Uvod v umetno inteligenco |
| **Šolsko leto** | 2025/2026 |

**Repozitorij:** https://github.com/AljazSavsek/UpTail
