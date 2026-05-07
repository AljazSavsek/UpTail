# 🐾 UpTail v2 — Brez API ključa

Popolnoma lokalen AI trening sistem za pse.
**Brez Anthropic / OpenAI / zunanjih API-jev.**

Model je natreniran direktno iz podatkov o pasmah in trikih.

---

## 🚀 Zagon v 3 korakih

```bash
# 1. Namesti odvisnosti
pip install -r requirements.txt

# 2. Natrenira model (samo prvič ~30 sek)
python train_model.py

# 3. Zaženi aplikacijo
python app.py
```

Odprite: **http://localhost:5000**

---

## 📁 Struktura

```
uptail/
├── train_model.py          ← Enkratni trening modela
├── app.py                  ← Flask backend (brez API ključa)
├── requirements.txt
├── Procfile
├── data/
│   ├── dog_breeds_dataset.xlsx
│   ├── dog_tricks_dataset.xlsx
│   └── dawgtraindataset.xlsx
├── model/                  ← Generirano po train_model.py
│   ├── success_model.joblib
│   ├── method_model.joblib
│   ├── breeds_lookup.json
│   ├── tricks_lookup.json
│   └── meta.json
└── templates/
    └── index.html
```

---

## 🧠 ML Model

| Metrika | Vrednost |
|---------|----------|
| Tip | GradientBoostingRegressor |
| Trening samples | 51,602 |
| MAE (test) | 0.056 / 1.0 |
| Method classifier acc | 99.9% |
| Pasme | 537 |
| Trikci | 32 |

**Vhodni features:**
- Trenabilnost, inteligenca, trmavost, občutljivost, energija pasme
- Težavnost trika + minimalne zahteve
- Starost psa, izkušnje lastnika
- Motivacijski profil (hrana / igra / pohvala / clicker)

---

## ☁️ Deploy (brez API ključa!)

### Railway
```bash
railway login && railway init && railway up
```

### Render
- Build: `pip install -r requirements.txt && python train_model.py`
- Start: `gunicorn app:app`

> ⚠️ Na cloudu zaženite `train_model.py` kot del build procesa,
> ker `model/` mapa ni v git repozitoriju.

---

## 🔌 API Endpointi

| Metoda | Pot | Opis |
|--------|-----|------|
| GET | `/api/breeds?q=` | Iskanje pasem |
| POST | `/api/recommend` | ML priporočila trikov |
| POST | `/api/training-plan` | 4-tedenski plan |
| POST | `/api/chat` | Smart advisor (streaming) |
| GET | `/api/model-info` | Model statistike |
