"""
UpTail v2 — Flask Backend
Lokalni ML model, brez zunanjega API ključa.
Port: 5001
"""
import os, json, re
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Pot do modela ──────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

# ── Naloži modele ──────────────────────────────────────────────
_model_path = os.path.join(MODEL_DIR, "success_model.joblib")
if not os.path.exists(_model_path):
    print("=" * 60)
    print("MODEL NI NAJDEN!")
    print("Najprej zazenite:  python train_model.py")
    print("=" * 60)
    raise SystemExit(1)

try:
    success_model = joblib.load(os.path.join(MODEL_DIR, "success_model.joblib"))
    method_model  = joblib.load(os.path.join(MODEL_DIR, "method_model.joblib"))
    with open(os.path.join(MODEL_DIR, "breeds_lookup.json"), encoding="utf-8") as f:
        BREEDS = json.load(f)
    with open(os.path.join(MODEL_DIR, "tricks_lookup.json"), encoding="utf-8") as f:
        TRICKS = json.load(f)
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
        META = json.load(f)
    METHOD_LABELS = META.get("method_labels", {})
    print(f"OK: {len(BREEDS)} pasem | {len(TRICKS)} trikov | MAE={META['mae']}")
except Exception as e:
    print(f"NAPAKA pri nalaganju modela: {e}")
    print("Zazenite: python train_model.py")
    raise SystemExit(1)


# ── Pomozne funkcije ───────────────────────────────────────────

def clean(v, default=None):
    if v is None:
        return default
    s = str(v).strip()
    return default if s.lower() in ("nan", "none", "") else s


def ease_label(p):
    if p >= 0.80: return "Zelo enostavno"
    if p >= 0.65: return "Enostavno"
    if p >= 0.45: return "Srednje"
    if p >= 0.28: return "Zahtevno"
    return "Zelo zahtevno"


def expected_sessions(p, diff):
    base = {1: 6, 2: 16, 3: 40}.get(int(diff), 6)
    return max(1, round(base * (1.5 - p)))


def get_breed(name):
    key = str(name or "").lower().strip()
    if key in BREEDS:
        return BREEDS[key]
    for k, v in BREEDS.items():
        if key and (key in k or k in key):
            return v
    return {
        "breed_name": name or "Neznan pes",
        "trainability_score": 3, "intelligence_score": 3,
        "stubbornness": 5, "sensitivity": 5,
        "energy_score": 3, "energy_level": "Moderate",
        "size": "Medium", "group": "Mixed",
        "good_kids": 7, "good_dogs": 7, "min_exercise": 30,
    }


def build_fv(breed, trick, profile):
    return [
        int(breed.get("trainability_score", 3)),
        int(breed.get("intelligence_score", 3)),
        int(breed.get("stubbornness", 5)),
        int(breed.get("sensitivity", 5)),
        int(breed.get("energy_score", 3)),
        int(trick.get("difficulty_num", 1)),
        int(trick.get("min_train_num", 1)),
        int(trick.get("min_intel_num", 1)),
        min(240, max(1, int(profile.get("age_months", 24)))),
        min(3,   max(1, int(profile.get("owner_experience", 1)))),
        1 if profile.get("food_motivated")   else 0,
        1 if profile.get("play_motivated")   else 0,
        1 if profile.get("praise_motivated") else 0,
        1 if profile.get("clicker_used")     else 0,
    ]


def best_method(trick, m_pref_idx, profile):
    methods = trick.get("methods") or []
    if not methods:
        return None
    pref  = METHOD_LABELS.get(str(m_pref_idx), "lure")
    food  = bool(profile.get("food_motivated"))
    intel = int(profile.get("_intel", 3))

    def score(m):
        n = (m.get("method_name") or "").lower()
        b = (m.get("best_for")    or "").lower()
        s = 0
        if "shaping" in n and pref == "shaping":   s += 3
        if "lure"    in n and pref == "lure_food": s += 3
        if "lure"    in n and food:                s += 2
        if "shaping" in n and intel >= 4:          s += 2
        if "all breed" in b:                       s += 1
        return s

    return max(methods, key=score)


def recommend(profile, top_n=12):
    breed = get_breed(profile.get("breed", ""))
    profile["_intel"] = int(breed.get("intelligence_score", 3))
    X       = np.array([build_fv(breed, t, profile) for t in TRICKS])
    probs   = np.clip(success_model.predict(X), 0.02, 0.97).tolist()
    m_prefs = method_model.predict(X).tolist()
    out = []
    for i, trick in enumerate(TRICKS):
        p    = float(probs[i])
        diff = int(trick.get("difficulty_num", 1))
        m    = best_method(trick, int(m_prefs[i]), profile)
        out.append({
            "trick_name":          trick["trick_name"],
            "category":            clean(trick.get("category"),          "General"),
            "difficulty":          clean(trick.get("difficulty"),        "Beginner"),
            "prerequisites":       clean(trick.get("prerequisites")),
            "ideal_reward":        clean(trick.get("ideal_reward"),      "Hrana"),
            "avg_learning_time":   clean(trick.get("avg_learning_time"), "1-2 tedna"),
            "notes":               clean(trick.get("notes")),
            "success_probability": round(p, 3),
            "expected_sessions":   expected_sessions(p, diff),
            "ease_label":          ease_label(p),
            "best_method":         clean(m.get("method_name")) if m else None,
            "method_steps":        clean(m.get("steps"))       if m else None,
            "method_best_for":     clean(m.get("best_for"))    if m else None,
        })
    out.sort(key=lambda x: x["success_probability"], reverse=True)
    return out[:top_n]


# ── Svetovalec ─────────────────────────────────────────────────

TOPICS = {
    "motivacija": ["motivacija", "priboljsek", "nagrada", "hrana", "igra", "pohvala",
                   "priboljšek"],
    "trik":       ["trik", "triku", "naucim", "nauci", "zacnem", "priporoc",
                   "naučim", "nauči", "začnem", "priporoč", "katere"],
    "seja":       ["seja", "dolg", "trajanje", "kako pogosto", "kako dolgo", "kdaj"],
    "pasma":      ["pasma", "breed", "znacilnost", "kaksen je",
                   "značilnost", "kakšen je"],
    "povodec":    ["povodec", "povodcu", "vlece", "vlecenje", "vlec", "pull", "hoja",
                   "vleče", "vlečenje", "vleč"],
    "lajanje":    ["laja", "lajanje", "bark"],
    "skakanje":   ["skaka", "skoci", "skace", "skok",
                   "skoči", "skače"],
    "grizenje":   ["grize", "grizenj", "grizlja", "bite"],
    "agresija":   ["agresij", "grozi", "renci", "napada",
                   "renči"],
}

ADVICE = {
    "motivacija": {
        "food":    "Hrana je najmocnejsi motivator. Uporabite visoko vredne priboljeske (piscanec, sir) za nove trike. Seji naj bodo kratke (5-8 min).",
        "play":    "Igra deluje odlicno pri energicnih pasmah. Po uspesnem ukazu kratka igra z igaco (5-10 sek). Clicker + igra je mocna kombinacija.",
        "praise":  "Pohvala deluje pri obcutljivih pasmah (Golden, Cavalier). Glasen vesel ton + fizicni stik, ce pes to uziva.",
        "default": "Kombinirajte motivatorje - pricnite z visoko vrednostjo (hrana), dodajte igro in pohvalo.",
    },
    "seja":    "Optimalna dolzina seje: 5-10 min za mladice, 10-15 min za odrasle. Vadite 2-3x dnevno. Vedno koncajte z uspehom!",
    "povodec": "Vlecenje povodca:\n1. Ko pes vlece - takoj se ustavite\n2. Pocakajte da se povodec sprosti - pohvalite in nadaljujte\n3. Spremenite smer brez opozorila\n4. Nagradite vsako hojo ob nogi",
    "lajanje": "Prekomerno lajanje:\n1. Ugotovite sprozilec (tujci, zvonec)\n2. Naucite ukaz 'tiho' - nagradite vsako sekundo tisine\n3. Nikoli ne kricite - pes misli, da lajate skupaj",
    "skakanje": "Skakanje na ljudi:\n1. Obrnite hrbet in ignorirajte\n2. Nagradite SAMO ko so vse 4 noge na tleh\n3. Naucite alternativo: 'sedi' ob prihodu gosta",
    "grizenje": "Grizenje:\n1. Ob grizenju takoj prekinite igro\n2. Ponudite igaco kot alternativo\n3. Zagotovite dovolj fizicne in mentalne stimulacije",
    "agresija": "Agresija:\n1. Ohranite varno razdaljo\n2. Counter-conditioning: sprozilec = priboljesek\n3. Priporocamo posvet z veterinarskim vedenjskim strokovnjakom",
}

GREETINGS = ["zivjo", "hej", "pozdravljeni", "pozdrav", "hello", "hi",
             "živjo", "dobro jutro", "dober dan"]


def advisor(message, profile):
    # Preden pošljemo zahtevek, poskusimo pridobiti funkcijo get_breed
    try:
        breed = get_breed(profile.get("breed", ""))
    except NameError:
        breed = {}  # Fallback, če funkcija get_breed ni definirana v vaši kodi

    # 🔑 Koda prebere ključ neposredno iz okolja (Environment), ki ga nastavi Render
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Če ključa ni v okolju, koda takoj sproži napako (lažje za debugiranje)
    if not api_key:
        return "Napaka: API ključ 'GROQ_API_KEY' ni nastavljen v okoljskih spremenljivkah na Renderju!"

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},  # Vstavljen skriti ključ
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": f"""Ti si UpTail, profesionalni AI trener psov.
Vedno odgovarjaj v slovenscini. Bodi kratek in konkreten.
Pes: {profile.get('dog_name','neznan')}, {profile.get('breed','neznan')}, {profile.get('age_months',12)} mesecev
Trenabilnost: {breed.get('trainability_score',3)}/5, Energija: {breed.get('energy_level','Moderate')}
Motivacija: {'hrana ' if profile.get('food_motivated') else ''}{'igra ' if profile.get('play_motivated') else ''}{'pohvala' if profile.get('praise_motivated') else ''}
Pravila: ce te vprasajo ime reci 'Sem UpTail'. Najvec 5 stavkov. Vedno omeni psa po imenu."""
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.3
            },
            timeout=30
        )
        return r.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        # Če pride do katerekoli druge napake (npr. timeout ali napačen ključ)
        exp = int(profile.get("owner_experience", 1))
        if exp == 1:
            return ("Za zacetnike priporocam: Sit, Down, Stay, Come.\n"
                    "Kratke seje (5 min), 2-3x dnevno, vedno pozitivno.")
        return ("Za naprednejsi trening:\n"
                "- Shaping metoda za kompleksne trike\n"
                "- Postopno povecevanje distrakcij")
# ── Flask Routes ───────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-info")
def api_model_info():
    return jsonify({
        "model_type":            "GradientBoostingRegressor",
        "mae":                   META.get("mae", 0),
        "method_model_accuracy": META.get("method_accuracy", 0),
        "n_breeds":              len(BREEDS),
        "n_tricks":              len(TRICKS),
        "training_samples":      51602,
    })


@app.route("/api/breeds")
def api_breeds():
    q       = request.args.get("q", "").lower().strip()
    matches = [v["breed_name"] for k, v in BREEDS.items() if not q or q in k]
    return jsonify({"breeds": sorted(set(matches))})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    body = request.get_json(force=True, silent=True) or {}
    try:
        tricks = recommend(body, top_n=int(body.get("top_n", 12)))
        breed  = get_breed(body.get("breed", ""))
        return jsonify({
            "tricks":     tricks,
            "breed_info": breed,
            "dog_name":   body.get("dog_name") or "Vas pes",
            "model_mae":  META.get("mae", 0),
        })
    except Exception as e:
        app.logger.error(f"recommend error: {e}")
        return jsonify({"error": str(e), "tricks": [], "breed_info": {}}), 500


@app.route("/api/training-plan", methods=["POST"])
def api_training_plan():
    body    = request.get_json(force=True, silent=True) or {}
    profile = body.get("dog_profile") or body
    try:
        all_t = recommend(profile, top_n=30)
        beg   = [t for t in all_t if t["difficulty"] == "Beginner"]
        mid   = [t for t in all_t if t["difficulty"] == "Intermediate"]
        adv   = [t for t in all_t if t["difficulty"] == "Advanced"]

        def fill(lst, n):
            if len(lst) >= n:
                return lst[:n]
            extra = [t for t in all_t if t not in lst]
            return (lst + extra)[:n]

        plan = {
            "week_1": {"focus": "Osnove in zaupanje",        "sessions_per_day": 2, "session_length_min": 5,  "tricks": fill(beg, 2)},
            "week_2": {"focus": "Utrjevanje + novi ukazi",   "sessions_per_day": 2, "session_length_min": 7,  "tricks": fill(beg[2:], 2)},
            "week_3": {"focus": "Napredek in kompleksnost",  "sessions_per_day": 2, "session_length_min": 10, "tricks": fill(mid, 2)},
            "week_4": {"focus": "Generalizacija in proofing","sessions_per_day": 3, "session_length_min": 10, "tricks": fill(mid[2:] or adv, 2)},
        }
        breed = get_breed(profile.get("breed", ""))
        notes = []
        if int(breed.get("stubbornness",       5)) >= 7: notes.append("Trmava pasma - kratke seje, visoka vrednost nagrade.")
        if int(breed.get("sensitivity",        5)) >= 8: notes.append("Obcutljiv pes - mehak glas, brez frustracije.")
        if int(breed.get("trainability_score", 3)) >= 4: notes.append("Visoko trenabilna pasma - dodajte mentalne izzive.")
        if int(breed.get("energy_score",       3)) >= 4: notes.append("Energicna pasma - pred treningom kratek sprehod.")
        return jsonify({"plan": plan, "breed_notes": notes, "breed_info": breed})
    except Exception as e:
        app.logger.error(f"plan error: {e}")
        return jsonify({"error": str(e), "plan": {}, "breed_notes": []}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body    = request.get_json(force=True, silent=True) or {}
    msgs    = body.get("messages") or []
    profile = body.get("dog_profile") or {}
    last    = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
    if not last.strip():
        return jsonify({"reply": "Prosim, vnesite vprasanje."})
    try:
        reply = advisor(last.strip(), profile)
        return jsonify({"reply": reply or "Priporocam zaceti s trikom Sit."})
    except Exception as e:
        app.logger.error(f"chat error: {e}")
        return jsonify({"reply": f"Napaka: {e}. Priporocam Sit - Down - Stay."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
