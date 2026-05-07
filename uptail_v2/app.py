"""
UpTail — Flask Backend
ML-powered dog training app — NO external API required.
Uses locally trained GradientBoosting + RandomForest models.
"""

import os, json, re
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, Response

app = Flask(__name__)

# ─────────────────────────────────────────────
# LOAD TRAINED MODELS + LOOKUPS
# ─────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

success_model = joblib.load(f"{MODEL_DIR}/success_model.joblib")
method_model  = joblib.load(f"{MODEL_DIR}/method_model.joblib")

with open(f"{MODEL_DIR}/breeds_lookup.json", encoding="utf-8") as f:
    BREEDS_LOOKUP = json.load(f)   # key: breed_name.lower()

with open(f"{MODEL_DIR}/tricks_lookup.json", encoding="utf-8") as f:
    TRICKS_LIST = json.load(f)     # list of trick dicts

with open(f"{MODEL_DIR}/meta.json") as f:
    META = json.load(f)

FEATURES = META["features"]

EASE_LABELS = [
    (0.80, "Zelo enostavno"),
    (0.65, "Enostavno"),
    (0.45, "Srednje"),
    (0.28, "Zahtevno"),
    (0.00, "Zelo zahtevno"),
]

METHOD_PREF_LABELS = META["method_labels"]

print(f"✅ Models loaded | {len(BREEDS_LOOKUP)} breeds | {len(TRICKS_LIST)} tricks")
print(f"   Success model MAE={META['mae']} | Method acc={META['method_accuracy']}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def ease_label(p: float) -> str:
    for thresh, label in EASE_LABELS:
        if p >= thresh:
            return label
    return "Zelo zahtevno"


def expected_sessions(p: float, diff: int) -> int:
    base = {1: 6, 2: 16, 3: 40}[diff]
    return max(1, round(base * (1.5 - p)))


def get_breed(breed_name: str) -> dict:
    key = breed_name.lower().strip()
    if key in BREEDS_LOOKUP:
        return BREEDS_LOOKUP[key]
    # Fuzzy partial match
    for k, v in BREEDS_LOOKUP.items():
        if key in k or k in key:
            return v
    # Default: average dog
    return {
        "breed_name": breed_name, "trainability_score": 3,
        "intelligence_score": 3, "stubbornness": 5, "sensitivity": 5,
        "energy_score": 3, "energy_level": "Moderate", "size": "Medium",
        "group": "", "good_kids": 7, "good_dogs": 7, "min_exercise": 30,
    }


def build_feature_vector(breed: dict, trick: dict, profile: dict) -> np.ndarray:
    return np.array([[
        int(breed.get("trainability_score", 3)),
        int(breed.get("intelligence_score", 3)),
        int(breed.get("stubbornness", 5)),
        int(breed.get("sensitivity", 5)),
        int(breed.get("energy_score", 3)),
        int(trick.get("difficulty_num", 1)),
        int(trick.get("min_train_num", 1)),
        int(trick.get("min_intel_num", 1)),
        int(profile.get("age_months", 24)),
        int(profile.get("owner_experience", 1)),
        int(bool(profile.get("food_motivated", True))),
        int(bool(profile.get("play_motivated", False))),
        int(bool(profile.get("praise_motivated", False))),
        int(bool(profile.get("clicker_used", False))),
    ]])


def pick_best_method(trick: dict, method_pref_idx: int, profile: dict) -> dict | None:
    methods = trick.get("methods", [])
    if not methods:
        return None
    pref_label = METHOD_PREF_LABELS.get(str(method_pref_idx), "lure")
    food  = bool(profile.get("food_motivated"))
    intel = profile.get("_breed_intel", 3)

    # Score each method
    def score_method(m):
        name = m.get("method_name", "").lower()
        best = m.get("best_for", "").lower()
        s = 0
        if pref_label == "shaping" and "shaping" in name: s += 3
        if pref_label == "lure_food" and "lure" in name:  s += 3
        if pref_label == "guidance" and ("guidance" in name or "edge" in name or "push" in name): s += 3
        if food and "lure" in name: s += 2
        if intel >= 4 and "shaping" in name: s += 2
        if "all breed" in best: s += 1
        return s

    return max(methods, key=score_method)


def recommend_tricks(profile: dict, top_n: int = 12) -> list:
    breed = get_breed(profile.get("breed", ""))
    profile["_breed_intel"] = int(breed.get("intelligence_score", 3))

    results = []
    # Batch prediction — one call per trick
    feature_batch = []
    for trick in TRICKS_LIST:
        feature_batch.append(build_feature_vector(breed, trick, profile)[0])

    X_batch = np.array(feature_batch)
    probs    = np.clip(success_model.predict(X_batch), 0.02, 0.97)
    m_prefs  = method_model.predict(X_batch)

    for i, trick in enumerate(TRICKS_LIST):
        p    = float(probs[i])
        diff = int(trick.get("difficulty_num", 1))
        best_method = pick_best_method(trick, int(m_prefs[i]), profile)

        results.append({
            "trick_name":         trick["trick_name"],
            "category":           trick["category"],
            "difficulty":         trick["difficulty"],
            "prerequisites":      trick.get("prerequisites"),
            "ideal_reward":       trick.get("ideal_reward", ""),
            "avg_learning_time":  trick.get("avg_learning_time", ""),
            "notes":              trick.get("notes", ""),
            "success_probability": round(p, 3),
            "expected_sessions":   expected_sessions(p, diff),
            "ease_label":          ease_label(p),
            "best_method":  best_method.get("method_name") if best_method else None,
            "method_steps": best_method.get("steps")       if best_method else None,
            "method_best_for": best_method.get("best_for") if best_method else None,
        })

    results.sort(key=lambda x: (x["success_probability"], -x["expected_sessions"]), reverse=True)
    return results[:top_n]


# ─────────────────────────────────────────────
# SMART ADVISOR  (replaces AI chat)
# Rule-based NLP over dataset — no API needed
# ─────────────────────────────────────────────

# Keyword → topic map
TOPICS = {
    "motivacija":    ["motivacija", "priboljšek", "nagrada", "hrana", "igra", "pohvala", "nagra"],
    "trik":          ["trik", "trikc", "triku", "naučim", "nauči", "naučit", "začnem", "začni"],
    "problematika":  ["vleč", "laja", "lajanje", "skaka", "grize", "agresij", "anksioz", "nervoz", "nemiren"],
    "seja":          ["seja", "dolg", "trajanje", "koliko časa", "kako pogosto", "kdaj"],
    "pasma":         ["pasma", "breed", "značilnost", "kateri tip"],
    "povodec":       ["povodec", "vlečenje", "hoja"],
    "lajanje":       ["laja", "lajanje", "bark"],
    "skakanje":      ["skaka", "skok"],
    "grizenje":      ["grize", "grizenj"],
    "agresija":      ["agresij", "grozi", "renči"],
}

ADVICE_DB = {
    "motivacija": {
        "food": "Hrana je najmočnejši motivator za večino psov. Uporabite visoko vredne priboljške (piščanec, sir) za nove trike, nato postopoma preidite na navadne. Seji naj bodo kratke (5–8 min) — pes mora ostati motiviran.",
        "play": "Igra deluje odlično pri energičnih pasemah (Husky, Border Collie, Aussie). Po vsakem uspešnem ukazu kratka igra z igračo (5–10 sek). Clicker + igra je močna kombinacija.",
        "praise": "Verbalna pohvala deluje pri zelo občutljivih pasmah (Golden, Cavalier). Glasen, vesel ton je ključen. Kombinirajte s fizičnim stikom, če pes to uživa.",
        "default": "Kombinirajte vse tri motivatorje — začnite z visoko vrednostjo (hrana), dodajte igro in pohvalo. Opazujte, kaj vašega psa najbolj navduši.",
    },
    "seja": "Optimalna dolžina seje: **5–10 minut** za mladiče in začetnike, **10–15 minut** za odrasle pse z izkušnjami. Vadite **2–3× dnevno** — kratke, pogoste seje so bolj učinkovite kot ena dolga. Vedno končajte z uspehom in pohvalo.",
    "povodec": "Za vlečenje povodca:\n1. Ko pes vleče → takoj se ustavite, brez besed\n2. Počakajte da povodec popusti → pohvalite + nadaljujte\n3. Spremenite smer brez opozorila\n4. Nagradite vsako hojo ob nogi z visoko vrednimi priboljški\nConsistency je ključna — enako naj reagirajo vsi člani družine.",
    "lajanje": "Za prekomerno lajanje:\n1. Ugotovite sprožilec (tujci, zvonec, osamljenost)\n2. Naučite ukaz 'tiho' — nagradite vsako sekundo tišine\n3. Desenzibilizacija: postopna izpostavljenost sprožilcu + nagrada\n4. Nikoli ne kričite — pes misli, da lajate skupaj",
    "skakanje": "Za skakanje na ljudi:\n1. Obrnite hrbet in ignorirajte — brez kontakta, brez besed\n2. Nagradite SAMO ko so vse 4 noge na tleh\n3. Prosite goste za enako reakcijo\n4. Učite alternativo: 'sedi' ob prihodu gosta",
    "grizenje": "Za grizenje/grizljanje:\n1. Ob grizenj takoj prekinite igro z glasnim 'Au!' in zamrznite\n2. Ponudite igračo kot alternativo\n3. Vadite 'mehka usta' z nadzorovanim stikom\n4. Zagotovite dovolj fizične in mentalne stimulacije",
    "agresija": "Za agresijo priporočamo:\n1. Ohranite varno razdaljo (pod pragom reakcije)\n2. Counter-conditioning: sprožilec = priboljšek (ne stik)\n3. Strukturirane paralele hoje — ne srečanje z nosom\n4. ⚠ Priporočamo posvet z izkušenim veterinarskim vedenjskim strokovnjakom",
}

def get_trick_advice(trick_name: str, breed_data: dict) -> str:
    trick = next((t for t in TRICKS_LIST if t["trick_name"].lower() in trick_name.lower()), None)
    if not trick:
        return f"Za trik '{trick_name}' priporočam lure metodo z visoko vrednimi priboljški. Začnite v mirnem okolju brez motenj."

    methods = trick.get("methods", [])
    tr = breed_data.get("trainability_score", 3)
    food = True  # default assumption

    # Pick suitable method
    if methods:
        if tr >= 4:
            m = next((x for x in methods if "shaping" in x["method_name"].lower()), methods[0])
        elif food:
            m = next((x for x in methods if "lure" in x["method_name"].lower()), methods[0])
        else:
            m = methods[0]

        steps = m.get("steps", "")
        # Format steps nicely
        steps_fmt = re.sub(r"(\d+\.)", r"\n\1", steps).strip()
        return (
            f"**{trick['trick_name']}** — {trick['difficulty']} ({trick['avg_learning_time']})\n\n"
            f"Metoda: **{m['method_name']}**\n"
            f"Najboljše za: {m['best_for']}\n\n"
            f"{steps_fmt}\n\n"
            f"💡 Nagrada: {trick['ideal_reward']}"
        )

    return f"{trick.get('notes', '')} Čas učenja: {trick.get('avg_learning_time', '1–2 tedna')}."


def smart_advisor(message: str, profile: dict) -> str:
    msg_lower = message.lower()
    breed_data = get_breed(profile.get("breed", ""))
    dog_name = profile.get("dog_name") or "vaš pes"
    breed_name = profile.get("breed") or "mešanec"
    tr = breed_data.get("trainability_score", 3)
    stub = breed_data.get("stubbornness", 5)

    # Detect if asking about specific trick
    for trick in TRICKS_LIST:
        tn = trick["trick_name"].lower()
        if tn in msg_lower or any(w in msg_lower for w in tn.split()):
            return get_trick_advice(trick["trick_name"], breed_data)

    # Detect topic
    for topic, keywords in TOPICS.items():
        if any(kw in msg_lower for kw in keywords):
            if topic == "motivacija":
                food  = bool(profile.get("food_motivated"))
                play  = bool(profile.get("play_motivated"))
                praise = bool(profile.get("praise_motivated"))
                if food:   return ADVICE_DB["motivacija"]["food"]
                if play:   return ADVICE_DB["motivacija"]["play"]
                if praise: return ADVICE_DB["motivacija"]["praise"]
                return ADVICE_DB["motivacija"]["default"]
            if topic == "seja":         return ADVICE_DB["seja"]
            if topic == "povodec":      return ADVICE_DB["povodec"]
            if topic == "lajanje":      return ADVICE_DB["lajanje"]
            if topic == "skakanje":     return ADVICE_DB["skakanje"]
            if topic == "grizenje":     return ADVICE_DB["grizenje"]
            if topic == "agresija":     return ADVICE_DB["agresija"]
            if topic == "pasma":
                return (
                    f"**{breed_data.get('breed_name', breed_name)}** — {breed_data.get('group','')} skupina\n\n"
                    f"• Trenabilnost: {tr}/5\n"
                    f"• Inteligenca: {breed_data.get('intelligence_score',3)}/5\n"
                    f"• Trmavost: {stub}/10\n"
                    f"• Energija: {breed_data.get('energy_level','Moderate')}\n"
                    f"• Min. gibanje: {breed_data.get('min_exercise',30)} min/dan\n\n"
                    f"{'Visoko trenabilna pasma — hitro napreduje, priporočam kompleksnejše trike.' if tr >= 4 else ''}"
                    f"{'Trmava pasma — bodite potrpežljivi, kratke seje, visoka nagrada.' if stub >= 7 else ''}"
                )
            if topic == "trik":
                # Get top 3 recommended tricks
                recs = recommend_tricks(profile, top_n=3)
                lines = "\n".join([
                    f"• **{t['trick_name']}** — {t['ease_label']} ({int(t['success_probability']*100)}% uspešnost, ~{t['expected_sessions']} sej)"
                    for t in recs
                ])
                return f"Za {dog_name} ({breed_name}) priporočam začeti s temi triki:\n\n{lines}\n\nKliknite na trik v zavihku 🎯 za korak-po-korak navodila."

    # Greeting/general
    if any(w in msg_lower for w in ["pozdravljeni", "pozdrav", "živjo", "hej", "hello", "hi"]):
        return (
            f"Pozdravljeni! 🐾 Sem UpTail trening sistem za {dog_name}.\n\n"
            f"Vprašajte me o:\n• Specifičnih trikih (Sit, Stay, Fetch...)\n"
            f"• Vedenjskih problemih (vlečenje, lajanje...)\n"
            f"• Optimalni dolžini sej\n• Motivatorjih za {breed_name}"
        )

    # Fallback — breed-aware generic advice
    exp = int(profile.get("owner_experience", 1))
    if exp == 1:
        return (
            f"Za začetnike priporočam začeti z **osnovnimi ukazi**: Sit → Stay → Come → Down.\n"
            f"Seji naj bodo kratke (5 min) in vedno pozitivne. "
            f"{'Pri ' + breed_name + ' je ključna potrpežljivost — trmava pasma.' if stub >= 7 else 'Pasma dobro reagira na pozitivno ojačanje.'}"
        )
    return (
        f"Odlično vprašanje za {dog_name}! Za naprednejši trening z {breed_name} priporočam:\n"
        f"• Shaping metodo za kompleksne trike\n"
        f"• Povečajte distrakcije postopoma (3D: Duration → Distance → Distraction)\n"
        f"• Proofing v različnih okoljih je ključen za zanesljive ukaze"
    )


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/breeds")
def api_breeds():
    q = request.args.get("q", "").lower().strip()
    if q:
        matches = [v["breed_name"] for k, v in BREEDS_LOOKUP.items() if q in k]
    else:
        matches = [v["breed_name"] for v in BREEDS_LOOKUP.values()]
    return jsonify({"breeds": sorted(matches)[:60]})


@app.route("/api/breed/<breed_name>")
def api_breed(breed_name):
    return jsonify(get_breed(breed_name))


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    body  = request.get_json()
    top_n = int(body.get("top_n", 12))
    tricks = recommend_tricks(body, top_n=top_n)
    breed_info = get_breed(body.get("breed", ""))
    return jsonify({
        "tricks":     tricks,
        "breed_info": breed_info,
        "dog_name":   body.get("dog_name", "Vaš pes"),
        "model_mae":  META["mae"],
    })


@app.route("/api/training-plan", methods=["POST"])
def api_training_plan():
    body    = request.get_json()
    profile = body.get("dog_profile", body)
    tricks  = recommend_tricks(profile, top_n=14)
    breed   = get_breed(profile.get("breed", ""))

    beginner     = [t for t in tricks if t["difficulty"] == "Beginner"]
    intermediate = [t for t in tricks if t["difficulty"] == "Intermediate"]
    advanced     = [t for t in tricks if t["difficulty"] == "Advanced"]

    plan = {
        "week_1": {"focus": "Osnove in zaupanje",          "tricks": beginner[:2],     "sessions_per_day": 2, "session_length_min": 5},
        "week_2": {"focus": "Utrjevanje + novi ukazi",      "tricks": beginner[2:4],    "sessions_per_day": 2, "session_length_min": 7},
        "week_3": {"focus": "Napredek in kompleksnost",     "tricks": intermediate[:2], "sessions_per_day": 2, "session_length_min": 10},
        "week_4": {"focus": "Generalizacija in proofing",   "tricks": intermediate[2:4],"sessions_per_day": 3, "session_length_min": 10},
    }

    notes = []
    if breed.get("stubbornness", 5) >= 7:
        notes.append("Trmava pasma — kratke seje (5 min), visoka vrednost nagrade, konec vedno z uspehom.")
    if breed.get("sensitivity", 5) >= 8:
        notes.append("Zelo občutljiv pes — mehak glas, nikoli frustracija, pozitivno vzdušje.")
    if breed.get("trainability_score", 3) >= 4:
        notes.append("Visoko trenabilna pasma — hitro napreduje, dodajte mentalne izzive.")
    if breed.get("energy_score", 3) >= 4:
        notes.append("Energična pasma — pred treningom krajši sprehod za umiritev.")

    return jsonify({"plan": plan, "breed_notes": notes, "breed_info": breed})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Smart rule-based advisor using ML model + dataset knowledge."""
    body    = request.get_json()
    messages = body.get("messages", [])
    profile  = body.get("dog_profile", {})

    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )

    reply = smart_advisor(last_user, profile)

    def stream():
        import time
        # Stream word by word for natural feel
        words = reply.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words)-1 else "")
            yield f"data: {json.dumps({'text': chunk})}\n\n"
            time.sleep(0.02)
        yield "data: [DONE]\n\n"

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/trick/<trick_name>/methods")
def api_trick_methods(trick_name):
    trick = next((t for t in TRICKS_LIST if t["trick_name"].lower() == trick_name.lower()), None)
    if not trick:
        trick = next((t for t in TRICKS_LIST if trick_name.lower() in t["trick_name"].lower()), None)
    if not trick:
        return jsonify({"methods": [], "trick": trick_name})
    return jsonify({"methods": trick.get("methods", []), "trick": trick["trick_name"]})


@app.route("/api/model-info")
def api_model_info():
    return jsonify({
        "model_type": "GradientBoostingRegressor",
        "mae": META["mae"],
        "method_model_accuracy": META["method_accuracy"],
        "n_breeds": len(BREEDS_LOOKUP),
        "n_tricks": len(TRICKS_LIST),
        "features": FEATURES,
        "training_samples": 51602,
    })


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)
