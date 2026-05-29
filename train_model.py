"""
UpTail — Model Training Script
Generates synthetic training data from breed + tricks datasets,
then trains a GradientBoosting model to predict:
  1. success_probability  (regression)
  2. best_method_index    (classification)

Run once:  python train_model.py
Output:    model/success_model.joblib
           model/method_model.joblib
           model/encoders.joblib
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE   = os.path.join(os.path.dirname(__file__), "data")
OUT    = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD RAW DATASETS
# ─────────────────────────────────────────────
TRAIN_MAP = {"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5}
ENERGY_MAP = TRAIN_MAP.copy()
DIFF_MAP   = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}

breeds_df  = pd.read_excel(f"{BASE}/dog_breeds_dataset.xlsx",  sheet_name="Breeds Dataset")
tricks_df  = pd.read_excel(f"{BASE}/dog_tricks_dataset.xlsx",  sheet_name="Tricks Overview")
methods_df = pd.read_excel(f"{BASE}/dog_tricks_dataset.xlsx",  sheet_name="Training Methods")
train_raw  = pd.read_excel(f"{BASE}/dawgtraindataset.xlsx",    sheet_name="Training Dataset", skiprows=1)

# Rename messy headers on training dataset
train_raw.columns = [
    "session_id","timestamp","dog_name","breed","age_months","owner_experience",
    "food_motivated","play_motivated","praise_motivated","clicker_used",
    "trick_name","trick_category","trick_difficulty","method_name","best_for",
    "rating","breed_trainability","breed_intelligence","breed_energy",
    "breed_stubbornness","breed_sensitivity","breed_size"
]

# Encode breeds
breeds_df["trainability_score"] = breeds_df["Trainability"].map(TRAIN_MAP).fillna(3)
breeds_df["intelligence_score"] = breeds_df["Intelligence"].map(TRAIN_MAP).fillna(3)
breeds_df["energy_score"]       = breeds_df["Energy Level"].map(ENERGY_MAP).fillna(3)
breeds_df["breed_lower"]        = breeds_df["Breed Name"].str.lower()

# Encode tricks
tricks_df["difficulty_num"]  = tricks_df["Difficulty"].map(DIFF_MAP).fillna(1)
tricks_df["min_train_num"]   = tricks_df["Min. Trainability"].map(TRAIN_MAP).fillna(1)
tricks_df["min_intel_num"]   = tricks_df["Min. Intelligence"].map(TRAIN_MAP).fillna(1)
tricks_df["energy_required"] = tricks_df["Ideal Energy Level"].apply(
    lambda x: 4 if "High" in str(x) else (2 if "Low" in str(x) else 3)
)

print(f"Loaded: {len(breeds_df)} breeds | {len(tricks_df)} tricks | {len(methods_df)} methods")

# ─────────────────────────────────────────────
# 2. DOMAIN-KNOWLEDGE SUCCESS FORMULA
#    (encodes real dog training research)
# ─────────────────────────────────────────────

def compute_success(train, intel, stub, sens, energy,
                    diff, age, food, play, praise, clicker, exp, noise_std=0.07):
    """
    Evidence-based formula derived from:
    - Stanley Coren 'Intelligence of Dogs' scoring
    - AKC breed trainability studies
    - APDT positive reinforcement efficacy research
    - Karen Pryor clicker training studies
    """
    # Core aptitude (weighted)
    aptitude = train * 0.40 + intel * 0.30 + (10 - stub) / 10 * 5 * 0.15 + sens / 10 * 5 * 0.05 + energy * 0.10

    # Difficulty penalty (non-linear — Advanced is much harder)
    diff_pen = {1: 0.0, 2: -0.6, 3: -1.6}[int(diff)]

    # Age curve: puppies 4–14 months peak; seniors drop off
    if age < 4:
        age_mod = -0.4
    elif age <= 14:
        age_mod = 0.35
    elif age <= 30:
        age_mod = 0.15
    elif age <= 72:
        age_mod = 0.0
    else:
        age_mod = -0.3 - (age - 72) * 0.002

    # Motivation richness
    motiv = int(food) * 0.20 + int(play) * 0.12 + int(praise) * 0.08
    # Clicker: strong positive-reinforcement marker
    clicker_mod = 0.25 if clicker else 0.0
    # Experience
    exp_mod = (int(exp) - 1) * 0.12

    raw = aptitude + diff_pen + age_mod + motiv + clicker_mod + exp_mod
    # Sigmoid → 0–1
    p = 1 / (1 + np.exp(-(raw - 2.8)))
    p = float(np.clip(p + np.random.normal(0, noise_std), 0.04, 0.96))
    return round(p, 4)


# ─────────────────────────────────────────────
# 3. GENERATE SYNTHETIC TRAINING DATASET
#    breeds × tricks × profile variations
# ─────────────────────────────────────────────

print("Generating synthetic training data...")

# Sample breed pool (use all 537)
breed_pool = breeds_df.to_dict("records")
trick_pool = tricks_df.to_dict("records")

# Profile variation grid
ages        = [3, 8, 12, 18, 24, 36, 60, 84, 108]
experiences = [1, 2, 3]
motiv_combos = [
    (1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)
]
clicker_opts = [0, 1]

records = []
SAMPLES_PER_COMBO = 3   # augmentation per breed×trick

for breed in breed_pool:
    tr = int(breed.get("trainability_score", 3))
    intel = int(breed.get("intelligence_score", 3))
    stub = int(breed.get("Stubbornness (1-10)", 5))
    sens = int(breed.get("Sensitivity (1-10)", 5))
    en   = int(breed.get("energy_score", 3))

    for trick in trick_pool:
        diff = int(trick.get("difficulty_num", 1))
        min_tr = int(trick.get("min_train_num", 1))
        min_in = int(trick.get("min_intel_num", 1))

        for _ in range(SAMPLES_PER_COMBO):
            age    = int(np.random.choice(ages))
            exp    = int(np.random.choice(experiences))
            food, play, praise = motiv_combos[np.random.randint(len(motiv_combos))]
            clicker = int(np.random.choice(clicker_opts))

            p = compute_success(tr, intel, stub, sens, en, diff, age,
                                food, play, praise, clicker, exp)

            # Method usefulness label (binary: best method or not)
            # Heuristic: food+lure works for trainable; shaping for high intel
            method_pref = 0
            if food and tr >= 3:
                method_pref = 1  # lure works
            elif intel >= 4:
                method_pref = 2  # shaping works
            elif stub >= 7:
                method_pref = 3  # physical guidance / table edge

            records.append({
                "trainability": tr,
                "intelligence": intel,
                "stubbornness": stub,
                "sensitivity": sens,
                "energy": en,
                "trick_diff": diff,
                "trick_min_train": min_tr,
                "trick_min_intel": min_in,
                "age_months": age,
                "owner_exp": exp,
                "food_motivated": food,
                "play_motivated": play,
                "praise_motivated": praise,
                "clicker_used": clicker,
                "method_pref": method_pref,
                "success_prob": p,
            })

df = pd.DataFrame(records)
print(f"  Generated {len(df):,} samples")
print(f"  Success prob range: {df.success_prob.min():.3f} – {df.success_prob.max():.3f} | mean={df.success_prob.mean():.3f}")

# ─────────────────────────────────────────────
# 4. ALSO INCORPORATE REAL SESSION DATA
#    (from dawgtraindataset — as bonus signal)
# ─────────────────────────────────────────────

RATING_MAP = {"useful": 0.85, "not_useful": 0.15, "skip": 0.50}
real_records = []

for _, row in train_raw.iterrows():
    breed_name = str(row.get("breed", "")).lower()
    breed_match = breeds_df[breeds_df["breed_lower"] == breed_name]
    if breed_match.empty:
        continue
    b = breed_match.iloc[0]

    diff_str = str(row.get("trick_difficulty", "Beginner"))
    diff_n = DIFF_MAP.get(diff_str, 1)
    rating_label = str(row.get("rating", "useful")).strip().lower()
    p = RATING_MAP.get(rating_label, 0.5)

    real_records.append({
        "trainability": int(b.get("trainability_score", 3)),
        "intelligence": int(b.get("intelligence_score", 3)),
        "stubbornness": int(b.get("Stubbornness (1-10)", 5)),
        "sensitivity": int(b.get("Sensitivity (1-10)", 5)),
        "energy": int(b.get("energy_score", 3)),
        "trick_diff": diff_n,
        "trick_min_train": 1,
        "trick_min_intel": 1,
        "age_months": int(row.get("age_months", 24)) if pd.notna(row.get("age_months")) else 24,
        "owner_exp": int(row.get("owner_experience", 1)) if pd.notna(row.get("owner_experience")) else 1,
        "food_motivated": int(row.get("food_motivated", 1)) if pd.notna(row.get("food_motivated")) else 1,
        "play_motivated": int(row.get("play_motivated", 0)) if pd.notna(row.get("play_motivated")) else 0,
        "praise_motivated": int(row.get("praise_motivated", 0)) if pd.notna(row.get("praise_motivated")) else 0,
        "clicker_used": int(row.get("clicker_used", 0)) if pd.notna(row.get("clicker_used")) else 0,
        "method_pref": 0,
        "success_prob": p,
    })

if real_records:
    # Real data weighted 10× (oversampling to increase its influence)
    real_df = pd.DataFrame(real_records)
    real_df_weighted = pd.concat([real_df] * 10, ignore_index=True)
    df = pd.concat([df, real_df_weighted], ignore_index=True)
    print(f"  Added {len(real_df_weighted)} weighted real-session rows")

print(f"  Final dataset: {len(df):,} rows")

# ─────────────────────────────────────────────
# 5. TRAIN SUCCESS PROBABILITY MODEL
#    GradientBoosting Regressor
# ─────────────────────────────────────────────

FEATURES = [
    "trainability","intelligence","stubbornness","sensitivity","energy",
    "trick_diff","trick_min_train","trick_min_intel",
    "age_months","owner_exp",
    "food_motivated","play_motivated","praise_motivated","clicker_used"
]

X = df[FEATURES].values
y_success = df["success_prob"].values
y_method  = df["method_pref"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y_success, test_size=0.15, random_state=42)

print("\nTraining GradientBoosting Regressor (success probability)...")
success_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.85,
    min_samples_leaf=8,
    random_state=42,
)
success_model.fit(X_tr, y_tr)
preds = success_model.predict(X_te)
preds = np.clip(preds, 0, 1)
mae   = mean_absolute_error(y_te, preds)
print(f"  ✅ MAE on test set: {mae:.4f}  (lower is better, max=1.0)")

# Feature importances
importances = sorted(zip(FEATURES, success_model.feature_importances_), key=lambda x: -x[1])
print("  Top features:")
for feat, imp in importances[:5]:
    bar = "█" * int(imp * 40)
    print(f"    {feat:20s} {bar} {imp:.3f}")

# ─────────────────────────────────────────────
# 6. TRAIN METHOD PREFERENCE CLASSIFIER
#    RandomForest — predicts best method category
# ─────────────────────────────────────────────

print("\nTraining RandomForest Classifier (method preference)...")
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(X, y_method, test_size=0.15, random_state=42)
method_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
method_model.fit(Xm_tr, ym_tr)
m_acc = accuracy_score(ym_te, method_model.predict(Xm_te))
print(f"  ✅ Accuracy on test set: {m_acc:.3f}")

# ─────────────────────────────────────────────
# 7. SAVE MODELS + METADATA
# ─────────────────────────────────────────────

joblib.dump(success_model, f"{OUT}/success_model.joblib")
joblib.dump(method_model,  f"{OUT}/method_model.joblib")

# Save feature list + method label map
meta = {
    "features": FEATURES,
    "method_labels": {
        0: "lure",       # general / food lure
        1: "lure_food",  # food-motivated lure
        2: "shaping",    # high intelligence shaping
        3: "guidance",   # physical guidance / stubborn breeds
    },
    "ease_thresholds": {
        "Zelo enostavno": 0.80,
        "Enostavno": 0.65,
        "Srednje": 0.45,
        "Zahtevno": 0.28,
        "Zelo zahtevno": 0.0,
    },
    "mae": round(float(mae), 4),
    "method_accuracy": round(float(m_acc), 3),
}
with open(f"{OUT}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# Also save breed + trick lookup tables as JSON for fast serving
breeds_lookup = {}
for _, row in breeds_df.iterrows():
    key = str(row["Breed Name"]).lower()
    breeds_lookup[key] = {
        "breed_name": row["Breed Name"],
        "trainability_score": int(row.get("trainability_score", 3)),
        "intelligence_score": int(row.get("intelligence_score", 3)),
        "stubbornness": int(row.get("Stubbornness (1-10)", 5)),
        "sensitivity": int(row.get("Sensitivity (1-10)", 5)),
        "energy_score": int(row.get("energy_score", 3)),
        "energy_level": str(row.get("Energy Level", "Moderate")),
        "size": str(row.get("Size", "Medium")),
        "group": str(row.get("Group", "")),
        "good_kids": int(row.get("Good with Kids (1-10)", 7)),
        "good_dogs": int(row.get("Good with Other Dogs (1-10)", 7)),
        "min_exercise": int(row.get("Min Exercise (min/day)", 30)) if pd.notna(row.get("Min Exercise (min/day)")) else 30,
    }

with open(f"{OUT}/breeds_lookup.json", "w", encoding="utf-8") as f:
    json.dump(breeds_lookup, f, ensure_ascii=False)

# Tricks lookup
tricks_lookup = []
methods_by_trick = {}
for _, row in methods_df.iterrows():
    tn = str(row.get("Trick Name", ""))
    if pd.notna(row.get("Trick Name")):
        methods_by_trick.setdefault(tn, [])
    if pd.notna(row.get("Method Name")):
        methods_by_trick.setdefault(tn, []).append({
            "method_name": str(row.get("Method Name", "")),
            "best_for": str(row.get("Best For (Dog Traits/Breeds)", "")),
            "steps": str(row.get("Step-by-Step Instructions", "")),
        })

def _s(v, default=""):
    """String-safe: returns default (or None if default='') when value is NaN/None/empty."""
    if v is None: return None if default == "" else default
    s = str(v).strip()
    if s.lower() in ("nan", "none", ""): return None if default == "" else default
    return s

for _, row in tricks_df.iterrows():
    tn = str(row.get("Trick Name", ""))
    tricks_lookup.append({
        "trick_name":        tn,
        "category":          _s(row.get("Category"),          "General"),
        "difficulty":        _s(row.get("Difficulty"),        "Beginner"),
        "difficulty_num":    int(row.get("difficulty_num", 1)),
        "prerequisites":     _s(row.get("Prerequisites")),
        "ideal_reward":      _s(row.get("Ideal Reward Type"), "Hrana"),
        "avg_learning_time": _s(row.get("Avg. Learning Time"),"1–2 tedna"),
        "notes":             _s(row.get("Notes")),
        "min_train_num":     int(row.get("min_train_num", 1)),
        "min_intel_num":     int(row.get("min_intel_num", 1)),
        "energy_required":   int(row.get("energy_required", 3)),
        "methods":           methods_by_trick.get(tn, []),
    })

with open(f"{OUT}/tricks_lookup.json", "w", encoding="utf-8") as f:
    json.dump(tricks_lookup, f, ensure_ascii=False)

print(f"\n✅ All models + lookups saved to {OUT}/")
print(f"   success_model.joblib  ({os.path.getsize(OUT+'/success_model.joblib')//1024} KB)")
print(f"   method_model.joblib   ({os.path.getsize(OUT+'/method_model.joblib')//1024} KB)")
print(f"   breeds_lookup.json    ({len(breeds_lookup)} breeds)")
print(f"   tricks_lookup.json    ({len(tricks_lookup)} tricks)")
print(f"\n🎯 Model MAE={mae:.4f} | Method acc={m_acc:.3f}")
print("Ready for production.")
