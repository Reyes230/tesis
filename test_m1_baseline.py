# test_m1_baseline.py
import json
import sys
import os

# Aseguramos que python encuentre el módulo src
sys.path.append(os.getcwd())

from datasets import load_dataset
from tqdm import tqdm

# Importa tu función m1_predict directamente
from src.agents.sentiment.sentiment_hf import m1_predict
# Importa tus métricas (asegúrate de que sentiment_eval.py tenga estas funciones)
from src.agents.sentiment.sentiment_eval import accuracy, macro_f1, mcc_multiclass

# --- Configuración del Test ---
DATASET_ID = "cardiffnlp/tweet_sentiment_multilingual"
SUBSET = "spanish"
SPLIT = "test"
MAX_SAMPLES = 1000 
# ------------------------------

print(f"Evaluando M1 ('{DATASET_ID}') en subset '{SUBSET}' (split '{SPLIT}')")

# Mapeo de etiquetas del dataset
mapping = {0: "negative", 1: "neutral", 2: "positive"}

# Cargar dataset
# Nota: Si te vuelve a dar error de 'trust_remote_code', usa el fix de datasets==2.16.1
# o simplemente confía en el código si ya lo arreglaste.
try:
    ds = load_dataset(DATASET_ID, SUBSET, split=SPLIT, trust_remote_code=True)
except Exception as e:
    print(f"Error cargando dataset: {e}")
    print("Intenta: pip install datasets==2.16.1")
    sys.exit(1)

if MAX_SAMPLES is not None and len(ds) > MAX_SAMPLES:
    ds = ds.shuffle(seed=42).select(range(MAX_SAMPLES))

y_true = []
y_pred = []

print(f"Procesando {len(ds)} ejemplos con M1 puro...")

for ex in tqdm(ds):
    text = ex["text"]
    raw_label = ex["label"]
    
    # Obtener etiqueta real
    gold = mapping[raw_label]
    
    # --- ¡Llamada directa a M1! ---
    pred_data = m1_predict(text)
    pred_label = pred_data["label"]
    # -----------------------------

    y_true.append(gold)
    y_pred.append(pred_label)

# Calcular métricas
try:
    acc = accuracy(y_true, y_pred)
    f1 = macro_f1(y_true, y_pred)
    mcc = mcc_multiclass(y_true, y_pred)
except NameError:
    # Por si acaso las funciones no se llaman igual en tu sentiment_eval
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)

print("\n--- Resultados (Solo M1) ---")
results = {
    "model": "M1 (twitter-xlm-roberta-base-sentiment)",
    "dataset": DATASET_ID,
    "subset": SUBSET,
    "n": len(y_true),
    "accuracy": acc,
    "macro_f1": f1,
    "mcc": mcc
}
print(json.dumps(results, indent=2))