import pandas as pd
import json
import os

# Configura aquí el archivo que quieres revisar (el que te dio baja confianza)
# Puede ser el de Economía o el de Apple
FILE_NAME = "Economía_en_Ecuador_es_cleaned_with_sentiment.jsonl" 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed", FILE_NAME)

def analyze_low_confidence():
    if not os.path.exists(DATA_PATH):
        print(f"❌ No encuentro el archivo: {DATA_PATH}")
        return

    print(f"🔬 ANALIZANDO: {FILE_NAME}")
    
    data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                # Aplanamos estructura para facilitar lectura
                row = {
                    "text": obj.get("text_raw", "")[:100], # Solo primeros 100 caracteres
                    "label": obj["sentiment"]["label"],
                    "conf": obj["sentiment"]["confidence"],
                    "probs": obj["sentiment"].get("details", {}).get("probs", {})
                }
                data.append(row)
    
    df = pd.DataFrame(data)
    
    # 1. ESTADÍSTICAS GENERALES
    print("\n--- 📊 ESTADÍSTICAS DE CONFIANZA ---")
    print(f"Promedio Global: {df['conf'].mean():.4f}")
    print(f"Mínimo: {df['conf'].min():.4f}")
    print(f"Máximo: {df['conf'].max():.4f}")
    
    # 2. LOS "PEORES" CASOS (Baja Confianza)
    print("\n--- ⚠️ TOP 5 CASOS DE MENOR CONFIANZA (¿Por qué duda el modelo?) ---")
    low_conf = df.sort_values(by="conf").head(10)
    
    for i, row in low_conf.iterrows():
        print(f"\n🔸 Texto: '{row['text']}...'")
        print(f"   Decisión: {row['label'].upper()} ({row['conf']:.2f})")
        # Mostramos la "duda" interna del modelo
        probs = row['probs']
        if probs:
            print(f"   Duda Interna: Neg={probs.get('negative',0):.2f} | Neu={probs.get('neutral',0):.2f} | Pos={probs.get('positive',0):.2f}")
        else:
            print("   (Sin detalles de probabilidad)")

if __name__ == "__main__":
    analyze_low_confidence()