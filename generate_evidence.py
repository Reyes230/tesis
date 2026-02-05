import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "evidence_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo académico limpio
sns.set_theme(style="whitegrid")

def load_and_process(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ Archivo no encontrado: {filename}")
        # Intentamos buscar variantes comunes si el nombre exacto falla
        return None
    
    print(f"📖 Leyendo: {filename}...")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Aplanar columna sentiment
    if 'sentiment' in df.columns:
        sent_df = pd.json_normalize(df['sentiment'])
        sent_df.columns = [f"sent_{c}" for c in sent_df.columns]
        df = pd.concat([df.drop(columns=['sentiment']), sent_df], axis=1)
        
    return df

def generate_graphs(df, topic_name):
    if df is None or df.empty: return

    print(f"📊 Generando evidencia para: {topic_name}")
    safe_topic = topic_name.replace(" ", "_")

    # 1. GRÁFICO DE BARRAS (Conteo)
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='sent_label', data=df, palette='viridis', order=['positive', 'neutral', 'negative'])
    plt.title(f'Distribución de Sentimiento: {topic_name}')
    plt.xlabel('Clasificación')
    plt.ylabel('Frecuencia')
    for i in ax.containers: ax.bar_label(i,)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_topic}_distribucion.png"))
    plt.close()
    
    # 2. HISTOGRAMA DE CONFIANZA (Robustez)
    plt.figure(figsize=(8, 6))
    sns.histplot(df['sent_confidence'], bins=10, kde=True, color='skyblue')
    plt.title(f'Nivel de Confianza del Modelo: {topic_name}')
    plt.xlabel('Probabilidad (Score)')
    plt.axvline(0.8, color='red', linestyle='--', label='Alta Confianza (>0.8)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_topic}_confianza.png"))
    plt.close()

    # 3. EXCEL DE EVIDENCIA
    excel_path = os.path.join(OUTPUT_DIR, f"{safe_topic}_data.xlsx")
    cols = ['text_raw', 'lang', 'sent_label', 'sent_confidence', 'sent_source']
    valid_cols = [c for c in cols if c in df.columns]
    df[valid_cols].to_excel(excel_path, index=False)
    print(f"✅ Tabla de datos guardada: {excel_path}")

def main():
    print("🚀 GENERANDO EVIDENCIA CIENTÍFICA...")
    
    # IMPORTANTE: Estos nombres deben coincidir con tus archivos en data/preprocessed
    # Si te da error, verifica los nombres exactos en tu carpeta.
    file_ec = "Economía_en_Ecuador_es_cleaned_with_sentiment.jsonl"
    file_ap = "Apple_M5_Max_en_cleaned_with_sentiment.jsonl"

    generate_graphs(load_and_process(file_ec), "Economía Ecuador")
    generate_graphs(load_and_process(file_ap), "Apple M5 Max")
    
    print(f"\n🏁 LISTO. Revisa la carpeta: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()