# audit_model.py
import pandas as pd
import json
import os
import sys
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Importamos tu motor actual
from src.agents.trends.topic_engine import TopicModelEngine

# 1. CARGAR DATOS EXISTENTES
# Cambia esta ruta por un archivo .jsonl que ya tengas con datos reales
INPUT_FILE = "data/preprocessed/Impacto_del_Teletrabajo_en_Salud_Mental_cleaned_with_sentiment.jsonl" 

if not os.path.exists(INPUT_FILE):
    # Si no tienes el archivo exacto, busca cualquiera en la carpeta
    files = [f for f in os.listdir("data/preprocessed") if f.endswith(".jsonl")]
    if files:
        INPUT_FILE = os.path.join("data/preprocessed", files[0])
    else:
        print("❌ No encontré archivos de datos en data/preprocessed. Ejecuta run_batch primero.")
        sys.exit()

print(f"📊 Auditando modelo con datos de: {INPUT_FILE}")

# Cargar textos
texts = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            # Intentamos leer el texto limpio o el raw
            t = obj.get('text_norm', obj.get('text', ''))
            if len(t) > 20: # Solo textos con algo de contenido
                texts.append(t)

print(f"   Documentos cargados: {len(texts)}")

# 2. EJECUTAR EL MOTOR (Como está configurado actualmente)
engine = TopicModelEngine()
topics, model = engine.fit_transform(texts)
embeddings = model._extract_embeddings(texts, method="document", verbose=False)

# 3. VER LAS "TRIPAS" DEL MODELO (Lo que quiere tu tutor)

print("\n--- 🔍 INSPECCIÓN DE TÓPICOS (Stopwords Check) ---")
topic_info = model.get_topic_info()
print(topic_info[['Topic', 'Count', 'Name']].head(10))

print("\n--- 🧐 PALABRAS CLAVE DEL TOPIC 0 (El más grande) ---")
# Aquí veremos si hay basura como "el", "la", "que"
print(model.get_topic(0))

# 4. CÁLCULO DE MÉTRICAS CIENTÍFICAS (Rigor Académico)
# Filtramos el ruido (-1) para calcular métricas justas
clean_indices = [i for i, t in enumerate(topics) if t != -1]

if len(clean_indices) > 0 and len(set(topics)) > 1:
    clean_embeddings = embeddings[clean_indices]
    clean_topics = [topics[i] for i in clean_indices]
    
    # Silhouette Score: (-1 a 1). Cuanto más alto, mejor definidos están los grupos.
    sil_score = silhouette_score(clean_embeddings, clean_topics)
    
    # Davies-Bouldin: (0 a infinito). Cuanto más BAJO, mejor.
    db_score = davies_bouldin_score(clean_embeddings, clean_topics)
    
    print(f"\n--- 📐 MÉTRICAS DE CALIDAD ---")
    print(f"✅ Silhouette Score: {sil_score:.4f} (Ideal > 0.1 para texto, >0.4 es excelente)")
    print(f"✅ Davies-Bouldin Index: {db_score:.4f} (Cuanto más bajo mejor)")
else:
    print("\n⚠️ No hay suficientes clusters para calcular métricas (todo es ruido o un solo grupo).")

# 5. GENERACIÓN DE GRÁFICOS (Para tu tesis)
print("\n--- 🖼️ GENERANDO GRÁFICOS INTERACTIVOS ---")
output_dir = "auditoria_graficos"
os.makedirs(output_dir, exist_ok=True)

try:
    # Gráfico de Barras (Palabras clave)
    fig1 = model.visualize_barchart(top_n_topics=5)
    fig1.write_html(f"{output_dir}/barchart_words.html")
    print(f"   --> {output_dir}/barchart_words.html")

    # Mapa de Distancia (Cómo se separan los temas)
    fig2 = model.visualize_topics()
    fig2.write_html(f"{output_dir}/intertopic_map.html")
    print(f"   --> {output_dir}/intertopic_map.html")
    
except Exception as e:
    print(f"Error generando gráficos: {e}")

print("\n🏁 Auditoría finalizada. Abre los HTML en tu navegador.")