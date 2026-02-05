import pandas as pd
import json
import os
import sys
import numpy as np

# Librerías para Coherencia (Gensim)
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

# Tu motor
from src.agents.trends.topic_engine import TopicModelEngine

def main():
    # ==========================================
    # 1. CARGAR DATOS + SENTIMIENTO
    # ==========================================
    data_dir = "data/preprocessed"
    
    # Nombre exacto de tu archivo verificado
    nombre_archivo = "ciberseguridad_bancos_cleaned_with_sentiment.jsonl" 

    INPUT_FILE = os.path.join(data_dir, nombre_archivo)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: No encuentro el archivo: {INPUT_FILE}")
        return

    print(f"📊 Auditando archivo ESPECÍFICO: {INPUT_FILE}")

    texts = []
    sentiment_scores = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                
                # 1. Obtener texto
                t = obj.get('text_norm', obj.get('text', ''))
                
                # 2. LÓGICA CIENTÍFICA DEFINITIVA (Weighted Polarity)
                raw_s = obj.get('sentiment', {})
                final_score = 0.0

                if isinstance(raw_s, dict):
                    # --- PRIORIDAD 1: RoBERTa con Probabilidades (Lo más preciso) ---
                    # Buscamos dentro de 'details' -> 'probs'
                    # Estructura: {"details": {"probs": {"positive": 0.01, "negative": 0.52}}}
                    details = raw_s.get('details', {})
                    probs = details.get('probs', {})
                    
                    if 'positive' in probs and 'negative' in probs:
                        # FÓRMULA CIENTÍFICA: Probabilidad Positiva - Probabilidad Negativa
                        # Resultado entre -1 (Muy negativo) y +1 (Muy positivo)
                        p_pos = float(probs.get('positive', 0.0))
                        p_neg = float(probs.get('negative', 0.0))
                        final_score = p_pos - p_neg
                        
                    # --- PRIORIDAD 2: RoBERTa simple (Solo label y confidence) ---
                    # (Respaldo si faltan las probs)
                    elif 'label' in raw_s and 'confidence' in raw_s:
                        label = raw_s['label']
                        conf = float(raw_s['confidence'])
                        if label == 'negative': final_score = -conf
                        elif label == 'positive': final_score = conf
                        else: final_score = 0.0
                    
                    # --- PRIORIDAD 3: VADER (Compound) ---
                    elif 'compound' in raw_s:
                        final_score = raw_s['compound']

                # --- PRIORIDAD 4: Número directo ---
                else:
                    try:
                        final_score = float(raw_s)
                    except:
                        final_score = 0.0
                
                # Solo guardamos si hay texto suficiente
                if len(t) > 20: 
                    texts.append(t)
                    sentiment_scores.append(final_score)

    print(f"✅ Documentos cargados: {len(texts)}")

    # ==========================================
    # 2. EJECUTAR TU MOTOR (BERTopic)
    # ==========================================
    engine = TopicModelEngine()
    topics, model = engine.fit_transform(texts)

    # ==========================================
    # 3. FASE 2: CÁLCULO DE COHERENCIA (C_v)
    # ==========================================
    print("\n🧠 Calculando Coherencia Semántica (Gensim)...")

    # A. Preprocesamiento simple
    tokens = [doc.split() for doc in texts]

    # B. Diccionario
    id2word = corpora.Dictionary(tokens)
    corpus = [id2word.doc2bow(text) for text in tokens]

    # C. Extraer palabras clave
    topic_words = []
    topic_info = model.get_topic_info()
    active_topics = [t for t in topic_info['Topic'] if t != -1]

    for topic_id in active_topics:
        words = [word[0] for word in model.get_topic(topic_id)]
        topic_words.append(words)

    # D. Calcular Score
    cv_score = 0.0
    if topic_words:
        coherence_model = CoherenceModel(
            topics=topic_words, 
            texts=tokens, 
            dictionary=id2word, 
            coherence='c_v',
            processes=1 # Importante para Windows
        )
        cv_score = coherence_model.get_coherence()

    print(f"📈 Score de Coherencia (C_v): {cv_score:.4f}")
    print("   (Referencia: 0.35+ es aceptable, 0.5+ es excelente)")

    # ==========================================
    # 4. FASE 4: TRIANGULACIÓN (TEMA vs SENTIMIENTO)
    # ==========================================
    print("\n❤️‍🔥 Analizando Causa Raíz (Tema -> Sentimiento)...")

    df_analisis = pd.DataFrame({
        'Topic': topics,
        'Sentiment': sentiment_scores,
        'Text': texts
    })

    # Agrupamos por Tópico
    df_resumen = df_analisis.groupby('Topic').agg({
        'Sentiment': 'mean',
        'Text': 'count'
    }).sort_values('Sentiment')

    print("\n--- CAUSA RAÍZ DE LA NEGATIVIDAD ---")
    print(f"{'TOPIC ID':<10} {'SENTIMIENTO PROMEDIO':<20} {'VOLUMEN':<10} {'NOMBRE DEL TEMA'}")
    print("-" * 70)

    for index, row in df_resumen.iterrows():
        topic_id = int(index)
        if topic_id == -1: continue 
        
        label = engine.get_topic_label(topic_id)
        sent_avg = row['Sentiment']
        count = int(row['Text'])
        
        # Umbrales para colores
        if sent_avg < -0.2:
            estado = "🔴 CRÍTICO" 
        elif sent_avg > 0.2:
            estado = "🟢 POSITIVO"
        else:
            estado = "⚪ NEUTRO "
        
        print(f"{topic_id:<10} {sent_avg:.4f} ({estado}) {count:<10} {label}")

    # ==========================================
    # 5. GUARDAR RESULTADOS
    # ==========================================
    with open("auditoria_resultados_metricas.txt", "w", encoding="utf-8") as f:
        f.write(f"Auditoria del Modelo\n")
        f.write(f"Coherencia C_v: {cv_score:.4f}\n")
        f.write(f"\nDesglose de Temas:\n")
        f.write(df_resumen.to_string())

    print("\n📝 Resultados guardados en 'auditoria_resultados_metricas.txt'")

if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except:
        pass
    main()