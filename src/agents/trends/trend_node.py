# src/agents/trends/trend_node.py
import json
import os
import pandas as pd
from src.agents.trends.topic_engine import TopicModelEngine
from src.agents.trends.trend_math import TrendMathEngine 

def trend_node(state):
    print("\n--- 📈 EJECUTANDO NODO DE TENDENCIAS (Robust Analysis) ---")
    
    # 1. Contexto
    ctx = state.get("context", {})
    input_path = ctx.get("last_sentiment_path") 
    
    if not input_path:
        input_path = state.get("current_file_path")

    if not input_path or not os.path.exists(input_path):
        print("   ❌ Error: No se encontró archivo de entrada.")
        return {"context": ctx}

    print(f"   📄 Leyendo datos desde: {os.path.basename(input_path)}")

    # 2. Cargar datos
    data = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except: continue
    except Exception as e:
        print(f"   ❌ Error leyendo archivo: {e}")
        return {"context": ctx}
    
    if not data: 
        print("   ⚠️ Archivo vacío. Saltando.")
        return {"context": ctx}

    df = pd.DataFrame(data)
    # Normalización de texto
    df['final_text'] = df.get('text_norm', df.get('text_raw', df.get('text', '')))
    
    # Lógica Científica de Sentimiento (TU LÓGICA ORIGINAL)
    def get_scientific_sentiment(row):
        sent = row.get('sentiment', {})
        if not sent: return 0.0
        if isinstance(sent, dict):
            # Probs
            details = sent.get('details', {})
            probs = details.get('probs', {})
            if 'positive' in probs and 'negative' in probs:
                return float(probs.get('positive', 0.0)) - float(probs.get('negative', 0.0))
            # Label/Conf
            label = sent.get('label', 'neutral')
            conf = float(sent.get('confidence', 0.0))
            if label == 'positive': return conf
            if label == 'negative': return -conf
            # VADER
            if 'compound' in sent: return float(sent['compound'])
        else:
            try: return float(sent)
            except: return 0.0
        return 0.0

    df['numeric_sentiment'] = df.apply(get_scientific_sentiment, axis=1)

    # ---------------------------------------------------------
    # 🛡️ VÁLVULA DE SEGURIDAD Y BYPASS DE MEMORIA
    # ---------------------------------------------------------
    raw_report = []
    total_docs = len(df)
    
    # Si hay menos de 20 docs, BERTopic/UMAP fallan matemáticamente o no valen la pena (gasto RAM)
    USE_BERTOPIC = True
    if total_docs < 20:
        print(f"   🚀 MODO LIGERO ACTIVADO ({total_docs} docs). Saltando Clustering para ahorrar recursos.")
        USE_BERTOPIC = False

    if USE_BERTOPIC:
        try:
            # 3. BERTopic (Modo Full)
            print(f"   🦾 Ejecutando BERTopic en {total_docs} documentos...")
            engine = TopicModelEngine()
            topics, _ = engine.fit_transform(df['final_text'].tolist())
            df['topic_id'] = topics
            
            # 4. Agregación (Solo si funcionó BERTopic)
            unique_topics = sorted(list(set(topics)))
            
            for tid in unique_topics:
                if tid == -1: continue 
                
                sub_df = df[df['topic_id'] == tid]
                if sub_df.empty: continue

                count = len(sub_df)
                avg_sent = sub_df['numeric_sentiment'].mean()
                label = engine.get_topic_label(tid)
                
                # Pasamos los primeros 5 textos como lista para que el Agente SR lea variedad
                examples = sub_df['final_text'].head(5).tolist()
                
                status = "⚪ NEUTRO"
                if avg_sent > 0.15: status = "🟢 POSITIVO"
                if avg_sent < -0.15: status = "🔴 NEGATIVO"

                raw_report.append({
                    "topic_id": int(tid),
                    "label": label,
                    "volume": int(count),
                    "sentiment_avg": float(round(avg_sent, 4)),
                    "status": status,
                    "example_text": examples # Lista de textos reales
                })
                
        except Exception as e:
            print(f"   ⚠️ ERROR EN CLUSTERING (Probablemente Memoria RAM): {e}")
            print("   ⚠️ Activando Fallback Manual...")
            USE_BERTOPIC = False # Forzamos el modo manual abajo

    # LÓGICA DE FALLBACK / MODO LIGERO (Si falló BERTopic o eran pocos datos)
    if not USE_BERTOPIC or not raw_report:
        avg_sent = df['numeric_sentiment'].mean()
        status = "⚪ NEUTRO"
        if avg_sent > 0.15: status = "🟢 POSITIVO"
        if avg_sent < -0.15: status = "🔴 NEGATIVO"
        
        raw_report.append({
            "topic_id": 1,
            "label": "discusión_general_baja_muestra",
            "volume": int(total_docs),
            "sentiment_avg": float(round(avg_sent, 4)),
            "status": status,
            "example_text": df['final_text'].head(10).tolist() # Pasamos hasta 10 ejemplos
        })

    # 5. MATEMÁTICA DE IMPACTO
    # Solo llamamos si tenemos algo en raw_report
    if raw_report:
        try:
            final_report = TrendMathEngine.calculate_impact(raw_report)
        except:
            final_report = raw_report # Fallback simple
    else:
        final_report = []

    # 6. Guardar (CON RUTA ABSOLUTA SEGURA)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "reports")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_filename = os.path.basename(input_path).replace(".jsonl", "_trends_report.json")
    output_filename = output_filename.replace("_cleaned_with_sentiment_trends_report", "_trends_report")
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Análisis guardado en: {output_path}")
    
    # Debug visual
    top_topic = final_report[0]['label'] if final_report else 'N/A'
    print(f"   📊 Top Tema: {top_topic}")

    # ACTUALIZACIÓN DE CONTEXTO
    new_ctx = ctx.copy()
    new_ctx["last_trends_path"] = output_path 
    
    return {
        "context": new_ctx,
        "messages": [f"Análisis de tendencias completado. Archivo: {output_filename}"]
    }