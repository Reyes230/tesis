# src/agents/nodes.py
import os
import glob
import json
from langchain_core.messages import ToolMessage
from src.agents.state import AgentState
from src.utils.text_cleaning import basic_clean

# --- IMPORTACIONES DE TU LÓGICA AVANZADA (TESIS) ---
try:
    from src.agents.sentiment.chunker import chunk_text
    from src.agents.sentiment.sentiment_precise import SentimentPrecise
    from src.agents.sentiment.sentiment_aggregator import aggregate_post
    ADVANCED_MODE = True
    print("   🎓 MODO TESIS: Componentes avanzados (Chunker/Precise/Aggregator) cargados.")
except ImportError as e:
    ADVANCED_MODE = False
    print(f"   ⚠️ MODO MVP: No se encontraron módulos avanzados ({e}). Usando lógica simple.")
    # Aquí podrías dejar el código simple de respaldo o fallar.

# Instancia global del modelo (Singleton) para no recargar
_ANALYZER = None

def get_analyzer():
    global _ANALYZER
    if _ANALYZER is None and ADVANCED_MODE:
        # SentimentPrecise ya maneja la carga de modelos HF internamente
        _ANALYZER = SentimentPrecise()
    return _ANALYZER

# --- NODO DE LIMPIEZA (Se mantiene igual, robusto) ---
def cleaning_node(state: AgentState):
    print("\n--- 🧹 INICIANDO NODO DE LIMPIEZA (GRADO MILITAR) ---")
    ctx = dict(state.get("context", {}))
    messages = state.get("messages", [])
    
    # Rutas dinámicas
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PREPROC_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
    os.makedirs(PREPROC_DIR, exist_ok=True)

    input_path = None
    # 1. Buscar ruta en el historial del Agente A
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if "path" in data and os.path.exists(data["path"]):
                    input_path = data["path"]
                    break
            except: continue
    
    # 2. Fallback (Último archivo modificado)
    if not input_path:
        files = glob.glob(os.path.join(RAW_DIR, "*.jsonl"))
        if files: input_path = max(files, key=os.path.getmtime)
        else: return {"context": ctx}

    base_name = os.path.basename(input_path).replace(".jsonl", "")
    output_path = os.path.join(PREPROC_DIR, f"{base_name}_cleaned.jsonl")
    
    count = 0
    kept = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    # Obtenemos cualquier variante de texto que venga de Reddit
                    raw_text = obj.get("selftext") or obj.get("body") or obj.get("title") or obj.get("text") or ""
                    
                    clean_text = basic_clean(raw_text)
                    
                    # FILTRO DE CALIDAD:
                    # Descartamos mensajes vacíos o que quedaron vacíos tras limpiar (ej: solo era un link)
                    # También descartamos mensajes muy cortos (< 5 caracteres) que suelen ser ruido
                    if clean_text and len(clean_text) > 5:
                        clean_obj = {
                            "text_norm": clean_text,      # Texto limpio para análisis
                            "text_original": raw_text,   # Guardamos original por auditoría
                            "post_id": obj.get("id", "N/A"),
                            "lang": obj.get("lang", "es"),
                            "timestamp": obj.get("created_utc", ""),
                            "source_file": input_path
                        }
                        json.dump(clean_obj, fout)
                        fout.write('\n')
                        kept += 1
                    count += 1
                except: continue
        
        print(f"   ✨ Limpieza completada: {kept} documentos útiles (de {count} originales).")
        ctx["last_cleaned_path"] = output_path
    except Exception as e: 
        print(f"Error IO en limpieza: {e}")
    
    return {"context": ctx}


# --- NODO DE SENTIMIENTO ---
def sentiment_node(state: AgentState):
    print("\n--- 🧠 INICIANDO NODO DE SENTIMIENTO (Arquitectura Avanzada) ---")
    ctx = dict(state.get("context", {}))
    
    input_path = ctx.get("last_cleaned_path")
    if not input_path or not os.path.exists(input_path):
        print("   ❌ Error: Sin input.")
        return {"context": ctx}

    output_path = input_path.replace(".jsonl", "_with_sentiment.jsonl")
    
    analyzer = get_analyzer() # Carga tu SentimentPrecise
    if not analyzer:
        print("   ❌ Error: No se pudo iniciar el Analizador Preciso.")
        return {"context": ctx}

    processed_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                original_obj = json.loads(line)
                text = original_obj.get("text_norm", "")
                
                # PASO 1: CHUNKING (Divide y Vencerás)
                # Usamos tu script chunker.py para romper textos largos
                # max_tokens=300, overlap=50 (para no perder contexto en cortes)
                chunks_data = chunk_text(text, max_tokens=300, overlap=50)
                
                analyzed_chunks = []
                
                # PASO 2: INFERENCIA POR CHUNK (Map)
                for i, (chunk_txt, start, end) in enumerate(chunks_data):
                    # Usamos tu SentimentPrecise para predecir
                    # Le pasamos el hint del idioma del post original
                    analysis = analyzer.analyze(chunk_txt, lang_hint=original_obj.get("lang"))
                    
                    analyzed_chunks.append({
                        "chunk_index": i,
                        "text": chunk_txt,
                        "span_tokens": [start, end],
                        "sentiment": analysis,
                        "lang": original_obj.get("lang")
                    })

                # PASO 3: AGREGACIÓN (Reduce)
                # Usamos tu sentiment_aggregator.py para ponderar por longitud
                final_result = aggregate_post(analyzed_chunks)
                
                # PASO 4: ESTRUCTURAR SALIDA
                # Combinamos la metadata original con el resultado científico
                output_obj = {
                    **original_obj,
                    "sentiment": {
                        "label": final_result["label_final"],
                        "confidence": final_result["score_final"],
                        # Guardamos el chunk decisivo para trazabilidad (evidencia)
                        "decider_snippet": final_result.get("decider_chunk", {}).get("text_snippet")
                    },
                    "analysis_meta": {
                        "total_chunks": final_result["total_chunks"],
                        "method": "chunking_weighted_aggregation"
                    }
                }

                json.dump(output_obj, fout)
                fout.write('\n')
                processed_count += 1
                
                if processed_count % 10 == 0: print(f"   Processing {processed_count}...", end="\r")

        print(f"\n   ✅ Análisis Científico completado: {processed_count} documentos.")
        print(f"   💾 Guardado en: {os.path.basename(output_path)}")
        ctx["last_sentiment_path"] = output_path
        
    except Exception as e:
        print(f"   ❌ Error crítico en pipeline de sentimiento: {e}")
        import traceback
        traceback.print_exc()

    return {"context": ctx}