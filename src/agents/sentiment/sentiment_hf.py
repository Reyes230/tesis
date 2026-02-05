# src/agents/sentiment/sentiment_hf.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# ---------------------------------------------------------
# CONFIGURACIÓN DE MODELOS (Rutas Absolutas)
# ---------------------------------------------------------
# Calculamos la raíz del proyecto dinámicamente
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROBERTA_ES_PATH = os.path.join(BASE_DIR, "models", "robertuito-finetuned")

ROBERTA_EN = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_ES = ROBERTA_ES_PATH 

# ---------------------------------------------------------
# SETUP DEL DISPOSITIVO
# ---------------------------------------------------------
def _device():
    if torch.cuda.is_available():
        return 0
    elif torch.backends.mps.is_available():
        return "mps"
    return -1  # CPU

def _build_pipeline(model_name: str) -> TextClassificationPipeline:
    """Carga tokenizador y modelo en un pipeline de HF."""
    print(f"   🔄 [SentimentHF] Cargando: {os.path.basename(model_name)} ...")
    try:
        # Intentamos cargar (Local o Remoto)
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        return TextClassificationPipeline(
            model=mdl,
            tokenizer=tok,
            device=_device(),
            top_k=None  # Retorna todas las puntuaciones
        )
    except Exception as e:
        print(f"   ⚠️ Error cargando {model_name}: {e}")
        # Fallback inteligente
        if model_name == ROBERTA_ES:
            print("   ⚠️ Usando fallback remoto 'pysentimiento/robertuito-sentiment-analysis'.")
            fallback = "pysentimiento/robertuito-sentiment-analysis"
            tok = AutoTokenizer.from_pretrained(fallback, use_fast=False)
            mdl = AutoModelForSequenceClassification.from_pretrained(fallback)
            return TextClassificationPipeline(
                model=mdl, tokenizer=tok, device=_device(), top_k=None
            )
        raise e

# ---------------------------------------------------------
# SINGLETONS
# ---------------------------------------------------------
_PIPE_EN: TextClassificationPipeline | None = None
_PIPE_ES: TextClassificationPipeline | None = None

def _get_pipe_en() -> TextClassificationPipeline:
    global _PIPE_EN
    if _PIPE_EN is None:
        _PIPE_EN = _build_pipeline(ROBERTA_EN)
    return _PIPE_EN

def _get_pipe_es() -> TextClassificationPipeline:
    global _PIPE_ES
    if _PIPE_ES is None:
        _PIPE_ES = _build_pipeline(ROBERTA_ES)
    return _PIPE_ES

# ---------------------------------------------------------
# UTILIDADES DE NORMALIZACIÓN (Tu código excelente)
# ---------------------------------------------------------
def _to_probs(hf_output: List[Dict[str, Any]]) -> Dict[str, float]:
    if isinstance(hf_output, list) and len(hf_output) > 0 and isinstance(hf_output[0], list):
        hf_output = hf_output[0]

    mp = {x["label"].upper(): x["score"] for x in hf_output}
    
    neg = mp.get("NEGATIVE", mp.get("NEG", mp.get("LABEL_0", 0.0)))
    neu = mp.get("NEUTRAL",  mp.get("NEU", mp.get("LABEL_1", 0.0)))
    pos = mp.get("POSITIVE", mp.get("POS", mp.get("LABEL_2", 0.0)))

    # Fallback por posición si las etiquetas fallan
    if (neg + neu + pos) < 0.01:
        try:
            sorted_out = sorted(hf_output, key=lambda x: x['label'])
            if len(sorted_out) == 3:
                neg = sorted_out[0]['score']
                neu = sorted_out[1]['score']
                pos = sorted_out[2]['score']
        except: pass

    return {"negative": float(neg), "neutral": float(neu), "positive": float(pos)}

def _argmax_label(probs: Dict[str, float]) -> Tuple[str, float]:
    if not probs: return "neutral", 0.0
    lab = max(probs, key=probs.get)
    return lab, float(probs[lab])

# ---------------------------------------------------------
# FUNCIONES PÚBLICAS (API)
# ---------------------------------------------------------
def predict_english(text: str) -> Dict[str, Any]:
    pipe = _get_pipe_en()
    # Lógica segura de longitud (Tu gran aporte)
    max_len = getattr(pipe.model.config, 'max_position_embeddings', 512)
    safe_len = min(max_len, 512) - 2 
    
    out = pipe(text, truncation=True, max_length=safe_len)
    probs = _to_probs(out)
    label, conf = _argmax_label(probs)
    return {
        "model_used": "roberta_en", "lang_scope": "en",
        "label": label, "confidence": conf, "probs": probs
    }

def predict_spanish(text: str) -> Dict[str, Any]:
    pipe = _get_pipe_es()
    # Lógica segura de longitud para Robertuito (Vital)
    max_len = getattr(pipe.model.config, 'max_position_embeddings', 128)
    safe_len = max_len - 2 

    out = pipe(text, truncation=True, max_length=safe_len)
    probs = _to_probs(out)
    label, conf = _argmax_label(probs)
    return {
        "model_used": "roberta_es", "lang_scope": "es",
        "label": label, "confidence": conf, "probs": probs
    }