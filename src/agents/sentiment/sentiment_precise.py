# src/agents/sentiment/sentiment_precise.py
from __future__ import annotations
from typing import Dict, Any, Optional
# Importamos los "Drivers" que acabamos de arreglar
from src.agents.sentiment.sentiment_hf import predict_english, predict_spanish

class SentimentPrecise:
    """
    Router de Alto Nivel.
    Decide qué modelo usar y estandariza la salida.
    """

    def __init__(self, **kwargs):
        pass

    def analyze(self, text: str, lang_hint: Optional[str] = None) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"label": "neutral", "confidence": 0.0, "source": "empty", "details": {}}

        # 1. Normalización del idioma
        lang = (lang_hint or "es").lower().strip()
        
        # 2. Ruteo (Decision Making)
        try:
            if lang.startswith("en"):
                # Delegamos al driver de Inglés
                res = predict_english(text)
                source_tag = "model_en_specialist"
            else:
                # Delegamos al driver de Español (Default para Ecuador)
                res = predict_spanish(text)
                source_tag = "model_es_finetuned"
                
        except Exception as e:
            print(f"   ❌ Error en SentimentPrecise router: {e}")
            return {"label": "neutral", "confidence": 0.0, "source": "error", "details": {"err": str(e)}}

        # 3. Empaquetado final
        return {
            "label": self._normalize_label(res["label"]),
            "confidence": res["confidence"],
            "source": source_tag,
            "details": res  # Guardamos toda la metadata técnica (probs, modelo usado)
        }

    @staticmethod
    def _normalize_label(label: str) -> str:
        t = label.strip().lower()
        if t in ("positive", "pos", "label_2"): return "positive"
        if t in ("negative", "neg", "label_0"): return "negative"
        return "neutral"