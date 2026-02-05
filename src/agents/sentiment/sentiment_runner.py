# src/agents/sentiment/sentiment_runner.py
from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from .sentiment_precise import SentimentPrecise


def _open_jsonl(path: str):
    """Generador que lee un JSONL y devuelve un dict por línea válida."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _ensure_dir(p: str):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)


def analyze_chunk(chunk: Dict[str, Any], sp: SentimentPrecise) -> Dict[str, Any]:
    """
    Aplica la cascada de sentimiento a un chunk.
    """
    text = (
        chunk.get("text_norm") or
        chunk.get("text")
    )       
    # Intentamos sacar el idioma del chunk, si no existe, pasamos None
    lang = chunk.get("lang") or (chunk.get("meta") or {}).get("lang")

    sentiment_res = sp.analyze(text, lang_hint=lang)

    # Devolvemos el chunk completo + resultado de sentimiento
    out = {
        **chunk,
        "sentiment": sentiment_res,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Runner de análisis de sentimiento.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--tau1", type=float, default=None) # Mantenido por compatibilidad
    ap.add_argument("--tau2", type=float, default=None) # Mantenido por compatibilidad
    ap.add_argument("--batch_log", type=int, default=200)
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        print(f"Error: no existe {args.inp}", file=sys.stderr)
        sys.exit(1)

    _ensure_dir(args.out)

    sp = SentimentPrecise() # Ya no necesita taus, es determinista

    total = 0
    written = 0
    
    print(f"🚀 Iniciando Runner desde CLI. Leyendo: {args.inp}")

    with open(args.out, "w", encoding="utf-8") as w:
        for chunk in _open_jsonl(args.inp):
            total += 1
            try:
                rec = analyze_chunk(chunk, sp)
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                print(f"\n❌ ERROR en chunk {chunk.get('chunk_id')}: {e}")
                traceback.print_exc()
                
                # Registrar error en archivo
                w.write(json.dumps({
                    "chunk_id": chunk.get("chunk_id"), 
                    "error": str(e)
                }, ensure_ascii=False) + "\n")

            if written % max(1, args.batch_log) == 0:
                print(f"… escritos {written} / leídos {total}")

    print(f"✅ Finalizado CLI → {args.out}")


# ---------------------------------------------------------------------
# ESTA ES LA FUNCIÓN QUE FALTABA Y QUE PIDE EL TEST_PIPELINE.PY
# ---------------------------------------------------------------------
def run_sentiment_pipeline(
    inp: str,
    out: str,
    tau1: float | None = None,
    tau2: float | None = None,
    batch_log: int = 200,
) -> Dict[str, int]:
    """
    Wrapper programático del runner, para usarlo desde tests o scripts.
    """
    if not os.path.exists(inp):
        raise FileNotFoundError(f"No existe el archivo de chunks: {inp}")

    _ensure_dir(out)
    
    # Inicializamos la clase 'SentimentPrecise'
    sp = SentimentPrecise()

    total = 0
    written = 0

    with open(out, "w", encoding="utf-8") as w:
        for chunk in _open_jsonl(inp):
            total += 1
            try:
                rec = analyze_chunk(chunk, sp)
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                # --- DEBUG PRINT ---
                print(f"\n❌ ERROR CRÍTICO durante el test en chunk {total}:")
                print(f"   Mensaje: {e}")
                traceback.print_exc()
                # -------------------
                
                w.write(json.dumps({
                    "chunk_id": chunk.get("chunk_id"),
                    "error": f"{type(e).__name__}: {e}"
                }, ensure_ascii=False) + "\n")
            
            if written % max(1, batch_log) == 0:
                print(f"… escritos {written} / leídos {total}")

    return {"read": total, "written": written}

if __name__ == "__main__":
    main()