# src/agents/sentiment/sentiment_report.py
from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from typing import Dict, Any, List

# ---------- Helpers de lectura ----------

def read_posts_jsonl(path: str) -> List[Dict[str, Any]]:
    """Lee reddit_sentiment_posts.jsonl y devuelve lista de dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

# ---------- Construcción de KPIs ----------

def build_report(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Construye KPIs agregados a partir de posts ya agregados.
    Espera formato:
      {
        "post_id": str,
        "label_final": "positive|neutral|negative",
        "score_final": float,
        "chunks": int,
        "valid_chunks": int,
        "lang_counts": {lang: count},
        "route_counts": {source: count},   # p.ej. m1_high_conf, m1_m2_consensus, gpt_fallback
        "decider": {...}
      }
    """
    total_posts = len(posts)

    # Distribución de polaridad
    labels = [p.get("label_final", "neutral") for p in posts]
    label_distribution = dict(Counter(labels))

    # Confianza promedio sobre los posts
    scores = [float(p.get("score_final", 0.0)) for p in posts]
    avg_confidence = (sum(scores) / len(scores)) if scores else 0.0

    # Mezcla de lenguajes (sumando lang_counts de cada post)
    lang_counter = Counter()
    for p in posts:
        lc = p.get("lang_counts") or {}
        for lang, cnt in lc.items():
            if lang:
                lang_counter[lang] += int(cnt)
    lang_counts = dict(lang_counter)

    # Distribución global de fuentes de decisión (M1, M2, GPT-5)
    source_counter = Counter()
    for p in posts:
        rc = p.get("route_counts") or {}
        for src, cnt in rc.items():
            if src:
                source_counter[src] += int(cnt)
    source_counts = dict(source_counter)

    # Top N posts más "fuertes" por score_final (los más definidos)
    top_abs_posts = sorted(
        posts,
        key=lambda r: abs(float(r.get("score_final", 0.0))),
        reverse=True
    )[:50]

    report = {
        "meta": {
            "description": "KPIs agregados de sentimiento por post (pipeline M1+M2+GPT-5)",
            "source": "reddit_sentiment_posts.jsonl",
        },
        "kpis": {
            "total_posts": total_posts,
            "label_distribution": label_distribution,
            "avg_confidence": avg_confidence,
            "lang_counts": lang_counts,
            "source_counts": source_counts,  # ← NUEVO
        },
        "data": {
            # lista compacta de posts para tablas / drill-down
            "posts_overview": [
                {
                    "post_id": p.get("post_id"),
                    "label_final": p.get("label_final"),
                    "score_final": float(p.get("score_final", 0.0)),
                    "chunks": p.get("chunks"),
                    "valid_chunks": p.get("valid_chunks"),
                }
                for p in posts
            ],
            "top_abs": [
                {
                    "post_id": r.get("post_id"),
                    "label_final": r.get("label_final"),
                    "score_final": float(r.get("score_final", 0.0)),
                    "chunks": r.get("chunks"),
                    "valid_chunks": r.get("valid_chunks"),
                }
                for r in top_abs_posts
            ],
        },
    }
    return report

# ---------- Exports opcionales a disco (para debug / offline) ----------

def export_report_files(report: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # JSON maestro
    with open(os.path.join(out_dir, "sentiment_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Distribución de labels
    with open(os.path.join(out_dir, "label_distribution.csv"), "w", encoding="utf-8") as f:
        f.write("label,count\n")
        for lbl, cnt in report["kpis"]["label_distribution"].items():
            f.write(f"{lbl},{cnt}\n")

    # Lenguajes
    with open(os.path.join(out_dir, "lang_counts.csv"), "w", encoding="utf-8") as f:
        f.write("lang,count\n")
        for lang, cnt in report["kpis"]["lang_counts"].items():
            f.write(f"{lang},{cnt}\n")

    # Fuentes de decisión (M1, M2, GPT-5)
    with open(os.path.join(out_dir, "source_counts.csv"), "w", encoding="utf-8") as f:
        f.write("source,count\n")
        for src, cnt in report["kpis"]["source_counts"].items():
            f.write(f"{src},{cnt}\n")

def main():
    ap = argparse.ArgumentParser(description="Genera KPIs de sentimiento a partir de reddit_sentiment_posts.jsonl")
    ap.add_argument("--posts", required=True, help="Path a reddit_sentiment_posts.jsonl")
    ap.add_argument("--out_dir", required=True, help="Directorio de salida para archivos de reporte")
    args = ap.parse_args()

    posts = read_posts_jsonl(args.posts)
    report = build_report(posts)
    export_report_files(report, args.out_dir)
    print(f"✅ Report generado en {args.out_dir}")

if __name__ == "__main__":
    main()
