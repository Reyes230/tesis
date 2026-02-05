# src/agents/sentiment/sentiment_eval_hf.py
from __future__ import annotations
import argparse
import json
from typing import List, Dict, Any

from datasets import load_dataset

from .sentiment_precise import SentimentPrecise
from .sentiment_eval import accuracy, macro_f1, mcc_multiclass, LABELS


def _eval_dataset(
    dataset_id: str,
    subset: str | None,
    split: str,
    text_field: str,
    label_field: str,
    label_mapping: Dict[Any, str],
    lang_hint: str | None = None,
    max_samples: int | None = 2000,
) -> Dict[str, Any]:
    """
    Evalúa SentimentPrecise sobre un dataset de HF.

    - dataset_id: p.ej. "cardiffnlp/tweet_eval"
    - subset:     p.ej. "sentiment" o "spanish" (o None si no hay)
    - split:      "train" | "validation" | "test"
    - text_field: nombre del campo de texto
    - label_field:nombre del campo de label
    - label_mapping: dict que mapea valor bruto → "negative|neutral|positive"
    - lang_hint: "en" / "es" (opcional)
    - max_samples: para no morir evaluando 200k ejemplos
    """

    if subset:
        ds = load_dataset(dataset_id, subset, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(dataset_id, split=split, trust_remote_code=True)

    if max_samples is not None and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    sp = SentimentPrecise()

    y_true: List[str] = []
    y_pred: List[str] = []

    for ex in ds:
        text = ex[text_field]
        raw_label = ex[label_field]

        # Mapear a nuestras 3 clases
        gold = label_mapping[raw_label]
        if gold not in LABELS:
            # por seguridad, saltar ejemplos raros
            continue

        pred = sp.analyze(text, lang_hint=lang_hint)
        y_true.append(gold)
        y_pred.append(pred["label"])

    return {
        "n": len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "mcc": mcc_multiclass(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser(description="Evalua SentimentPrecise sobre datasets de HuggingFace")
    ap.add_argument("--dataset", required=True, help="ID del dataset en HF (p.ej. cardiffnlp/tweet_eval)")
    ap.add_argument("--subset", default=None, help="Config/subset (p.ej. sentiment, spanish, default)")
    ap.add_argument("--split", default="test", help="Split: train/validation/test")
    ap.add_argument("--max_samples", type=int, default=2000, help="Máx. ejemplos a evaluar")
    args = ap.parse_args()

    # Ejemplos de routing según dataset_id
    if args.dataset == "cardiffnlp/tweet_eval":
        # Inglés, tweets
        mapping = {0: "negative", 1: "neutral", 2: "positive"}
        res = _eval_dataset(
            dataset_id="cardiffnlp/tweet_eval",
            subset=args.subset or "sentiment",
            split=args.split,
            text_field="text",
            label_field="label",
            label_mapping=mapping,
            lang_hint="en",
            max_samples=args.max_samples,
        )

    elif args.dataset == "cardiffnlp/tweet_sentiment_multilingual":
        # Español, tweets
        # subset debe ser "spanish"
        mapping = {0: "negative", 1: "neutral", 2: "positive"}
        res = _eval_dataset(
            dataset_id="cardiffnlp/tweet_sentiment_multilingual",
            subset=args.subset or "spanish",
            split=args.split,
            text_field="text",
            label_field="label",
            label_mapping=mapping,
            lang_hint="es",
            max_samples=args.max_samples,
        )

    elif args.dataset == "SetFit/amazon_reviews_multi_en":
        # Reseñas largas en inglés
        # label: 0..4   (usa mapping por estrellas)
        def map_star_label(raw: int) -> str:
            # 0..4 → 1..5 estrellas (ver card del dataset)
            # 0 -> 1 star, 4 -> 5 stars
            if raw in (0, 1):
                return "negative"
            elif raw == 2:
                return "neutral"
            else:  # 3, 4
                return "positive"

        mapping = {i: map_star_label(i) for i in range(5)}

        res = _eval_dataset(
            dataset_id="SetFit/amazon_reviews_multi_en",
            subset=args.subset or "default",
            split=args.split,
            text_field="text",
            label_field="label",
            label_mapping=mapping,
            lang_hint="en",
            max_samples=args.max_samples,
        )

    else:
        raise ValueError(f"No tengo configurado el dataset {args.dataset} aún 🤷‍♀️")

    print(json.dumps({"dataset": args.dataset, "subset": args.subset, **res}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
