# train_sentiment.py
import os
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# --- CONFIGURACIÓN ---
MODEL_ID = "pysentimiento/robertuito-sentiment-analysis"
DATASET_ID = "cardiffnlp/tweet_sentiment_multilingual"
SUBSET = "spanish"
OUTPUT_DIR = "./models/robertuito-finetuned" # Donde se guardará tu nuevo modelo
# ---------------------

def main():
    print(f"🚀 Iniciando Fine-Tuning de {MODEL_ID} en {SUBSET}...")

    # 1. Cargar Dataset
    # Usamos 'train' para entrenar y 'test' para validar
    print("📥 Cargando dataset...")
    dataset = load_dataset(DATASET_ID, SUBSET, trust_remote_code=True)
    
    # El dataset tiene 'train', 'validation', 'test'. Usaremos train y validation.
    train_ds = dataset["train"]
    eval_ds = dataset["validation"] # O usamos 'test' si validation es muy pequeño

    # 2. Tokenizador
    print("📚 Cargando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    print("⚙️ Tokenizando datos (esto puede tardar un poco)...")
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_eval = eval_ds.map(tokenize_function, batched=True)

    # 3. Modelo
    # Mapeo: 0: negative, 1: neutral, 2: positive
    print("🧠 Cargando modelo base...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, 
        num_labels=3
    )

    # 4. Métricas
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        return {
            "accuracy": acc["accuracy"],
            "macro_f1": f1["f1"],
        }

    # 5. Configuración del Entrenamiento (Hyperparámetros)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,           # Tasa de aprendizaje baja para no romper el modelo
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,           # 3 pasadas completas por los datos
        weight_decay=0.01,
        eval_strategy="epoch",  # Evaluar al final de cada época
        save_strategy="epoch",        # Guardar checkpoint al final de cada época
        load_best_model_at_end=True,  # Al final, quedarse con el mejor checkpoint
        metric_for_best_model="macro_f1",
        push_to_hub=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. ¡Entrenar!
    print("\n🔥 ¡EMPEZANDO ENTRENAMIENTO! 🔥")
    print("Ve por un café, esto tomará unos minutos (o más si no tienes GPU)...")
    trainer.train()

    # 7. Evaluar resultado final
    print("\n📊 Evaluando modelo final...")
    metrics = trainer.evaluate()
    print(metrics)

    # 8. Guardar modelo final
    print(f"\n💾 Guardando modelo final en: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ ¡Listo! Modelo guardado.")

if __name__ == "__main__":
    main()