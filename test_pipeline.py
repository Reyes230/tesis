import os
import json
import shutil
from src.agents.sentiment.chunker import chunk_file
from src.agents.sentiment.sentiment_runner import run_sentiment_pipeline
from src.agents.sentiment.sentiment_aggregator import run_aggregator

# --- CONFIGURACIÓN DE RUTAS TEMPORALES ---
TEST_DIR = "./test_data_temp"
RAW_FILE = f"{TEST_DIR}/1_raw_posts.jsonl"
CHUNKS_FILE = f"{TEST_DIR}/2_chunks.jsonl"
SENTIMENT_FILE = f"{TEST_DIR}/3_sentiment.jsonl"
FINAL_FILE = f"{TEST_DIR}/4_final_report.jsonl"

def setup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)

def create_dummy_data():
    """Crea 3 posts sintéticos: Español, Inglés y Mixto."""
    print("📝 Creando datos de prueba...")
    data = [
        {
            "post_id": "post_ES_001",
            "text_norm": "Este gobierno es un desastre total, la economía se cae a pedazos. No hay esperanza.",
            "lang": "es",
            "timestamp": "2025-11-21T10:00:00",
            "channel": "twitter",
            "meta": {"url": "http://test.com/1"},
            "content_sha256": "hash1"
        },
        {
            "post_id": "post_EN_002",
            "text_norm": "I absolutely love the new parks in Quito. It feels so safe and green now! Best management ever.",
            "lang": "en",
            "timestamp": "2025-11-21T11:00:00",
            "channel": "reddit",
            "meta": {"url": "http://test.com/2"},
            "content_sha256": "hash2"
        },
        {
            "post_id": "post_MIX_003",
            "text_norm": "El servicio fue regular. Not bad but not good either. Esperaba más.",
            "lang": "es", # Forzamos 'es' para ver si el router obedece al hint o si entra al default
            "timestamp": "2025-11-21T12:00:00",
            "channel": "facebook",
            "meta": {"url": "http://test.com/3"},
            "content_sha256": "hash3"
        }
    ]
    
    with open(RAW_FILE, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def run_tests():
    # 1. CHUNKING
    print("\n🔹 1. Ejecutando CHUNKER...")
    chunk_stats = chunk_file(RAW_FILE, CHUNKS_FILE)
    print(f"   Items: {chunk_stats['items']} -> Chunks generados: {chunk_stats['chunks']}")

    # 2. SENTIMENT RUNNER (Aquí se cargan los modelos)
    print("\n🔹 2. Ejecutando RUNNER (Esto puede tardar cargando modelos)...")
    # Nota: No pasamos umbrales porque ahora es determinista por idioma
    runner_stats = run_sentiment_pipeline(CHUNKS_FILE, SENTIMENT_FILE)
    print(f"   Leídos: {runner_stats['read']} -> Escritos: {runner_stats['written']}")

    # 3. AGGREGATOR
    print("\n🔹 3. Ejecutando AGGREGATOR...")
    run_aggregator(SENTIMENT_FILE, FINAL_FILE)
    print("   Agregación lista.")

def verify_results():
    print("\n🔎 VERIFICACIÓN DE RESULTADOS:")
    print("-" * 60)
    
    with open(FINAL_FILE, "r", encoding="utf-8") as f:
        posts = [json.loads(line) for line in f]

    for p in posts:
        pid = p["post_id"]
        label = p["label_final"]
        score = p["score_final"]
        text_full = p.get("text_full", "❌ NO ENCONTRADO")
        sources = p.get("route_counts", {})
        
        print(f"📌 Post ID: {pid}")
        print(f"   Texto recuperado: '{text_full}'")
        print(f"   Sentimiento: {label} ({score:.4f})")
        print(f"   Fuentes usadas: {sources}")
        
        # Validaciones específicas
        if pid == "post_ES_001":
            if "negative" not in label: print("   ⚠️ ALERTA: Se esperaba NEGATIVE")
            if "model_es_finetuned" not in str(sources): print("   ⚠️ ALERTA: No usó el modelo ESPAÑOL")
        
        elif pid == "post_EN_002":
            if "positive" not in label: print("   ⚠️ ALERTA: Se esperaba POSITIVE")
            if "model_en_specialist" not in str(sources): print("   ⚠️ ALERTA: No usó el modelo INGLÉS")
            
        print("-" * 60)

if __name__ == "__main__":
    setup()
    create_dummy_data()
    run_tests()
    verify_results()