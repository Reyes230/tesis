# C:/Users/Matias/Documents/tesis/src/agents/trends/test_pipeline.py
import os
import json
import shutil
from trend_node import TrendDetectionAgent

# ==========================================
# GENERADOR DE DATOS FALSOS (MOCK DATA)
# ==========================================
def crear_jsonl_prueba(filename, topics_dict):
    """
    Genera un archivo .jsonl repitiendo frases para simular volumen.
    topics_dict: {'Tema': cantidad_de_posts}
    """
    data = []
    # Frases base para cada tema simulado
    frases = {
        "python": "I love Python programming for data science and agents.",
        "futbol": "The match yesterday was amazing, great goal by Messi.",
        "crypto": "Bitcoin is going to the moon, huge pump incoming!",
        "aliens": "UFO sighting confirmed in New York! The aliens are here!"
    }

    print(f"Generando {filename}...")
    for tema, cantidad in topics_dict.items():
        base_text = frases.get(tema, "Random text content.")
        for i in range(cantidad):
            # Añadimos un ID único y un poco de variación
            entry = {
                "id": f"{tema}_{i}",
                "text_norm": f"{base_text} {i}", # Variación mínima para que no sean idénticos
                "timestamp": "2025-11-27T10:00:00Z"
            }
            data.append(entry)
    
    # Guardar en disco
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

# ==========================================
# LIMPIEZA INICIAL
# ==========================================
# Borramos la carpeta artifacts para empezar la prueba desde cero
artifacts_path = os.path.join(os.path.dirname(__file__), "artifacts")
if os.path.exists(artifacts_path):
    print("--- [TEST] Limpiando memoria anterior (borrando artifacts)... ---")
    shutil.rmtree(artifacts_path)

# ==========================================
# ESCENARIO DE PRUEBA
# ==========================================

# Instanciamos el agente
agent = TrendDetectionAgent()

print("\n" + "="*50)
print(">>> DÍA 1: ESTABLECIENDO LÍNEA BASE")
print("="*50)
# Situación: Mucho Python, algo de Futbol. NADA de Aliens.
# Generamos 50 posts de Python y 30 de Futbol
crear_jsonl_prueba("data_day_1.jsonl", {"python": 50, "futbol": 30})

# Ejecutamos el agente
reporte_dia_1 = agent.run("data_day_1.jsonl")
print("Resultados Día 1:", json.dumps(reporte_dia_1, indent=2))


print("\n" + "="*50)
print(">>> DÍA 2: LA LLEGADA DE LA TENDENCIA")
print("="*50)
# Situación: Python sigue estable. Futbol baja. ¡ALIENS EXPLOTA!
# Python: 55 (Estable), Futbol: 10 (Baja), Aliens: 60 (NUEVO Y EXPLOSIVO)
crear_jsonl_prueba("data_day_2.jsonl", {"python": 55, "futbol": 10, "aliens": 60})

# Ejecutamos el agente de nuevo (debe recordar el Día 1)
reporte_dia_2 = agent.run("data_day_2.jsonl")

print("\n--- ANÁLISIS FINAL ---")
print("Buscando si detectó a los Aliens como tendencia...")
found_aliens = False
for trend in reporte_dia_2['data']:
    # El nombre del tema puede variar, pero buscamos palabras clave
    print(f"- Tema detectado: {trend['topic_name']} | Crecimiento: {trend['growth_rate']} | Tipo: {trend['trend_type']}")
    
    if "alien" in trend['topic_name'].lower() or "ufo" in trend['topic_name'].lower():
        found_aliens = True

print("-" * 30)
if found_aliens:
    print("✅ PRUEBA EXITOSA: ¡Tendencia de Aliens detectada!")
else:
    print("❌ PRUEBA FALLIDA: No se detectó la tendencia nueva.")