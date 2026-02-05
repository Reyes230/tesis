# C:/Users/Matias/Documents/tesis/src/agents/trends/config.py
import os
# =======================================================
# 1. RUTAS Y DIRECTORIOS
# =======================================================
# Obtenemos la ruta absoluta de donde está este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta para guardar el 'cerebro' del agente (Modelos y Logs)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Aseguramos que la carpeta exista (si no, da error al guardar)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Archivos específicos
MODEL_FILE = os.path.join(ARTIFACTS_DIR, "bertopic_model.pkl")
HISTORY_FILE = os.path.join(ARTIFACTS_DIR, "trend_history_window.csv")
REPORT_OUTPUT = os.path.join(ARTIFACTS_DIR, "latest_trend_report.json")

# =======================================================
# 2. CONFIGURACIÓN DEL MODELO NLP
# =======================================================
# Usamos un modelo multilingüe ligero pero potente (RoBERTa variant)
# Opción A: 'paraphrase-multilingual-MiniLM-L12-v2' (Rápido, bueno para Reddit)
# Opción B: 'xlm-r-bert-base-nli-stsb-mean-tokens' (Más pesado, mejor comprensión)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Configuración de BERTopic
MIN_TOPIC_SIZE = 10  # Mínimo de posts para formar un tema
VERBOSE_LOGS = True

# =======================================================
# 3. UMBRALES MATEMÁTICOS DE TENDENCIA
# =======================================================
# Estos son los valores que ajustaremos en tus experimentos

# ALPHA: Tasa de crecimiento mínima (0.5 = 50% de crecimiento)
ALPHA_GROWTH = 0.5 

# BETA: Volumen mínimo en la ventana actual para considerar el crecimiento
# (Evita que pasar de 1 a 2 menciones sea una tendencia del 100%)
BETA_MIN_VOLUME = 5

# GAMMA: Umbral de persistencia (Volumen alto)
# Si un tema tiene más de X menciones, es tendencia aunque no crezca
GAMMA_HIGH_VOLUME = 50