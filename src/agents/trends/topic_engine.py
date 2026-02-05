# src/agents/trends/topic_engine.py

# --- 1. Importaciones Nuevas para Limpieza ---
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer # Usamos el estándar para listas custom

# --- Importaciones de BERTopic y Clustering ---
from bertopic import BERTopic
from sklearn.cluster import MiniBatchKMeans

try:
    from src.agents.trends import config
except ImportError:
    from . import config

class TopicModelEngine:
    """
    Motor de Tópicos 'Stateless' (Sin Estado).
    Se entrena desde cero en cada ejecución para adaptarse 
    al tema consultado en ese momento, con limpieza avanzada de idiomas.
    """

    def __init__(self):
        self.model = None
        # Precarga de NLTK para no fallar en ejecución
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("[TopicEngine] Descargando stopwords de NLTK...")
            nltk.download('stopwords')

    def _get_custom_stopwords(self):
        """Genera la super-lista de palabras a ignorar"""
        stop_es = stopwords.words('spanish')
        stop_en = stopwords.words('english')
        
        # Intentamos leer la lista del config, si no existe, usamos lista vacía
        stop_custom = getattr(config, 'CUSTOM_STOP_WORDS', []) 
        
        # Unimos todo: Español + Inglés + Tu lista de Config
        return stop_es + stop_en + stop_custom

    def fit_transform(self, texts):
        """
        Entrena el modelo con los textos actuales y retorna los tópicos.
        """
        if not texts:
            return [], None

        print(f"[TopicEngine] 🚀 Iniciando análisis 'Snapshot' para {len(texts)} documentos...")

        # 1. Configuración de Clustering (MiniBatchKMeans)
        # Forzamos 5 clusters para encontrar narrativas incluso con pocos datos
        cluster_model = MiniBatchKMeans(
            n_clusters=5, 
            random_state=42,
            batch_size=256
        )

        # 2. Vectorizador AVANZADO (El cambio importante)
        # Aquí inyectamos la lista combinada de stopwords
        final_stopwords = self._get_custom_stopwords()
        
        vectorizer_model = CountVectorizer(
            stop_words=final_stopwords, 
            min_df=2 # La palabra debe aparecer al menos 2 veces
        )

        # 3. Inicializar BERTopic con el vectorizador limpio
        self.model = BERTopic(
            embedding_model=config.EMBEDDING_MODEL_NAME,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model, # <--- Aquí entra la limpieza
            min_topic_size=3,  
            verbose=True,
            calculate_probabilities=False,
            language="multilingual" # Importante declararlo multilingüe explícitamente
        )

        # 4. Entrenar y Transformar
        try:
            topics, probs = self.model.fit_transform(texts)
            
            # Debug: Mostrar qué encontró (ahora debería salir limpio)
            info = self.model.get_topic_info()
            print(f"[TopicEngine] ✅ Tópicos detectados:\n{info.head()}")
            
            return topics, self.model
            
        except Exception as e:
            print(f"[TopicEngine] ⚠️ Advertencia: Error al generar clusters: {e}")
            return [-1] * len(texts), None

    def get_topic_label(self, topic_id):
        """Obtiene un nombre legible para el tópico"""
        if self.model is None or topic_id == -1:
            return "General / Disperso"
            
        try:
            topic_words = self.model.get_topic(topic_id)
            if topic_words:
                # Tomamos las palabras más relevantes que YA NO SERÁN stopwords
                return "_".join([word[0] for word in topic_words[:3]])
        except:
            pass
        return f"Tema_{topic_id}"