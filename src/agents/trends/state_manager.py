# C:/Users/Matias/Documents/tesis/src/agents/trends/state_manager.py
import pandas as pd
import os
import joblib
import config 

class TrendStateManager:
    """
    Encargado de gestionar la persistencia del modelo y la memoria histórica.
    """

    def __init__(self):
        self.history_path = config.HISTORY_FILE
        self.model_path = config.MODEL_FILE

    def load_previous_window(self):
        """
        Carga los datos de la ventana anterior (t-1).
        """
        if os.path.exists(self.history_path):
            try:
                df = pd.read_csv(self.history_path)
                print(f"[StateManager] Ventana anterior cargada: {len(df)} temas.")
                return df
            except Exception as e:
                print(f"[StateManager] Error leyendo histórico: {e}. Iniciando vacío.")
                
        return pd.DataFrame(columns=["topic_id", "count_prev"])

    def save_current_window(self, current_counts_df):
        """
        Guarda los conteos actuales.
        """
        # 1. Asegurar que el directorio exista (CORRECCIÓN AQUÍ)
        # Esto evita el error "OSError: Cannot save file into a non-existent directory"
        directory = os.path.dirname(self.history_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # 2. Preparar datos
        df_to_save = current_counts_df.rename(columns={"count": "count_prev"})
        df_to_save = df_to_save[["topic_id", "count_prev"]]
        
        # 3. Guardar
        df_to_save.to_csv(self.history_path, index=False)
        print(f"[StateManager] Estado guardado en {self.history_path}")

    def save_model(self, topic_model):
        """Serializa el modelo BERTopic entrenado"""
        directory = os.path.dirname(self.model_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        try:
            # CORRECCIÓN: Quitamos 'serialization="safetensors"' y 'save_ctfidf=True'
            # Dejamos que BERTopic use su guardado por defecto (pickle) que es más compatible.
            topic_model.save(self.model_path) 
            print("[StateManager] Modelo BERTopic actualizado y guardado exitosamente.")
        except Exception as e:
            print(f"[StateManager] Error guardando modelo: {e}")

    def model_exists(self):
        """Verifica si ya existe un modelo entrenado en disco"""
        return os.path.exists(self.model_path)