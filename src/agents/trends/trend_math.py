# src/agents/trends/trend_math.py
import pandas as pd
import numpy as np

class TrendMathEngine:
    """
    Motor matemático para análisis de 'Snapshot' (Momento actual).
    Ya no calcula velocidad temporal, sino 'Impacto' y 'Dominancia'.
    """

    @staticmethod
    def calculate_impact(topics_data):
        """
        Calcula qué tan relevante es un tema basándose en Volumen y Sentimiento.
        
        Args:
            topics_data (list of dict): La lista de reportes que genera trend_node.
                                        [{'topic_id': 0, 'volume': 20, 'sentiment_avg': -0.5}, ...]
        
        Returns:
            list of dict: La misma lista pero ordenada por importancia y con métricas extra.
        """
        if not topics_data:
            return []

        df = pd.DataFrame(topics_data)

        # 1. Cálculo de Dominancia (% del total de la conversación)
        total_volume = df['volume'].sum()
        if total_volume == 0:
            return topics_data # Evitar división por cero
            
        df['share_of_voice'] = df['volume'] / total_volume

        # 2. Cálculo de Índice de Impacto (Impact Score)
        # FÓRMULA: Volumen * (1 + Intensidad del Sentimiento)
        # Explicación: 
        # - Un tema neutro (sent=0) vale su volumen puro.
        # - Un tema muy polarizado (sent=0.9 o -0.9) casi DUPLICA su peso.
        # - Usamos abs() porque tanto el amor extremo como el odio extremo son virales.
        
        df['impact_score'] = df['volume'] * (1 + df['sentiment_avg'].abs())

        # 3. Categorización de Prioridad (Matriz de Riesgo)
        # Alta Prioridad = Volumen Alto Y Sentimiento Negativo Alto
        def get_priority(row):
            if row['sentiment_avg'] < -0.2 and row['volume'] > (total_volume * 0.1):
                return "CRÍTICA" # Negativo y grande (>10% del total)
            if row['impact_score'] > (df['impact_score'].mean() * 1.5):
                return "ALTA"    # Destaca mucho sobre el promedio
            return "MEDIA"

        df['priority'] = df.apply(get_priority, axis=1)

        # 4. Ordenar por Impacto (No solo por volumen)
        df_sorted = df.sort_values(by='impact_score', ascending=False)

        # Convertir a float nativo de Python para que JSON no falle
        return df_sorted.to_dict(orient='records')