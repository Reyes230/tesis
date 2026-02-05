# Archivo: src/utils/text_cleaning.py
import re
import html

def basic_clean(text: str) -> str:
    """
    Limpieza robusta para NLP en Redes Sociales (Reddit/Twitter).
    Objetivo: Eliminar ruido técnico manteniendo la riqueza semántica (emojis).
    """
    if not text: 
        return ""
    
    # 1. Decodificar entidades HTML (ej: &amp; -> &)
    text = html.unescape(text)
    
    # 2. Eliminar URLs (http, https, www)
    # Las URLs confunden al modelo de sentimiento y generan tópicos basura como "https"
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Eliminar etiquetas de markdown de Reddit (ej: [link](...))
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)

    # 4. Eliminar menciones de usuario (u/usuario, @usuario)
    # No aportan sentimiento y ensucian los clusters
    text = re.sub(r'\bu/\w+|@\w+', '', text)
    
    # 5. Normalizar espacios (eliminar dobles espacios, tabulaciones, saltos excesivos)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # NOTA: NO eliminamos emojis ni puntuación básica (?!.) porque 
    # son vitales para que RoBERTa detecte emociones.
    
    return text