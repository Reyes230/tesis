# src/agents/tools/deterministic_clean.py
import re
import html
import json
import os

class TextCleaner:
    """
    Limpiador determinístico HÍBRIDO.
    Diseñado para servir tanto a RoBERTa (Sentimiento) como a BERTopic (Tendencias).
    
    Estrategia:
    - Elimina ruido técnico (HTML, URLs).
    - PRESERVA estructura de frases, puntuación y emojis (Vital para RoBERTa).
    - La limpieza de stopwords se delegará al vectorizador de BERTopic más adelante.
    """

    @staticmethod
    def clean(text):
        if not text or not isinstance(text, str):
            return ""

        # 1. Decodificar entidades HTML (&amp; -> &, &quot; -> ")
        # A veces Reddit trae basura como &lt;div&gt;
        text = html.unescape(text)

        # 2. Eliminar URLs (http://... o www...)
        # Las URLs no dan sentimiento y ensucian los tópicos.
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # 3. Eliminar etiquetas HTML residuales (<br>, <b>, <i>)
        text = re.sub(r'<.*?>', '', text)

        # 4. Normalizar espacios en blanco
        # Convierte tabs, saltos de línea y dobles espacios en un solo espacio.
        text = re.sub(r'\s+', ' ', text).strip()

        # OJO: 
        # - NO pasamos a minúsculas (Lowercasing): "DIOS" vs "dios" tiene diferente intensidad.
        # - NO quitamos acentos: "papá" vs "papa".
        # - NO quitamos emojis ni signos (?!): Son el 50% del sentimiento.
        
        return text

def run_cleaning_pipeline(input_path):
    """
    Proceso Batch: Lee JSONL sucio -> Limpia -> Guarda JSONL limpio.
    """
    if not input_path or not os.path.exists(input_path):
        return None, f"Error: Archivo no encontrado {input_path}"

    # Preparamos ruta de salida: data/raw/tema.jsonl -> data/preprocessed/tema_cleaned.jsonl
    # Aseguramos que el directorio de salida exista
    output_dir = "data/preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(input_path)
    name_no_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{name_no_ext}_cleaned.jsonl")
    
    clean_count = 0
    
    print(f"   🧹 Limpiando {filename}...")
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip(): continue
            
            try:
                obj = json.loads(line)
                
                # Buscamos el texto original (puede venir como 'text', 'body', 'title')
                # Prioridad: title + body (si existen) para tener más contexto
                raw_text = ""
                if "title" in obj: raw_text += obj["title"] + ". "
                if "body" in obj: raw_text += obj["body"]
                if "text" in obj and not raw_text: raw_text = obj["text"]
                
                # --- APLICAMOS LA LIMPIEZA ---
                cleaned_text = TextCleaner.clean(raw_text)
                
                # Guardamos en un campo estandarizado 'text_norm'
                obj["text_norm"] = cleaned_text
                
                # Filtro de calidad mínima: Si quedaron menos de 10 caracteres, es basura.
                if len(cleaned_text) > 10:
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    clean_count += 1
                    
            except json.JSONDecodeError:
                continue

    return output_path, f"Limpieza completada. {clean_count} posts procesados y guardados en {output_path}"