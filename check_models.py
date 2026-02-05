import os
import google.generativeai as genai
from dotenv import load_dotenv

# Carga tu API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") 

if not api_key:
    print("❌ Error: No se encontró la API Key en .env")
else:
    genai.configure(api_key=api_key)
    print("🔍 Buscando modelos disponibles para tu API Key...\n")
    
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"👉 Nombre: {m.display_name}")
                print(f"   ID Técnico: {m.name}") # <--- ESTE ES EL QUE NECESITAS COPIAR
                print("-" * 30)
    except Exception as e:
        print(f"Error conectando: {e}")