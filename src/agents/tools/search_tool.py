# src/agents/tools/search_tool.py
import warnings
import time
from langchain_core.tools import tool
from duckduckgo_search import DDGS

# --- SILENCIADOR DE ADVERTENCIAS ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

@tool("web_search")
def web_search(query: str) -> str:
    """
    Realiza una búsqueda en internet con REDUNDANCIA para evitar bloqueos.
    Intenta múltiples estrategias (API -> HTML -> Lite) para garantizar resultados.
    """
    print(f"   🌐 [Internet] Buscando: '{query}'...")
    
    # ESTRATEGIA
    # 'api': El más rápido (default), pero a veces bloquea.
    # 'html': Scrapeo clásico, más lento pero muy robusto.
    # 'lite': Versión ligera para conexiones lentas.
    backends_to_try = ["api", "html", "lite"]
    
    results_text = ""
    last_error = ""

    for backend in backends_to_try:
        try:
            # Pequeña pausa de seguridad si estamos reintentando
            if backend != "api": 
                time.sleep(1)
                # print(f"   🔄 Reintentando con modo '{backend}'...") 

            with DDGS() as ddgs:
                # max_results=4 punto dulce entre velocidad y seguridad
                gen_results = list(ddgs.text(
                    query, 
                    region="wt-wt", 
                    safesearch="off", 
                    max_results=4, 
                    backend=backend
                ))
                
                if gen_results:
                    results_text += f"--- RESULTADOS DE BÚSQUEDA ('{query}') ---\n"
                    for r in gen_results:
                        title = r.get('title', 'Sin título')
                        body = r.get('body', 'Sin contenido')
                        href = r.get('href', '#')
                        
                        if "amazon" in href or "ebay" in href: 
                            continue
                            
                        results_text += f"TITULO: {title}\n"
                        results_text += f"RESUMEN: {body}\n"
                        results_text += f"FUENTE: {href}\n\n"
                    
                    return results_text
                
        except Exception as e:
            last_error = str(e)
            # Si falla, el bucle 'for' intentará el siguiente backend automáticamente
            continue 

    # Si fallaron los 3 backends 
    fallback_msg = (
        f"AVISO TÉCNICO: No se pudieron obtener resultados externos tras 3 intentos. "
        f"El análisis continuará solo con datos internos. (Error: {last_error})"
    )
    return fallback_msg