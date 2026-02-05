#C:\Users\Matias\Documents\tesis\src\agents\sr/app.py
import streamlit as st
import matplotlib.pyplot as plt
from src.agents.sr.tools import SynthesisAgent

# Configuración visual
st.set_page_config(page_title="Agente de Reporte Tesis", layout="wide")

st.title("📊 Agente de Síntesis y Reporte")
st.markdown("Arquitectura Neuro-Simbólica: Pandas (Cálculo) + LLM (Interpretación)")

# 1. CARGA DE ARCHIVO
uploaded_file = st.file_uploader("Sube el archivo JSONL (Salida del Agente de Sentimientos)", type="jsonl")

if uploaded_file:
    # Instanciamos el agente (Ya sabe dónde buscar el .env)
    try:
        agent = SynthesisAgent()
        st.sidebar.success("✅ Credenciales cargadas correctamente")
    except Exception as e:
        st.error(f"Error de credenciales: {e}")
        st.stop()
    
    with st.spinner("Procesando flujo deterministico y generando síntesis neuronal..."):
        # Paso A: Procesar Datos
        result = agent.process_data(uploaded_file)
        
        if result.get("status") == "error":
            st.error(f"Error: {result.get('message')}")
            if "text" in str(result.get("message")):
                 st.info("💡 Consejo: Asegúrate de que el JSONL incluya el campo 'text'.")
        else:
            # Paso B: Generar Narrativa con LLM
            narrative = agent.generate_narrative(result)
            
            # --- MOSTRAR RESULTADOS ---
            st.subheader("📝 Reporte Ejecutivo")
            st.info(narrative)
            
            st.divider()
            
            col1, col2 = st.columns([1, 2])
            metrics = result["metrics"]
            
            with col1:
                st.markdown("### Métricas (Simbólicas)")
                st.metric("Total Posts", metrics['total'])
                st.metric("Tendencia", metrics['dominant'].upper())
                st.metric("Confianza IA", f"{metrics['confidence']*100:.1f}%")
            
            with col2:
                st.markdown("### Distribución")
                counts = metrics['counts']
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.bar(counts.keys(), counts.values(), color=['#4CAF50', '#FF5252', '#FFC107'])
                st.pyplot(fig)

            with st.expander("🔍 Ver Dataframe procesado"):
                st.dataframe(result['raw_df'])