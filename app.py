import streamlit as st
import pandas as pd
import json
import os
import glob
import time
import io
import plotly.express as px

# --- IMPORTACIÓN DEL MOTOR DE INTELIGENCIA ---
try:
    from src.pipeline_controller import run_analysis_pipeline
except ImportError:
    st.error("⚠️ Error crítico: No se encuentra 'src.pipeline_controller'.")

# --- 1. CONFIGURACIÓN PROFESIONAL ---
st.set_page_config(
    page_title="Sistema de Análisis Social IA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ESTILO ACADÉMICO MODERNO (TU DISEÑO) ---
def local_css():
    st.markdown("""
    <style>
        /* FUENTE ACADÉMICA MODERNA (Inter) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #0F172A; color: #F8FAFC; }
        
        /* TÍTULOS */
        h1, h2, h3 { color: #F8FAFC !important; font-weight: 600 !important; }
        h1 { font-size: 2.5rem !important; border-bottom: 1px solid #334155; padding-bottom: 10px; }
        
        /* COMPONENTES */
        section[data-testid="stSidebar"] { background-color: #1E293B; border-right: 1px solid #334155; }
        div[data-testid="metric-container"] { background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        div[data-testid="stMetricValue"] { color: #38BDF8 !important; font-weight: 600; }
        
        /* BOTONES */
        .stButton > button { background-color: #3B82F6; color: white; border: none; border-radius: 6px; padding: 0.5rem 1rem; width: 100%; transition: background 0.2s; }
        .stButton > button:hover { background-color: #2563EB; }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid #334155; }
        .stTabs [data-baseweb="tab"] { height: 50px; background-color: transparent; color: #94A3B8; font-weight: 500; }
        .stTabs [aria-selected="true"] { color: #38BDF8 !important; border-bottom: 2px solid #38BDF8; }
        
        /* STATUS */
        .stStatus { background-color: #1E293B; border: 1px solid #334155; color: #E2E8F0; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. LÓGICA DE DATOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
PREPROC_DIR = os.path.join(BASE_DIR, "data", "preprocessed")

def get_available_reports():
    files = glob.glob(os.path.join(REPORTS_DIR, "*_trends_report.json"))
    options = []
    for f in files:
        base = os.path.basename(f)
        name = base.replace("_trends_report.json", "").replace("_", " ").title()
        options.append({
            "label": name, 
            "file_path": f, 
            "base_name": base.replace("_trends_report.json", ""),
            "mtime": os.path.getmtime(f)
        })
    options.sort(key=lambda x: x["mtime"], reverse=True)
    return options

def load_report_data(base_name):
    data = {}
    
    # 1. Cargar JSON (Metadata)
    json_path = os.path.join(REPORTS_DIR, f"{base_name}_trends_report.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f: data["trends"] = json.load(f)
    
    # 2. Cargar Markdown (Lógica Blindada + Fuzzy)
    # A. Intento Directo (Dictadura del Input)
    target_md = os.path.join(REPORTS_DIR, f"reporte_{base_name}.md")
    if os.path.exists(target_md):
        with open(target_md, 'r', encoding='utf-8') as f: 
            data["markdown"] = f.read()
            data["md_filename"] = os.path.basename(target_md)
    else:
        # B. Intento Fuzzy (Backup)
        keywords = [k for k in base_name.lower().replace("_", " ").split() if len(k) > 2]
        all_mds = glob.glob(os.path.join(REPORTS_DIR, "*.md"))
        
        best_candidate = None
        max_hits = 0
        latest_time = 0

        for md_path in all_mds:
            filename = os.path.basename(md_path).lower()
            if not filename.startswith("report"): continue
            
            hits = sum(1 for k in keywords if k in filename)
            md_time = os.path.getmtime(md_path)
            
            if hits > max_hits:
                max_hits = hits
                best_candidate = md_path
                latest_time = md_time
            elif hits == max_hits and hits > 0:
                if md_time > latest_time:
                    best_candidate = md_path
                    latest_time = md_time

        if best_candidate:
            with open(best_candidate, 'r', encoding='utf-8') as f: 
                data["markdown"] = f.read()
                data["md_filename"] = os.path.basename(best_candidate)
        else:
            data["markdown"] = f"⚠️ No se encontró reporte textual para '{base_name}'."

    # 3. Cargar Raw Data y Preparar Links
    raw_path_pattern = os.path.join(PREPROC_DIR, f"{base_name}*_cleaned_with_sentiment.jsonl")
    found_files = glob.glob(raw_path_pattern)
    if found_files:
        raw_rows = []
        with open(found_files[0], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): raw_rows.append(json.loads(line))
        df = pd.DataFrame(raw_rows)
        if 'sentiment' in df.columns and not df.empty:
            sent = pd.json_normalize(df['sentiment'])
            sent.columns = [f"sent_{c}" for c in sent.columns]
            df = pd.concat([df.drop(columns=['sentiment']), sent], axis=1)
            
            # --- CORRECCIÓN TUTOR: CREAR LINKS A REDDIT ---
            if 'post_id' in df.columns:
                # Si el ID existe, crea el link, si no, pone #
                df['link_post'] = df['post_id'].apply(
                    lambda x: f"https://reddit.com/comments/{x}" if pd.notnull(x) and str(x) != "nan" else "#"
                )
            
        data["raw_df"] = df
    else:
        data["raw_df"] = pd.DataFrame()
        
    return data

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

# --- 4. INTERFAZ GRÁFICA ---

st.title("Sistema Multi-Agente de Análisis Social")
st.markdown("""<div style='color: #94A3B8; margin-bottom: 20px;'><strong>Proyecto de Tesis de Ingeniería</strong> | Procesamiento de Lenguaje Natural & Detección de Comunidades</div>""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("Panel de Configuración")
mode = st.sidebar.radio("Modo de Operación:", ["📂 Historial de Análisis", "🚀 Nuevo Análisis (En Vivo)"])

if mode == "📂 Historial de Análisis":
    reports = get_available_reports()
    if not reports:
        st.sidebar.warning("No se encontraron datasets.")
    else:
        options_map = {r["label"]: r["base_name"] for r in reports}
        choice = st.sidebar.selectbox("Seleccionar Dataset:", list(options_map.keys()))
        if st.sidebar.button("Cargar Análisis"):
            st.session_state['current_base_name'] = options_map[choice]
            st.session_state['current_topic_label'] = choice
            st.rerun()

elif mode == "🚀 Nuevo Análisis (En Vivo)":
    st.sidebar.markdown("---")
    topic_input = st.sidebar.text_input("Tema a Investigar:", placeholder="Ej: Elecciones 2025")
    if st.sidebar.button("INICIAR EJECUCIÓN"):
        if not topic_input:
            st.sidebar.error("Ingresa un tema.")
        else:
            with st.status("⚙️ Ejecutando Pipeline de IA...", expanded=True) as status:
                def update_ui(msg): status.write(msg)
                success = run_analysis_pipeline(topic_input, update_ui)
                if success:
                    status.update(label="✅ Análisis Completado", state="complete")
                    time.sleep(1)
                    # Normalización forzosa para coincidir con el backend
                    safe_topic = topic_input.lower().replace(" ", "_")
                    st.session_state['current_base_name'] = safe_topic
                    st.session_state['current_topic_label'] = topic_input
                    st.rerun()
                else:
                    status.update(label="❌ Error en la Ejecución", state="error")

st.sidebar.markdown("---")
st.sidebar.caption("v1.1.0 | Release Candidato Tesis")

# --- VISUALIZACIÓN ---

if 'current_base_name' in st.session_state:
    base_name = st.session_state['current_base_name']
    label = st.session_state.get('current_topic_label', base_name)
    data = load_report_data(base_name)
    
    st.markdown(f"### 🔎 Resultados para: **{label}**")
    
    tab1, tab2, tab3 = st.tabs(["📄 Síntesis Ejecutiva", "🧪 Validación y Evidencia", "🕸️ Clusters Temáticos"])
    
    # --- TAB 1: SÍNTESIS ---
    with tab1:
        if "trends" in data:
            trends = data["trends"]
            df_trends = pd.DataFrame(trends)
            kpi1, kpi2, kpi3 = st.columns(3)
            vol = df_trends['volume'].sum() if not df_trends.empty else 0
            sent_avg = df_trends['sentiment_avg'].mean() if not df_trends.empty else 0
            kpi1.metric("Muestra Total (n)", f"{vol}")
            kpi2.metric("Índice de Sentimiento", f"{sent_avg:.3f}")
            kpi3.metric("Comunidades Detectadas", len(df_trends))
            
        st.markdown("---")
        if "markdown" in data:
            st.markdown(f"<div style='background-color: #1E293B; padding: 25px; border-radius: 8px; border-left: 4px solid #3B82F6; color: #E2E8F0; line-height: 1.6;'>{data['markdown']}</div>", unsafe_allow_html=True)
            fname = data.get("md_filename", f"{base_name}_reporte.md")
            st.download_button("📥 Descargar Reporte (Markdown)", data["markdown"], file_name=fname, mime="text/markdown")
        else:
            st.info("No hay reporte textual disponible.")

    # --- TAB 2: SENTIMIENTO Y EVIDENCIA (RESTAURADA) ---
    with tab2:
        df_raw = data.get("raw_df")
        if df_raw is not None and not df_raw.empty:
            
            col_chart1, col_chart2 = st.columns(2)
            
            # 1. GRÁFICO DE PASTEL (DONUT) - Distribución General
            with col_chart1:
                st.markdown("#### Distribución de Sentimiento")
                if 'sent_label' in df_raw.columns:
                    sent_counts = df_raw['sent_label'].value_counts().reset_index()
                    sent_counts.columns = ['Sentimiento', 'Conteo']
                    color_map = {'negative': '#EF4444', 'neutral': '#94A3B8', 'positive': '#10B981'}
                    fig_pie = px.pie(sent_counts, values='Conteo', names='Sentimiento', 
                                     color='Sentimiento', color_discrete_map=color_map, hole=0.5)
                    fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#F8FAFC', margin=dict(t=0, b=0, l=0, r=0), height=250)
                    st.plotly_chart(fig_pie, use_container_width=True)

            # 2. HISTOGRAMA DE CONFIANZA (RESTAURADO) - Calidad del Modelo
            with col_chart2:
                st.markdown("#### Calidad del Modelo (Confianza)")
                if 'sent_confidence' in df_raw.columns:
                    fig_hist = px.histogram(df_raw, x="sent_confidence", nbins=20, 
                                          labels={'sent_confidence': 'Score de Certeza'},
                                          color_discrete_sequence=['#38BDF8'])
                    fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#F8FAFC', margin=dict(t=0, b=0, l=0, r=0), height=250, showlegend=False)
                    fig_hist.add_vline(x=0.8, line_dash="dash", line_color="#10B981")
                    st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🕵️ Auditoría de Publicaciones (Evidencia)")
            
            # 3. TABLA CON LINKS
            cols_to_show = ['sent_label', 'sent_confidence', 'text_raw', 'link_post']
            cols_exist = [c for c in cols_to_show if c in df_raw.columns]
            df_display = df_raw[cols_exist].copy()
            
            st.dataframe(
                df_display,
                use_container_width=True,
                height=400,
                column_config={
                    "link_post": st.column_config.LinkColumn("Fuente", display_text="Ver en Reddit"),
                    "sent_confidence": st.column_config.ProgressColumn("Certeza", min_value=0, max_value=1, format="%.2f"),
                    "text_raw": st.column_config.TextColumn("Texto", width="large"),
                    "sent_label": st.column_config.TextColumn("Sentimiento")
                }
            )

    # --- TAB 3: CLUSTERS (RENOVADO: BUBBLE CHART) ---
    with tab3:
        if "trends" in data:
            st.markdown("#### Mapa Estratégico de Tópicos")
            st.caption("Eje X: Polaridad (Negativo ↔ Positivo) | Eje Y: Volumen de Conversación | Tamaño: Impacto")
            
            df_trends = pd.DataFrame(data["trends"])
            
            if not df_trends.empty:
                # 4. BUBBLE CHART (Mucho mejor que Treemap para ver distribución)
                fig_bubble = px.scatter(
                    df_trends, 
                    x="sentiment_avg", 
                    y="volume",
                    size="volume",
                    color="sentiment_avg",
                    hover_name="label",
                    text="label", # Muestra el nombre del tema
                    color_continuous_scale='RdBu', # Rojo a Azul
                    range_x=[-1.1, 1.1], # Fija el rango para ver claramente negativo vs positivo
                    size_max=60
                )
                
                # Línea central y estilo
                fig_bubble.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
                fig_bubble.update_traces(textposition='top center')
                fig_bubble.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.05)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font_color='#F8FAFC',
                    xaxis_title="Sentimiento Promedio",
                    yaxis_title="Volumen de Menciones",
                    height=500
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
                
                st.markdown("#### Detalle de Comunidades")
                st.dataframe(df_trends[['label', 'volume', 'status', 'example_text']], use_container_width=True)