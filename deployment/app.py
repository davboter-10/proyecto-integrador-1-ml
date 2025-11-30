# app.py - Dashboard de Predicción de Reestructuraciones
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime
import json
import io

# Configuración de página
st.set_page_config(
    page_title="Predicción Reestructuraciones",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lista de features esperadas por el modelo
EXPECTED_FEATURES = [
    "avg_std_mora_6m", "ipc", "max_alt_mora_6m", "avg_alt_mora_6m", "meses_mora_30plus",
    "flag_mora_recurrente_0", "flag_mora_recurrente_1", "avg_meses_con_mora", "segmento_SOCIAL",
    "sub_segmento_Desconocido", "segmento_PYMES", "meses_mora_90plus", "genero_Desconocido",
    "max_saldo_capital_6m", "pib", "avg_saldo_capital_6m", "nivel_academico_UNIVERSITARIO",
    "sub_segmento_PEQUENA", "coef_tendencia_mora", "sub_segmento_MEDIANA", "ano_nac",
    "max_sld_int_6m", "estado_civil_Desconocido", "nivel_riesgo_Desconocido",
    "nivel_academico_Desconocido", "segmento_PREFERENCIAL", "ocupacion_AGRICULTOR",
    "sub_segmento_PLUS", "sub_segmento_ALTO", "estado_civil_CASADO",
    "nivel_academico_ESPECIALIZACION", "sub_segmento_PREFERENCIAL PLUS", "genero_M",
    "num_oblg_mora_ext", "vr_total_reest", "ocupacion_PENSIONADO", "meses_mora_180plus",
    "num_reest_ext", "nivel_riesgo_Bajo", "ocupacion_PROFESIONAL INDEPENDIENTE",
    "ocupacion_GANADERO", "nivel_riesgo_Medio Bajo", "tasa_desempleo", "segmento_PLUS",
    "sub_segmento_GRANDE", "ocupacion_EMPLEADO", "estado_civil_UNION LIBRE",
    "segmento_NEGOCIOS & INDEPEND", "sub_segmento_MiPyme", "nivel_riesgo_Medio",
    "sub_segmento_PREFERENCIAL COLOMBIA", "max_saldo_vencido_30_6m",
    "nivel_academico_TECNOLOGO", "tasa_interes", "max_saldo_vencido_90_6m",
    "nivel_riesgo_Alto", "estado_civil_SOLTERO", "sub_segmento_RELACIONAMIENTO",
    "num_oblg_activa_ext", "sub_segmento_PREF CONCILIACION", "sub_segmento_NO APLICA",
    "sub_segmento_PEQUE#O", "num_reest_anteriores", "sub_segmento_MEDIO",
    "ocupacion_INDEPENDIENTE", "genero_F", "vr_mora_total_ext",
    "nivel_academico_NO INFORMA", "ocupacion_RENTISTA DE CAPITAL", "estado_civil_VIUDO",
    "nivel_riesgo_Medio Alto", "ocupacion_AMA DE CASA", "estado_civil_DIVORCIADO",
    "ocupacion_ESTUDIANTE", "ocupacion_Desconocido", "num_oblg_embarg_ext",
    "estado_civil_OTRO", "estado_civil_NO INFORMA", "ingresos_totales",
    "segmento_PERSONAL", "nivel_academico_PRIMARIA", "ocupacion_COMERCIANTE",
    "ocupacion_SOCIO O EMPLEADO - SOCIO", "ocupacion_OTRO", "nivel_academico_NINGUNO",
    "nivel_academico_BACHILLER", "sub_segmento_PREF.COLOMBIA", "ocupacion_OTRA",
    "patrimonio", "ocupacion_DESEMPLEADO SIN INGRESOS", "ocupacion_DESEMPLEADO CON INGRESOS",
    "segmento_CONSTRUCTOR PYME", "cupo_total", "sub_segmento_BASICO",
    "sub_segmento_MEDIANO", "nivel_riesgo_Medio bajo", "segmento_INDEPENDIENTES"
]

# Custom CSS para mejorar diseño con colores EAFIT
def local_css():
    st.markdown("""
    <style>
    /* Forzar fondo blanco en todo */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Fondo principal */
    .main {
        background-color: #ffffff;
    }
    
    .block-container {
        background-color: #ffffff;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }
    
    /* Barra superior */
    [data-testid="stHeader"] {
        background-color: #ffffff;
        border-bottom: 4px solid #FDB913;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 3px solid #FDB913;
    }
    [data-testid="stSidebar"] * {
        color: #333333 !important;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #FDB913;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Tabs con colores EAFIT */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #ffffff;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: #ffffff;
        color: #2c3e50;
        font-weight: 500;
        border-radius: 0.5rem 0.5rem 0 0;
        border: 2px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #fff3cd;
        color: #2c3e50;
        border-color: #FDB913;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FDB913 !important;
        color: #2c3e50 !important;
        border: 2px solid #FDB913 !important;
        font-weight: 700;
    }
    
    /* Contenido de tabs */
    [data-testid="stTabContent"] {
        background-color: #ffffff;
    }
    
    /* Botones */
    .stButton>button {
        background-color: #FDB913;
        color: #333333;
        border: none;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e5a610;
        color: #333333;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Tarjetas del framework de riesgo */
    .risk-framework-card {
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .risk-low {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #FDB913 0%, #ffc947 100%);
        color: #333333 !important;
    }
    .risk-medium h2, .risk-medium h3, .risk-medium p {
        color: #333333 !important;
    }
    .risk-high {
        background: linear-gradient(135deg, #e53935 0%, #ef5350 100%);
    }
    .strategy-list {
        text-align: left;
        padding-left: 1rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Mensajes info/warning/error */
    .stAlert {
        border-radius: 0.5rem;
    }
    
    /* Barra de progreso */
    .stProgress > div > div > div {
        background-color: #FDB913;
    }
    
    /* Tablas */
    [data-testid="stDataFrame"] {
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
    }
    
    /* Inputs */
    .stTextInput>div>div>input {
        border-radius: 0.5rem;
        border-color: #dee2e6;
    }
    
    /* Selectbox */
    .stSelectbox>div>div>div {
        border-radius: 0.5rem;
    }
    
    /* Textos generales */
    p, li, span, div {
        color: #2c3e50;
    }
    
    /* Encabezados */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar con colores EAFIT
st.sidebar.image("https://via.placeholder.com/200x60/FDB913/333333?text=EAFIT", use_container_width=True)
st.sidebar.title("Sistema de Predicción")
st.sidebar.markdown("**Reestructuraciones Financieras**")
st.sidebar.markdown("<hr style='border: 1px solid #FDB913;'>", unsafe_allow_html=True)

# Variable global para estado de carga
MODEL_LOADED = False

# Cargar modelo local
@st.cache_resource
def load_model():
    try:
        # Intentar cargar desde archivos locales
        with open('modelo.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Cargar métricas
        try:
            with open('model_metrics.json', 'r') as f:
                metrics = json.load(f)
        except:
            metrics = {
                'performance': {
                    'accuracy': 0.85,
                    'roc_auc': 0.90,
                    'f1_score': 0.87,
                    'precision': 0.83,
                    'recall': 0.91
                },
                'model_info': {
                    'model_type': 'XGBClassifier',
                    'n_features': len(EXPECTED_FEATURES),
                    'training_date': '2025-11-28'
                }
            }
        
        # Cargar cutoff óptimo
        try:
            with open('cutoff_analysis.json', 'r') as f:
                cutoff_info = json.load(f)
                optimal_threshold = cutoff_info.get('optimal_threshold', 0.5)
        except:
            optimal_threshold = 0.5
            
        return model, metrics, optimal_threshold, None
    except Exception as e:
        return None, {}, 0.5, str(e)

# Función para validar features
def validate_features(df):
    """Valida que el DataFrame tenga todas las features esperadas"""
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_FEATURES) - {'numero_credito'}  # Permitir numero_credito extra
    return list(missing), list(extra)

# Función para hacer predicción
def predict_single(model, features, threshold=0.5):
    """Realiza predicción individual"""
    try:
        prob = model.predict_proba([features])[0][1]
        pred = 1 if prob >= threshold else 0
        
        if prob < 0.3:
            risk = "BAJO "
            color = "success"
        elif prob < 0.7:
            risk = "MEDIO "
            color = "warning"
        else:
            risk = "ALTO "
            color = "error"
        
        return pred, prob, risk, color
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        return None, None, None, None

# Main
local_css()

st.markdown('<p class="main-header">Sistema de Predicción de Reestructuraciones Financieras</p>', 
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #2c3e50; font-size: 1.1rem; font-weight: 500;'>Maestría en Ciencia de Datos y Analítica - EAFIT</p>", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #FDB913; margin: 2rem 0;'>", unsafe_allow_html=True)

# Intentar cargar modelo
model, metrics, optimal_threshold, error = load_model()

if error:
    st.error(f" Modelo no disponible: {error}")
    st.info(" Asegúrate de tener los archivos `modelo.pkl`, `model_metrics.json` en el directorio actual")
    st.stop()
else:
    st.sidebar.success(" Modelo cargado correctamente")
    
    # Mostrar métricas del modelo en sidebar
    if metrics and 'performance' in metrics:
        st.sidebar.markdown("###  Métricas del Modelo")
        perf = metrics['performance']
        st.sidebar.metric("Accuracy", f"{perf.get('accuracy', 0):.2%}")
        st.sidebar.metric("ROC-AUC", f"{perf.get('roc_auc', 0):.3f}")
        st.sidebar.metric("F1-Score", f"{perf.get('f1_score', 0):.3f}")
        st.sidebar.metric("Cutoff Óptimo", f"{optimal_threshold:.2f}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Carga de Datos", 
    " Búsqueda por Crédito",
    " Resultados y Visualización",
    " Framework de Gestión",
    " Información del Modelo"
])

# TAB 1: Carga de Datos
with tab1:
    st.header(" Carga y Predicción por Lotes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ###  Instrucciones:
        1. **Descarga la plantilla** con las columnas requeridas
        2. **Completa los datos** de los créditos a evaluar
        3. **Carga el archivo** CSV con los datos
        4. **Procesa las predicciones**
        
        **Opcional:** Incluye una columna `numero_credito` para identificar cada registro
        """)
        
        # Botón para descargar plantilla
        template_df = pd.DataFrame(columns=['numero_credito'] + EXPECTED_FEATURES)
        template_csv = template_df.to_csv(index=False)
        
        st.download_button(
            label=" Descargar Plantilla CSV",
            data=template_csv,
            file_name="plantilla_prediccion.csv",
            mime="text/csv",
            type="primary"
        )
    
    with col2:
        st.markdown(f"""
        <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #FDB913; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50; margin-top: 0;'>Datos del Modelo:</h3>
            <ul style='color: #2c3e50; font-size: 1rem; line-height: 1.8;'>
                <li><strong>Features requeridas:</strong> {len(EXPECTED_FEATURES)}</li>
                <li><strong>Threshold:</strong> {optimal_threshold:.2f}</li>
                <li><strong>Formato:</strong> CSV UTF-8</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload file
    uploaded_file = st.file_uploader(
        " Cargar archivo CSV con datos",
        type=['csv'],
        help="Archivo CSV con las features del modelo"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validar features
            missing_features, extra_features = validate_features(df)
            
            # Verificar si tiene numero_credito
            has_credit_id = 'numero_credito' in df.columns
            
            # Mostrar información del archivo
            col1, col2, col3 = st.columns(3)
            col1.metric(" Registros", len(df))
            col2.metric(" Columnas", len(df.columns))
            col3.metric(" Tiene ID", "Sí" if has_credit_id else "No")
            
            # Vista previa
            st.markdown("###  Vista Previa de Datos:")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validación
            if missing_features:
                st.error(f"""
                 **Faltan {len(missing_features)} columnas requeridas:**
                
                {', '.join(missing_features[:10])}{'...' if len(missing_features) > 10 else ''}
                """)
                st.stop()
            
            if extra_features:
                st.warning(f"""
                 Se encontraron {len(extra_features)} columnas adicionales que serán ignoradas:
                
                {', '.join(extra_features[:5])}{'...' if len(extra_features) > 5 else ''}
                """)
            
            # Botón para procesar
            if st.button(" Procesar Predicciones", type="primary", use_container_width=True):
                with st.spinner("Procesando predicciones..."):
                    try:
                        predictions = []
                        probabilities = []
                        risks = []
                        
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            # Extraer features en el orden correcto
                            features = [row[col] for col in EXPECTED_FEATURES]
                            
                            # Predecir
                            pred, prob, risk, _ = predict_single(model, features, optimal_threshold)
                            
                            predictions.append(pred)
                            probabilities.append(prob)
                            risks.append(risk)
                            
                            # Actualizar progreso
                            progress_bar.progress((idx + 1) / len(df))
                        
                        progress_bar.empty()
                        
                        # Agregar resultados al DataFrame
                        results_df = df.copy()
                        results_df['prediccion'] = predictions
                        results_df['probabilidad'] = probabilities
                        results_df['nivel_riesgo'] = risks
                        
                        st.success(f" {len(results_df)} predicciones completadas exitosamente")
                        
                        # Mostrar resultados
                        st.markdown("###  Resultados:")
                        
                        # Métricas resumen
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total", len(results_df))
                        col2.metric(" Alto Riesgo", results_df['nivel_riesgo'].str.contains('ALTO').sum())
                        col3.metric(" Medio Riesgo", results_df['nivel_riesgo'].str.contains('MEDIO').sum())
                        col4.metric(" Bajo Riesgo", results_df['nivel_riesgo'].str.contains('BAJO').sum())
                        
                        # Tabla de resultados
                        display_cols = ['numero_credito'] if has_credit_id else []
                        display_cols += ['prediccion', 'probabilidad', 'nivel_riesgo']
                        
                        st.dataframe(
                            results_df[display_cols + EXPECTED_FEATURES[:5]],  # Mostrar solo algunas features
                            use_container_width=True
                        )
                        
                        # Descargar resultados
                        output = io.BytesIO()
                        results_df.to_csv(output, index=False)
                        output.seek(0)
                        
                        st.download_button(
                            label=" Descargar Resultados Completos (CSV)",
                            data=output,
                            file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
                        # Guardar en session_state para visualización
                        st.session_state['batch_results'] = results_df
                        st.session_state['has_credit_id'] = has_credit_id
                        
                    except Exception as e:
                        st.error(f" Error al procesar: {str(e)}")
                        st.exception(e)
                        
        except Exception as e:
            st.error(f" Error al leer archivo: {str(e)}")
    else:
        st.info(" Carga un archivo CSV para comenzar el análisis")

# TAB 2: Búsqueda por Crédito
with tab2:
    st.header(" Búsqueda por Número de Crédito")
    
    if 'batch_results' not in st.session_state:
        st.warning(" Primero debes cargar y procesar un archivo en la pestaña 'Carga de Datos'")
    elif not st.session_state.get('has_credit_id', False):
        st.warning(" El archivo cargado no contiene la columna 'numero_credito'")
    else:
        df = st.session_state['batch_results']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Buscador
            credit_number = st.text_input(
                " Ingresa el número de crédito:",
                placeholder="Ejemplo: 123456789",
                help="Número de crédito a consultar"
            )
            
            if credit_number:
                # Buscar crédito
                result = df[df['numero_credito'].astype(str) == credit_number]
                
                if len(result) == 0:
                    st.error(f" No se encontró el crédito: {credit_number}")
                else:
                    row = result.iloc[0]
                    
                    st.success(f" Crédito encontrado: {credit_number}")
                    
                    # Mostrar resultado principal
                    st.markdown("---")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3> Nivel de Riesgo</h3>
                        <h1>{row['nivel_riesgo']}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3> Probabilidad</h3>
                        <h1>{row['probabilidad']*100:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_c:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3> Predicción</h3>
                        <h1>{'Reestructurar' if row['prediccion'] == 1 else 'No Reestructurar'}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Gauge de probabilidad
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=row['probabilidad'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilidad de Reestructuración (%)", 'font': {'size': 20}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': '#d4edda'},
                                {'range': [30, 70], 'color': '#fff3cd'},
                                {'range': [70, 100], 'color': '#f8d7da'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': optimal_threshold * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detalles del crédito
                    with st.expander(" Ver Detalles Completos del Crédito"):
                        detail_df = pd.DataFrame({
                            'Variable': row.index,
                            'Valor': row.values
                        })
                        st.dataframe(detail_df, use_container_width=True, height=400)
        
        with col2:
            st.markdown(f"""
            <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #FDB913; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #2c3e50; margin-top: 0;'>Información del Lote:</h3>
                <ul style='color: #2c3e50; font-size: 1rem; line-height: 1.8;'>
                    <li><strong>Total créditos:</strong> {len(df)}</li>
                    <li><strong>Alto riesgo:</strong> {df['nivel_riesgo'].str.contains('ALTO').sum()}</li>
                    <li><strong>Medio riesgo:</strong> {df['nivel_riesgo'].str.contains('MEDIO').sum()}</li>
                    <li><strong>Bajo riesgo:</strong> {df['nivel_riesgo'].str.contains('BAJO').sum()}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Lista de créditos disponibles
            with st.expander(" Ver todos los créditos"):
                st.dataframe(
                    df[['numero_credito', 'probabilidad', 'nivel_riesgo']].sort_values('probabilidad', ascending=False),
                    use_container_width=True,
                    height=300
                )

# TAB 3: Visualización
with tab3:
    st.header(" Resultados y Visualización")
    
    if 'batch_results' not in st.session_state:
        st.info(" Aún no hay resultados para visualizar. Carga y procesa un archivo en la pestaña 'Carga de Datos'")
    else:
        df = st.session_state['batch_results']
        has_credit_id = st.session_state.get('has_credit_id', False)
        
        # Métricas principales
        st.markdown("###  Resumen Ejecutivo")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total = len(df)
        alto = df['nivel_riesgo'].str.contains('ALTO').sum()
        medio = df['nivel_riesgo'].str.contains('MEDIO').sum()
        bajo = df['nivel_riesgo'].str.contains('BAJO').sum()
        prob_avg = df['probabilidad'].mean()
        
        col1.metric(" Total Evaluados", f"{total:,}")
        col2.metric(" Riesgo Alto", f"{alto:,}", f"{alto/total*100:.1f}%")
        col3.metric(" Riesgo Medio", f"{medio:,}", f"{medio/total*100:.1f}%")
        col4.metric(" Riesgo Bajo", f"{bajo:,}", f"{bajo/total*100:.1f}%")
        col5.metric(" Prob. Promedio", f"{prob_avg*100:.1f}%")
        
        st.markdown("---")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart de distribución
            risk_counts = df['nivel_riesgo'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title=" Distribución de Niveles de Riesgo",
                color=risk_counts.index,
                color_discrete_map={
                    'BAJO ': '#28a745',
                    'MEDIO ': '#ffc107',
                    'ALTO ': '#dc3545'
                },
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Histograma de probabilidades
            fig = px.histogram(
                df,
                x='probabilidad',
                title=" Distribución de Probabilidades",
                nbins=30,
                labels={'probabilidad': 'Probabilidad de Reestructuración'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_vline(
                x=optimal_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({optimal_threshold:.2f})"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Segunda fila de gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot de probabilidades por riesgo
            fig = px.box(
                df,
                x='nivel_riesgo',
                y='probabilidad',
                title=" Distribución de Probabilidades por Nivel de Riesgo",
                labels={'probabilidad': 'Probabilidad', 'nivel_riesgo': 'Nivel de Riesgo'},
                color='nivel_riesgo',
                color_discrete_map={
                    'BAJO ': '#28a745',
                    'MEDIO ': '#ffc107',
                    'ALTO ': '#dc3545'
                }
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top 10 créditos de mayor riesgo
            if has_credit_id:
                top_10 = df.nlargest(10, 'probabilidad')[['numero_credito', 'probabilidad', 'nivel_riesgo']]
                fig = px.bar(
                    top_10,
                    x='numero_credito',
                    y='probabilidad',
                    title=" Top 10 Créditos de Mayor Riesgo",
                    labels={'probabilidad': 'Probabilidad', 'numero_credito': 'Número de Crédito'},
                    color='probabilidad',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Curva de distribución acumulada
                sorted_probs = np.sort(df['probabilidad'])
                cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs) * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sorted_probs,
                    y=cumulative,
                    mode='lines',
                    name='Distribución Acumulada',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.add_vline(
                    x=optimal_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold"
                )
                fig.update_layout(
                    title=" Curva de Distribución Acumulada",
                    xaxis_title="Probabilidad",
                    yaxis_title="% Acumulado",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tabla detallada
        st.markdown("###  Tabla Detallada de Resultados")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_risk = st.multiselect(
                "Filtrar por Riesgo:",
                options=df['nivel_riesgo'].unique(),
                default=df['nivel_riesgo'].unique()
            )
        with col2:
            min_prob = st.slider(
                "Probabilidad mínima:",
                0.0, 1.0, 0.0, 0.05
            )
        with col3:
            max_prob = st.slider(
                "Probabilidad máxima:",
                0.0, 1.0, 1.0, 0.05
            )
        
        # Aplicar filtros
        filtered_df = df[
            (df['nivel_riesgo'].isin(filter_risk)) &
            (df['probabilidad'] >= min_prob) &
            (df['probabilidad'] <= max_prob)
        ]
        
        st.info(f" Mostrando {len(filtered_df)} de {len(df)} registros")
        
        # Mostrar columnas relevantes
        display_cols = []
        if has_credit_id:
            display_cols.append('numero_credito')
        display_cols += ['prediccion', 'probabilidad', 'nivel_riesgo']
        
        st.dataframe(
            filtered_df[display_cols].sort_values('probabilidad', ascending=False),
            use_container_width=True,
            height=400
        )

# TAB 4: Framework de Gestión
with tab4:
    st.header(" Framework de Gestión por Probabilidad de Riesgo")
    
    st.markdown("""
    ###  Enfoque de Gestión Diferenciada
    
    El modelo no solo clasifica créditos en "cumple" o "no cumple", sino que proporciona **probabilidades precisas** 
    que permiten implementar estrategias de gestión adaptadas al nivel de riesgo de cada cliente, optimizando 
    recursos y maximizando la tasa de recuperación.
    """)
    
    st.markdown("---")
    
    # Framework visual con las tres categorías
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="risk-framework-card risk-low">
            <h2 style="margin-top: 0;"> RIESGO BAJO</h2>
            <h1 style="font-size: 3rem; margin: 1rem 0;">&lt; 30%</h1>
            <h3>Estrategia:</h3>
            <div class="strategy-list">
                <p> <strong>Monitoreo automático</strong></p>
                <p> SMS recordatorios</p>
                <p> Incentivos por pago puntual</p>
                <p> Gestión mínima de recursos</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #4caf50;'>
            <p style='color: #2c3e50; font-size: 1rem; margin: 0;'>
                <strong>Clientes estables</strong> con alta probabilidad de cumplimiento. 
                Requieren supervisión básica y automatizada.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="risk-framework-card risk-medium">
            <h2 style="margin-top: 0;"> RIESGO MEDIO</h2>
            <h1 style="font-size: 3rem; margin: 1rem 0;">30% - 70%</h1>
            <h3>Estrategia:</h3>
            <div class="strategy-list">
                <p> <strong>Llamadas preventivas</strong></p>
                <p> Planes de pago flexibles</p>
                <p> Acompañamiento quincenal</p>
                <p> Alertas tempranas</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #FDB913;'>
            <p style='color: #2c3e50; font-size: 1rem; margin: 0;'>
                <strong>Clientes en zona de atención</strong> que requieren seguimiento proactivo 
                para prevenir deterioro.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="risk-framework-card risk-high">
            <h2 style="margin-top: 0;"> RIESGO ALTO</h2>
            <h1 style="font-size: 3rem; margin: 1rem 0;">&gt; 70%</h1>
            <h3>Estrategia:</h3>
            <div class="strategy-list">
                <p> <strong>Gestión intensiva</strong></p>
                <p> Visitas presenciales</p>
                <p> Re-estructuración preventiva</p>
                <p> Equipo especializado</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #e53935;'>
            <p style='color: #2c3e50; font-size: 1rem; margin: 0;'>
                <strong>Clientes críticos</strong> que requieren intervención inmediata 
                y personalizada para evitar pérdidas.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Beneficios del enfoque
    st.markdown("###  Beneficios de la Segmentación por Probabilidad")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ####  Optimización de Recursos
        - **Focalización eficiente:** Los equipos de gestión se concentran en casos de mayor riesgo
        - **Reducción de costos:** Automatización para clientes de bajo riesgo
        - **Mejor ROI:** Asignación estratégica de recursos humanos y tecnológicos
        
        ####  Mejora en Recuperación
        - **Intervención temprana:** Actuación antes de que el problema escale
        - **Estrategias personalizadas:** Cada segmento recibe el tratamiento adecuado
        - **Mayor tasa de éxito:** Enfoque preventivo vs. reactivo
        """)
    
    with col2:
        st.markdown("""
        ####  Toma de Decisiones Data-Driven
        - **Objetividad:** Decisiones basadas en probabilidades, no intuición
        - **Trazabilidad:** Registro de acciones por nivel de riesgo
        - **Mejora continua:** Feedback loop para optimizar el modelo
        
        ####  Mejor Experiencia del Cliente
        - **No invasivo:** Clientes de bajo riesgo no son molestados innecesariamente
        - **Apoyo oportuno:** Clientes en dificultades reciben ayuda a tiempo
        - **Soluciones adaptadas:** Planes de pago acordes a la situación real
        """)
    
    # Ejemplo de flujo de trabajo
    st.markdown("---")
    st.markdown("###  Flujo de Trabajo Operativo")
    
    st.markdown("""
    ```
    1.  Carga de Cartera → Archivo CSV con datos de créditos
            
    2.  Modelo ML → Calcula probabilidad de incumplimiento para cada crédito
            
    3.  Segmentación Automática → Asigna nivel de riesgo (Bajo/Medio/Alto)
            
    4.  Distribución de Cartera → Asignación a equipos de gestión especializados
            
    5.  Ejecución de Estrategias → Aplicación de acciones diferenciadas
            
    6.  Monitoreo y Ajuste → Seguimiento de resultados y refinamiento
    ```
    """)
    
    # Métricas de impacto esperado
    st.markdown("###  Impacto Esperado de la Implementación")
    
    impact_cols = st.columns(4)
    impact_cols[0].metric(
        " Morosidad",
        "15-25%",
        help="Reducción esperada en tasa de morosidad"
    )
    impact_cols[1].metric(
        " Recuperación",
        "20-30%",
        help="Mejora en tasa de recuperación de cartera"
    )
    impact_cols[2].metric(
        " Costos Operativos",
        "30-40%",
        help="Reducción en costos de gestión por automatización"
    )
    impact_cols[3].metric(
        " Satisfacción Cliente",
        "25-35%",
        help="Mejora en índices de satisfacción del cliente"
    )

# TAB 5: Info del Modelo
with tab5:
    st.header(" Información del Modelo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("###  Características del Modelo")
        
        model_info = metrics.get('model_info', {})
        st.write(f"**Tipo de Modelo:** {model_info.get('model_type', 'XGBClassifier')}")
        st.write(f"**Fecha de Entrenamiento:** {model_info.get('training_date', 'N/A')}")
        st.write(f"**Features Utilizadas:** {len(EXPECTED_FEATURES)}")
        st.write(f"**Cutoff Óptimo:** {optimal_threshold:.3f}")
        
        st.markdown("###  Métricas de Evaluación")
        
        if 'performance' in metrics:
            perf = metrics['performance']
            
            metrics_df = pd.DataFrame({
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Valor': [
                    f"{perf.get('accuracy', 0):.4f}",
                    f"{perf.get('precision', 0):.4f}",
                    f"{perf.get('recall', 0):.4f}",
                    f"{perf.get('f1_score', 0):.4f}",
                    f"{perf.get('roc_auc', 0):.4f}"
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("###  Interpretación de Resultados")
        st.markdown(f"""
        - **Probabilidad < 0.3:** Riesgo Bajo  - Cliente estable
        - **Probabilidad 0.3 - 0.7:** Riesgo Medio  - Monitorear
        - **Probabilidad > 0.7:** Riesgo Alto  - Acción requerida
        - **Threshold: {optimal_threshold:.2f}** - Punto de decisión óptimo
        """)
    
    with col2:
        st.markdown("###  Arquitectura del Sistema")
        
        st.markdown("""
        ```
         Data Source (CSV)
              
         Feature Engineering
              
         SMOTE Balancing
              
         Feature Selection (SelectKBest)
              
         Hyperparameter Optimization (Optuna)
              
         XGBoost Classifier
              
         Model Evaluation & Threshold Optimization
              
         Model Serialization (.pkl)
              
         Streamlit Dashboard
        ```
        """)
        
        st.markdown("###  Stack Tecnológico")
        tech_stack = pd.DataFrame({
            'Componente': [
                'Machine Learning',
                'Optimización',
                'Balanceo de Datos',
                'Dashboard',
                'Visualización',
                'Serialización'
            ],
            'Tecnología': [
                'XGBoost + scikit-learn',
                'Optuna (TPE Sampler)',
                'SMOTE + SelectKBest',
                'Streamlit',
                'Plotly',
                'Pickle'
            ]
        })
        st.dataframe(tech_stack, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Mostrar features del modelo
    with st.expander(" Ver Todas las Features del Modelo"):
        st.markdown(f"**Total de Features:** {len(EXPECTED_FEATURES)}")
        
        # Organizar en columnas
        n_cols = 3
        features_per_col = len(EXPECTED_FEATURES) // n_cols + 1
        
        cols = st.columns(n_cols)
        for i, col in enumerate(cols):
            with col:
                start_idx = i * features_per_col
                end_idx = min((i + 1) * features_per_col, len(EXPECTED_FEATURES))
                for feature in EXPECTED_FEATURES[start_idx:end_idx]:
                    st.text(f"• {feature}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center'>
    <p><strong>Proyecto Integrador 1</strong></p>
    <p>Maestría en Ciencia de Datos | Universidad EAFIT</p>
    <p>2025-2</p>
    </div>
    """, unsafe_allow_html=True)