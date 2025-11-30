**Proyecto Integrador 1** - Maestría en Ciencia de Datos y Analítica  
**Universidad EAFIT** - 2025-2

## Descripción

Sistema de Machine Learning para predecir la probabilidad de pago de un cliente luego de una reestructuracion créditos financieros

## Equipo

- David Botero Londoño
- Jorge Alberto Giraldo Montoya
- Samuel Padierna Zapata

## Objetivo

Desarrollar un modelo predictivo que identifique clientes con alta probabilidad de no pago luego de una reestructuracion, permitiendo tomar acciones preventivas y reducir el riesgo crediticio.

## Dataset

- **Tamaño:** 852.136 registros
- **Features:** 40 variables financieras y demográficas
- **Target:** cumple_6m (binaria: 0/1)
- **Fuente:** Datos históricos de créditos financieros
- **Período:** 2022-2025

## 🏗️ Arquitectura
```
CSV Data → S3 Data Lake → SageMaker Training → Modelo XGBoost → Streamlit Dashboard
```

### Stack Tecnológico

- **Cloud:** AWS (S3, SageMaker)
- **ML Framework:** XGBoost, Scikit-learn
- **Optimización:** Optuna (hyperparameter tuning)
- **Balanceo:** SMOTE + EditedNearestNeighbours
- **Visualización:** Streamlit + Plotly
- **Lenguaje:** Python 3.9

## 📁 Estructura del Proyecto
```
proyecto-integrador-ml/
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb       # Análisis exploratorio
│   ├── 02_feature_engineering.ipynb
│   └── 03_entrenamiento.ipynb
├── models/                # Modelos entrenados
│   └── trained/
│       ├── best_model.pkl
│       └── model_metrics.json
├── deployment/            # Scripts de deployment
│   ├── app.py            # Streamlit dashboard
│   └── requirements.txt
├── data/                  # Datos
│   └── README.md         # Descripción de datos
├── docs/                  # Documentación
│   └── informe_final.pdf
└── README.md
```

## Quick Start

### Instalación
```bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/proyecto-integrador-ml.git
cd proyecto-integrador-ml

# Instalar dependencias
pip install -r deployment/requirements.txt
```

### Configurar AWS
```bash
# Crear archivo de secrets
mkdir .streamlit
nano .streamlit/secrets.toml
```
```toml
[aws]
aws_access_key_id = "YOUR_KEY"
aws_secret_access_key = "YOUR_SECRET"
aws_session_token = "YOUR_TOKEN"
region = "us-east-1"
```

### Ejecutar Dashboard
```bash
streamlit run deployment/app.py
```

## Resultados

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 85.2% |
| **ROC-AUC** | 0.902 |
| **Precision** | 83.4% |
| **Recall** | 81.7% |
| **F1-Score** | 82.5% |


## Metodología

### 1. Análisis Exploratorio (EDA)
- Análisis de distribuciones
- Detección de valores atípicos
- Correlación entre variables

### 2. Feature Engineering
- Selección de features (SelectKBest)
- Creación de variables derivadas
- Normalización y escalado

### 3. Balanceo de Datos
- SMOTE (Synthetic Minority Over-sampling)
- EditedNearestNeighbours (undersampling)

### 4. Entrenamiento
- Modelos evaluados: Random Forest, Gradient Boosting, XGBoost
- Optimización con Optuna (50 trials)
- Validación cruzada (5 folds)

### 5. Deployment
- Dashboard interactivo con Streamlit
- Predicción individual en tiempo real
- Análisis por lotes (CSV upload)

## Dashboard Features

- **Predicción Individual:** Formulario interactivo con gauge chart
- **Análisis por Lotes:** Upload CSV y descarga de resultados
- **Visualización:** Gráficos de distribución de riesgo
- **Métricas:** Performance del modelo en tiempo real

## Links

- **Dashboard en vivo:** [https://tu-app.streamlit.app](URL cuando despliegues)
- **S3 Bucket:** `s3://ml-reestructuraciones-029885540752`
- **Documentación:** [Ver docs/](docs/)

## Licencia

Este proyecto es de uso académico para la Maestría en Ciencia de Datos - EAFIT.

---

**Proyecto Integrador 1** | EAFIT 2025-2
