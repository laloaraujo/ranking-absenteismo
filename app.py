import streamlit as st
import pandas as pd
import numpy as np
import glob
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import warnings
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise ML · Afastamentos",
    page_icon="⬜",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 600; letter-spacing: 0.04em; }
    div[data-testid="metric-container"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------

st.set_page_config(page_title="Score de Risco de Absenteísmo", layout="wide")

st.title("Score de Risco de Absenteísmo em 90 Dias")

# ---------------------------------------------------

with st.sidebar:

    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col2:
        if os.path.exists("lalo.png"):
            st.image("lalo.png", use_container_width=True)

st.sidebar.markdown("""
<div style='text-align: center;'>
    <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Da planilha ao modelo de IA.</p>
    <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Coleta, tratamento e análise de dados com métodos de Machine Learning em Python.</p>
    <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Python · Numpy · Pandas · Streamlit · XGBoost · Scikit-Learn</p>
    <p style='color: #1e3a5f; font-weight: 600; margin: 6px 0 2px 0;'>Jorge Eduardo de Araujo Oliveira</p>
    <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Tecnólogo em Análise e Desenvolvimento de Sistemas</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# CARREGAR CSV DA PASTA
# ---------------------------------------------------

files = glob.glob("*.csv")

if len(files) == 0:
    st.error("Nenhum CSV encontrado na pasta.")
    st.stop()

dfs = []

for file in files:
    df = pd.read_csv(file, dtype={"MAT": str})
    df.columns = df.columns.str.strip().str.upper()
    dfs.append(df)

df = pd.concat(dfs)

df["MAT"] = df["MAT"].astype(str).str.zfill(6)
df["DATA"] = pd.to_datetime(df["DATA"], dayfirst=True)
df = df.sort_values(["MAT", "DATA"])

hoje = df["DATA"].max()

# ---------------------------------------------------
# MÉTRICAS PRINCIPAIS
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total de empregados", df["MAT"].nunique())
col2.metric("Total de atestados", len(df))
col3.metric("Dias afastados", int(df["DIAS"].sum()))

# ---------------------------------------------------
# FEATURE ENGINEERING ENRIQUECIDA
# ---------------------------------------------------

# Totais históricos
freq = df.groupby("MAT").size().reset_index(name="total_atestados")
dias = df.groupby("MAT")["DIAS"].sum().reset_index(name="dias_afastados")
ultimo = df.groupby("MAT")["DATA"].max().reset_index(name="data_ultimo")
primeiro = df.groupby("MAT")["DATA"].min().reset_index(name="data_primeiro")

# Atestados nos últimos 6 meses
ultimos_6m = df[df["DATA"] >= hoje - pd.DateOffset(months=6)]
freq_6m = ultimos_6m.groupby("MAT").size().reset_index(name="atestados_6m")

# Atestados nos últimos 3 meses
ultimos_3m = df[df["DATA"] >= hoje - pd.DateOffset(months=3)]
freq_3m = ultimos_3m.groupby("MAT").size().reset_index(name="atestados_3m")

# Dias afastados nos últimos 6 meses
dias_6m = ultimos_6m.groupby("MAT")["DIAS"].sum().reset_index(name="dias_afastados_6m")

# Montar base de features
features = freq.merge(dias, on="MAT")
features = features.merge(ultimo, on="MAT")
features = features.merge(primeiro, on="MAT")
features = features.merge(freq_6m, on="MAT", how="left")
features = features.merge(freq_3m, on="MAT", how="left")
features = features.merge(dias_6m, on="MAT", how="left")
features = features.fillna(0)

# Features derivadas
features["dias_desde_ultimo"] = (hoje - features["data_ultimo"]).dt.days

# Tempo de histórico em meses (mínimo 1 para evitar divisão por zero)
features["meses_historico"] = (
    (features["data_ultimo"] - features["data_primeiro"]).dt.days / 30
).clip(lower=1)

# Frequência mensal de atestados
features["freq_mensal"] = features["total_atestados"] / features["meses_historico"]

# Média de dias por atestado
features["media_dias_atestado"] = (
    features["dias_afastados"] / features["total_atestados"]
)

# Tendência: proporção recente vs histórica (>1 = piorando)
features["tendencia_recente"] = (
    features["atestados_6m"] / (features["total_atestados"] + 1)
)

# Score de recência: quanto mais recente o último atestado, maior o risco
features["score_recencia"] = 1 / (features["dias_desde_ultimo"] + 1)

# ---------------------------------------------------
# TARGET CONTÍNUO PONDERADO
# ---------------------------------------------------

# Normaliza cada componente individualmente antes de ponderar
scaler_tmp = MinMaxScaler()

componentes = pd.DataFrame({
    "c1": features["total_atestados"],
    "c2": features["dias_afastados"],
    "c3": features["atestados_6m"],
    "c4": features["freq_mensal"],
    "c5": features["tendencia_recente"],
    "c6": features["score_recencia"],
})

componentes_norm = pd.DataFrame(
    scaler_tmp.fit_transform(componentes),
    columns=componentes.columns
)

# Pesos: histórico geral + recência + tendência
features["target"] = (
    componentes_norm["c1"] * 0.25 +  # total de atestados
    componentes_norm["c2"] * 0.20 +  # dias afastados
    componentes_norm["c3"] * 0.20 +  # atestados últimos 6m
    componentes_norm["c4"] * 0.15 +  # frequência mensal
    componentes_norm["c5"] * 0.10 +  # tendência recente
    componentes_norm["c6"] * 0.10    # recência
) * 100  # escala 0-100

# ---------------------------------------------------
# TREINO DO MODELO
# ---------------------------------------------------

feature_cols = [
    "dias_desde_ultimo",
    "total_atestados",
    "dias_afastados",
    "atestados_6m",
    "atestados_3m",
    "dias_afastados_6m",
    "freq_mensal",
    "media_dias_atestado",
    "tendencia_recente",
    "score_recencia",
]

X = features[feature_cols]
y = features["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model.fit(X_train, y_train)

# Score final: predição do modelo, re-normalizada para 0-100
pred = model.predict(X)

scaler_final = MinMaxScaler(feature_range=(0, 100))
features["score_risco"] = scaler_final.fit_transform(
    pred.reshape(-1, 1)
).flatten().round(1)

# Classificação de risco
def classificar_risco(score):
    if score >= 70:
        return "🔴 Alto"
    elif score >= 40:
        return "🟡 Médio"
    else:
        return "🟢 Baixo"

features["nivel_risco"] = features["score_risco"].apply(classificar_risco)

# ---------------------------------------------------
# RANKING DE RISCO
# ---------------------------------------------------

st.subheader("Dados Tabulados e Ranking de Absenteísmo")

resultado = features[[
    "MAT",
    "score_risco",
    "nivel_risco",
    "total_atestados",
    "dias_afastados",
    "atestados_6m",
    "dias_desde_ultimo",
]].copy()

ranking = resultado.sort_values("score_risco", ascending=False).reset_index(drop=True)
ranking.index = ranking.index + 1
ranking.index.name = "posição"

ranking = ranking.rename(columns={
    "MAT": "Empregado",
    "score_risco": "Score de risco",
    "nivel_risco": "Nível de risco",
    "total_atestados": "Total de atestados",
    "dias_afastados": "Dias afastados",
    "atestados_6m": "Atestados (6m)",
    "dias_desde_ultimo": "Dias desde último atestado",
})

st.dataframe(
    ranking.style.background_gradient(
        subset=["Score de risco"],
        cmap="RdYlGn_r"
    ),
    use_container_width=True,
     height=(len(ranking) + 1) * 35 + 3
)

# ---------------------------------------------------
# MÉTRICAS DE RISCO
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

alto = (features["nivel_risco"] == "🔴 Alto").sum()
medio = (features["nivel_risco"] == "🟡 Médio").sum()
baixo = (features["nivel_risco"] == "🟢 Baixo").sum()

col1.metric("🔴 Alto risco", alto)
col2.metric("🟡 Médio risco", medio)
col3.metric("🟢 Baixo risco", baixo)
