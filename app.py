import streamlit as st
import pandas as pd
import numpy as np
import glob
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings("ignore")

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

st.title("Score de Risco de Absenteísmo em 90 Dias")

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

# ── Classificação de risco por capítulo CID-10 ───────────────────────────────
# Peso de recorrência: quanto maior, maior a tendência de se repetir
# Baseado em evidências clínicas de cronicidade e reincidência
GRUPO_CID = {
    # Capítulo F — Transtornos mentais e comportamentais (alto risco de recorrência)
    "F": {"grupo": "Mental/Comportamental", "peso": 4.0},

    # Capítulo M — Musculoesquelético (alto risco, especialmente crônico)
    "M": {"grupo": "Musculoesquelético", "peso": 3.5},

    # Capítulo I — Circulatório (risco moderado-alto, doenças crônicas)
    "I": {"grupo": "Cardiovascular", "peso": 3.0},

    # Capítulo G — Neurológico (moderado-alto)
    "G": {"grupo": "Neurológico", "peso": 3.0},

    # Capítulo E — Endócrino/Metabólico (diabetes, obesidade — crônico)
    "E": {"grupo": "Endócrino/Metabólico", "peso": 2.5},

    # Capítulo N — Geniturinário (moderado)
    "N": {"grupo": "Geniturinário", "peso": 2.0},

    # Capítulo K — Digestivo (moderado)
    "K": {"grupo": "Digestivo", "peso": 2.0},

    # Capítulo H — Olhos e ouvidos (moderado)
    "H": {"grupo": "Olhos/Ouvidos", "peso": 2.0},

    # Capítulo L — Pele (moderado-baixo)
    "L": {"grupo": "Dermatológico", "peso": 1.5},

    # Capítulo S — Lesões/Traumas (baixo — geralmente pontual)
    "S": {"grupo": "Trauma/Lesão", "peso": 1.5},

    # Capítulo J — Respiratório (baixo — geralmente agudo)
    "J": {"grupo": "Respiratório", "peso": 1.5},

    # Capítulo R — Sintomas inespecíficos
    "R": {"grupo": "Sintomas Inespecíficos", "peso": 1.5},

    # Capítulo A/B — Infecciosas (baixo — geralmente agudo)
    "A": {"grupo": "Infecciosa", "peso": 1.0},
    "B": {"grupo": "Infecciosa", "peso": 1.0},

    # Capítulo Z — Consultas/exames (baixíssimo risco)
    "Z": {"grupo": "Preventivo/Exame", "peso": 0.5},

    # Capítulo C/D — Neoplasias (caso especial — afastamentos longos)
    "C": {"grupo": "Neoplasia", "peso": 3.5},
    "D": {"grupo": "Neoplasia/Sangue", "peso": 3.5},
}

def get_cid_info(cid):
    """Retorna grupo e peso de risco de recorrência para um CID."""
    if pd.isna(cid) or cid == "":
        return "Não informado", 1.0
    cid = str(cid).strip().upper()
    letra = cid[0] if cid else "?"
    info = GRUPO_CID.get(letra, {"grupo": "Outro", "peso": 1.0})
    return info["grupo"], info["peso"]

# ── Carregar CSVs ─────────────────────────────────────────────────────────────
files = glob.glob("*.csv")

if len(files) == 0:
    st.error("Nenhum CSV encontrado na pasta.")
    st.stop()

dfs = []
for file in files:
    df_tmp = pd.read_csv(file, dtype={"MAT": str})
    df_tmp.columns = df_tmp.columns.str.strip().str.upper()
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
df["MAT"] = df["MAT"].astype(str).str.zfill(6)
df["DATA"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["DATA"])
df = df.sort_values(["MAT", "DATA"])

# Normalizar CID (uppercase, sem espaços)
df["CID"] = df["CID"].astype(str).str.strip().str.upper()

hoje = df["DATA"].max()

# ── Enriquecer com info de CID ────────────────────────────────────────────────
df[["grupo_cid", "peso_cid"]] = df["CID"].apply(
    lambda c: pd.Series(get_cid_info(c))
)

# ── Métricas principais ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Total de empregados", df["MAT"].nunique())
col2.metric("Total de atestados", len(df))
col3.metric("Dias afastados", int(df["DIAS"].sum()))

# ── Feature Engineering ───────────────────────────────────────────────────────

# Totais históricos
freq       = df.groupby("MAT").size().reset_index(name="total_atestados")
dias       = df.groupby("MAT")["DIAS"].sum().reset_index(name="dias_afastados")
ultimo     = df.groupby("MAT")["DATA"].max().reset_index(name="data_ultimo")
primeiro   = df.groupby("MAT")["DATA"].min().reset_index(name="data_primeiro")

# Recortes temporais
ultimos_6m = df[df["DATA"] >= hoje - pd.DateOffset(months=6)]
ultimos_3m = df[df["DATA"] >= hoje - pd.DateOffset(months=3)]
freq_6m    = ultimos_6m.groupby("MAT").size().reset_index(name="atestados_6m")
freq_3m    = ultimos_3m.groupby("MAT").size().reset_index(name="atestados_3m")
dias_6m    = ultimos_6m.groupby("MAT")["DIAS"].sum().reset_index(name="dias_afastados_6m")

# ── Features de CID ───────────────────────────────────────────────────────────
# Peso máximo de CID por funcionário (pior diagnóstico registrado)
peso_max_cid = df.groupby("MAT")["peso_cid"].max().reset_index(name="peso_cid_max")

# Peso médio ponderado pelos dias (diagnósticos que geraram mais dias pesam mais)
df["peso_x_dias"] = df["peso_cid"] * df["DIAS"]
peso_pond = (
    df.groupby("MAT")
    .apply(lambda g: g["peso_x_dias"].sum() / g["DIAS"].sum() if g["DIAS"].sum() > 0 else 1.0)
    .reset_index(name="peso_cid_ponderado")
)

# Diversidade de grupos CID (muitos grupos diferentes = problema sistêmico)
diversidade_cid = (
    df.groupby("MAT")["grupo_cid"]
    .nunique()
    .reset_index(name="diversidade_cid")
)

# CID crônico: tem algum diagnóstico com peso >= 3.0?
df["cid_cronico"] = (df["peso_cid"] >= 3.0).astype(int)
tem_cronico = df.groupby("MAT")["cid_cronico"].max().reset_index(name="tem_cid_cronico")

# ── Montar base de features ───────────────────────────────────────────────────
features = freq.merge(dias, on="MAT")
features = features.merge(ultimo, on="MAT")
features = features.merge(primeiro, on="MAT")
features = features.merge(freq_6m, on="MAT", how="left")
features = features.merge(freq_3m, on="MAT", how="left")
features = features.merge(dias_6m, on="MAT", how="left")
features = features.merge(peso_max_cid, on="MAT", how="left")
features = features.merge(peso_pond, on="MAT", how="left")
features = features.merge(diversidade_cid, on="MAT", how="left")
features = features.merge(tem_cronico, on="MAT", how="left")
features = features.fillna(0)

# Features derivadas
features["dias_desde_ultimo"] = (hoje - features["data_ultimo"]).dt.days

features["meses_historico"] = (
    (features["data_ultimo"] - features["data_primeiro"]).dt.days / 30
).clip(lower=1)

features["freq_mensal"] = features["total_atestados"] / features["meses_historico"]

features["media_dias_atestado"] = (
    features["dias_afastados"] / features["total_atestados"]
)

features["tendencia_recente"] = (
    features["atestados_6m"] / (features["total_atestados"] + 1)
)

features["score_recencia"] = 1 / (features["dias_desde_ultimo"] + 1)

# ── Target ponderado com CID ──────────────────────────────────────────────────
scaler_tmp = MinMaxScaler()

componentes = pd.DataFrame({
    "c1": features["total_atestados"],
    "c2": features["dias_afastados"],
    "c3": features["atestados_6m"],
    "c4": features["freq_mensal"],
    "c5": features["tendencia_recente"],
    "c6": features["score_recencia"],
    "c7": features["peso_cid_max"],          # pior diagnóstico
    "c8": features["peso_cid_ponderado"],    # diagnósticos ponderados por dias
    "c9": features["diversidade_cid"],       # variedade de problemas
    "c10": features["tem_cid_cronico"],      # tem doença crônica?
})

componentes_norm = pd.DataFrame(
    scaler_tmp.fit_transform(componentes),
    columns=componentes.columns
)

# Pesos revisados — CID representa 30% do score total
features["target"] = (
    componentes_norm["c1"]  * 0.18 +   # total de atestados
    componentes_norm["c2"]  * 0.15 +   # dias afastados
    componentes_norm["c3"]  * 0.15 +   # atestados últimos 6m
    componentes_norm["c4"]  * 0.10 +   # frequência mensal
    componentes_norm["c5"]  * 0.07 +   # tendência recente
    componentes_norm["c6"]  * 0.05 +   # recência
    componentes_norm["c7"]  * 0.10 +   # peso CID máximo
    componentes_norm["c8"]  * 0.10 +   # peso CID ponderado por dias
    componentes_norm["c9"]  * 0.05 +   # diversidade de grupos
    componentes_norm["c10"] * 0.05     # presença de CID crônico
) * 100

# ── Treino do modelo ──────────────────────────────────────────────────────────
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
    "peso_cid_max",
    "peso_cid_ponderado",
    "diversidade_cid",
    "tem_cid_cronico",
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

pred = model.predict(X)
scaler_final = MinMaxScaler(feature_range=(0, 100))
features["score_risco"] = scaler_final.fit_transform(
    pred.reshape(-1, 1)
).flatten().astype(float).round(3)

features["score_risco"] = features["score_risco"].round(3)

def classificar_risco(score):
    if score >= 70:
        return "🔴 Alto"
    elif score >= 40:
        return "🟡 Médio"
    else:
        return "🟢 Baixo"

features["nivel_risco"] = features["score_risco"].apply(classificar_risco)

# ── Enriquecer resultado com grupo CID predominante ───────────────────────────
grupo_predominante = (
    df.groupby("MAT")
    .apply(lambda g: g.loc[g["peso_cid"].idxmax(), "grupo_cid"])
    .reset_index(name="grupo_cid_principal")
)
features = features.merge(grupo_predominante, on="MAT", how="left")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Ranking de Risco", "🔬 Análise por CID", "📊 Importância das Features"])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Ranking de Absenteísmo")

    resultado = features[[
        "MAT",
        "score_risco",
        "nivel_risco",
        "grupo_cid_principal",
        "total_atestados",
        "dias_afastados",
        "atestados_6m",
        "dias_desde_ultimo",
        "peso_cid_max",
    ]].copy()

    ranking = resultado.sort_values("score_risco", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    ranking.index.name = "posição"

    ranking = ranking.rename(columns={
        "MAT": "Empregado",
        "score_risco": "Score de risco",
        "nivel_risco": "Nível de risco",
        "grupo_cid_principal": "Grupo CID principal",
        "total_atestados": "Total atestados",
        "dias_afastados": "Dias afastados",
        "atestados_6m": "Atestados (6m)",
        "dias_desde_ultimo": "Dias desde último",
        "peso_cid_max": "Peso CID",
    })

    st.dataframe(
        ranking.style
            .background_gradient(subset=["Score de risco"], cmap="RdYlGn_r")
            .format({"Score de risco": "{:.3f}"}),  # ← adicione esta linha
        use_container_width=True,
        height=min((len(ranking) + 1) * 35 + 3, 600)
)

    col1, col2, col3 = st.columns(3)
    alto  = (features["nivel_risco"] == "🔴 Alto").sum()
    medio = (features["nivel_risco"] == "🟡 Médio").sum()
    baixo = (features["nivel_risco"] == "🟢 Baixo").sum()
    col1.metric("🔴 Alto risco", alto)
    col2.metric("🟡 Médio risco", medio)
    col3.metric("🟢 Baixo risco", baixo)

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Distribuição por Grupo CID")

    # Frequência de atestados por grupo CID
    grupo_freq = (
        df.groupby("grupo_cid")
        .agg(
            atestados=("MAT", "count"),
            dias_totais=("DIAS", "sum"),
            funcionarios=("MAT", "nunique"),
        )
        .reset_index()
        .sort_values("atestados", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            grupo_freq,
            x="atestados",
            y="grupo_cid",
            orientation="h",
            title="Atestados por grupo CID",
            color="atestados",
            color_continuous_scale="Reds",
            labels={"grupo_cid": "Grupo", "atestados": "Nº de atestados"},
        )
        fig1.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            grupo_freq,
            x="dias_totais",
            y="grupo_cid",
            orientation="h",
            title="Dias afastados por grupo CID",
            color="dias_totais",
            color_continuous_scale="Oranges",
            labels={"grupo_cid": "Grupo", "dias_totais": "Total de dias"},
        )
        fig2.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Peso de risco de recorrência por grupo CID")
    st.markdown("""
    O peso de recorrência reflete a tendência clínica de cada grupo diagnóstico
    se repetir ou se cronificar. É utilizado como feature no modelo de ranking.
    """)

    peso_ref = pd.DataFrame([
        {"Grupo": v["grupo"], "Peso de recorrência": v["peso"], "Capítulo CID": k}
        for k, v in GRUPO_CID.items()
    ]).sort_values("Peso de recorrência", ascending=False)

    fig3 = px.bar(
        peso_ref,
        x="Peso de recorrência",
        y="Grupo",
        orientation="h",
        color="Peso de recorrência",
        color_continuous_scale="RdYlGn_r",
        title="Peso de recorrência clínica por grupo CID-10",
        text="Peso de recorrência",
    )
    fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig3.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Importância das Features (XGBoost)")

    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importância": model.feature_importances_,
    }).sort_values("Importância", ascending=True)

    nomes_pt = {
        "dias_desde_ultimo":     "Dias desde último atestado",
        "total_atestados":       "Total de atestados",
        "dias_afastados":        "Total de dias afastados",
        "atestados_6m":          "Atestados (últimos 6m)",
        "atestados_3m":          "Atestados (últimos 3m)",
        "dias_afastados_6m":     "Dias afastados (6m)",
        "freq_mensal":           "Frequência mensal",
        "media_dias_atestado":   "Média de dias por atestado",
        "tendencia_recente":     "Tendência recente",
        "score_recencia":        "Score de recência",
        "peso_cid_max":          "Peso CID máximo ★",
        "peso_cid_ponderado":    "Peso CID ponderado ★",
        "diversidade_cid":       "Diversidade de grupos CID ★",
        "tem_cid_cronico":       "Presença de CID crônico ★",
    }

    importances["Feature"] = importances["Feature"].map(nomes_pt)

    fig4 = px.bar(
        importances,
        x="Importância",
        y="Feature",
        orientation="h",
        color="Importância",
        color_continuous_scale="Blues",
        title="Importância de cada feature no modelo XGBoost (★ = features de CID)",
    )
    fig4.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    pct_cid = importances[importances["Feature"].str.contains("★")]["Importância"].sum()
    st.info(
        f"**Features de CID respondem por {pct_cid*100:.1f}% da importância total do modelo.**"
    )
