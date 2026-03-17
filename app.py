import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
from datetime import datetime
from fpdf import FPDF

warnings.filterwarnings("ignore")

def gerar_pdf(df_ranking):
    # 'L' para Landscape (Paisagem) - A4 tem 297mm de largura
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    
    # Título do Relatório
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Relatorio de Score de Absenteismo - Previsao 90 Dias", ln=True, align='C')
    pdf.ln(5)
    
    # Configuração do Cabeçalho
    pdf.set_font("Arial", "B", 8)
    pdf.set_fill_color(30, 58, 95) # Azul Escuro
    pdf.set_text_color(255, 255, 255) # Texto Branco
    
    # Nomes das colunas e larguras ajustadas (Total: 275mm)
    cols = ["Pos", "Empregado", "Score", "Risco", "Previstos", "Grupo CID", "Total", "Dias", "6m", "S/ Ates.", "Peso"]
    widths = [10, 25, 20, 25, 25, 50, 20, 20, 20, 30, 15] 
    
    for i, col_name in enumerate(cols):
        pdf.cell(widths[i], 10, col_name, border=1, align='C', fill=True)
    pdf.ln()
    
    # Configuração do corpo da tabela
    pdf.set_font("Arial", "", 9) # Aumentei um pouco a fonte para leitura
    pdf.set_text_color(0, 0, 0)
    
    for index, row in df_ranking.iterrows():
        # 1. Limpeza de Emojis e Texto de Risco
        texto_risco = str(row["Nível de risco"]).replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")
        
        # 2. Definição de cores de fundo por risco
        if "Alto" in texto_risco:
            pdf.set_fill_color(255, 210, 210) # Vermelho suave
        elif "Médio" in texto_risco:
            pdf.set_fill_color(255, 255, 210) # Amarelo suave
        else:
            pdf.set_fill_color(210, 255, 210) # Verde suave

        # 3. Formatação dos números (O SEGREDO PARA NÃO ESCANGALHAR)
        score_formatado     = f"{float(row['Score']):.1f}"
        previsto_formatado  = f"{float(row['Previstos (90d)']):.1f}"
        peso_formatado      = f"{float(row['Peso CID']):.1f}"
        total_atestados     = str(int(row['Total atestados']))
        dias_afastados      = str(int(row['Dias afastados']))
        atestados_6m        = str(int(row['Atestados (6m)']))
        dias_sem_atestado   = str(int(row['Dias s/ atestado']))

        # 4. Renderização das células com valores limpos
        pdf.cell(widths[0], 8, str(index), border=1, align='C', fill=True)
        pdf.cell(widths[1], 8, str(row["Empregado"]), border=1, align='C', fill=True)
        pdf.cell(widths[2], 8, score_formatado, border=1, align='C', fill=True)
        pdf.cell(widths[3], 8, texto_risco, border=1, align='C', fill=True)
        pdf.cell(widths[4], 8, previsto_formatado, border=1, align='C', fill=True)
        pdf.cell(widths[5], 8, str(row["Grupo CID"])[:25], border=1, align='L', fill=True) # Limita texto longo
        pdf.cell(widths[6], 8, total_atestados, border=1, align='C', fill=True)
        pdf.cell(widths[7], 8, dias_afastados, border=1, align='C', fill=True)
        pdf.cell(widths[8], 8, atestados_6m, border=1, align='C', fill=True)
        pdf.cell(widths[9], 8, dias_sem_atestado, border=1, align='C', fill=True)
        pdf.cell(widths[10], 8, peso_formatado, border=1, align='C', fill=True)
        pdf.ln()
        
    return bytes(pdf.output())

# ── Login ─────────────────────────────────────────────────────────────────────
if "logado" not in st.session_state:
    st.session_state.logado = False

if not st.session_state.logado:
    st.title("🔒 Acesso Restrito")
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if usuario == "rhli" and senha == "Rhli@2026":
            st.session_state.logado = True
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")
    st.stop()

if st.sidebar.button("Sair"):
    st.session_state.logado = False
    st.rerun()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Score de Risco de Absenteísmo",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
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
    col1, col2, col3 = st.columns([1, 2, 1])
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
GRUPO_CID = {
    "F": {"grupo": "Mental/Comportamental",  "peso": 4.0},
    "M": {"grupo": "Musculoesquelético",     "peso": 3.5},
    "C": {"grupo": "Neoplasia",              "peso": 3.5},
    "D": {"grupo": "Neoplasia/Sangue",       "peso": 3.5},
    "I": {"grupo": "Cardiovascular",         "peso": 3.0},
    "G": {"grupo": "Neurológico",            "peso": 3.0},
    "E": {"grupo": "Endócrino/Metabólico",   "peso": 2.5},
    "N": {"grupo": "Geniturinário",          "peso": 2.0},
    "K": {"grupo": "Digestivo",              "peso": 2.0},
    "H": {"grupo": "Olhos/Ouvidos",          "peso": 2.0},
    "L": {"grupo": "Dermatológico",          "peso": 1.5},
    "S": {"grupo": "Trauma/Lesão",           "peso": 1.5},
    "J": {"grupo": "Respiratório",           "peso": 1.5},
    "R": {"grupo": "Sintomas Inespecíficos", "peso": 1.5},
    "A": {"grupo": "Infecciosa",             "peso": 1.0},
    "B": {"grupo": "Infecciosa",             "peso": 1.0},
    "Z": {"grupo": "Preventivo/Exame",       "peso": 0.5},
}

def get_cid_info(cid):
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
df["CID"] = df["CID"].astype(str).str.strip().str.upper()
df[["grupo_cid", "peso_cid"]] = df["CID"].apply(lambda c: pd.Series(get_cid_info(c)))
hoje = pd.Timestamp(datetime.now().date())

# ── Métricas principais ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Total de empregados", df["MAT"].nunique())
col2.metric("Total de atestados", len(df))
col3.metric("Dias afastados", int(df["DIAS"].sum()))

# ── Janela temporal ───────────────────────────────────────────────────────────
JANELA_IDEAL = 90
span_dias = (hoje - df["DATA"].min()).days
if span_dias < JANELA_IDEAL * 2:
    JANELA_DIAS = max(7, int(span_dias * 0.35))
    st.warning(
        f"⚠️ O histórico total abrange apenas **{span_dias} dias**. "
        f"Para previsões em 90 dias é recomendável ao menos 180 dias de dados. "
        f"A janela de previsão foi ajustada automaticamente para **{JANELA_DIAS} dias**."
    )
else:
    JANELA_DIAS = JANELA_IDEAL

data_corte = hoje - pd.Timedelta(days=JANELA_DIAS)

# ── Histórico e futuro ─────────────────────────────────────────────────────────
todos_empregados = pd.DataFrame(df["MAT"].unique(), columns=["MAT"])
historico = df[df["DATA"] < data_corte].copy()
historico = todos_empregados.merge(historico, on="MAT", how="left")
historico["DIAS"] = historico["DIAS"].fillna(0)
historico["CID"]  = historico["CID"].fillna("Z")
historico["DATA"] = historico["DATA"].fillna(df["DATA"].min())
historico["peso_cid"] = historico.apply(lambda row: get_cid_info(row["CID"])[1], axis=1)
historico["grupo_cid"] = historico.apply(lambda row: get_cid_info(row["CID"])[0], axis=1)

futuro = df[(df["DATA"] >= data_corte) & (df["DATA"] <= hoje)].copy()

# ── Feature Engineering ───────────────────────────────────────────────────────
def build_features(source: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    freq     = source.groupby("MAT").size().reset_index(name="total_atestados")
    dias     = source.groupby("MAT")["DIAS"].sum().reset_index(name="dias_afastados")
    ultimo   = source.groupby("MAT")["DATA"].max().reset_index(name="data_ultimo")
    primeiro = source.groupby("MAT")["DATA"].min().reset_index(name="data_primeiro")

    ultimos_6m = source[source["DATA"] >= ref_date - pd.DateOffset(months=6)]
    ultimos_3m = source[source["DATA"] >= ref_date - pd.DateOffset(months=3)]

    freq_6m = ultimos_6m.groupby("MAT").size().reset_index(name="atestados_6m")
    freq_3m = ultimos_3m.groupby("MAT").size().reset_index(name="atestados_3m")
    dias_6m = ultimos_6m.groupby("MAT")["DIAS"].sum().reset_index(name="dias_afastados_6m")

    peso_max_cid = source.groupby("MAT")["peso_cid"].max().reset_index(name="peso_cid_max")

    source["peso_x_dias"] = source["peso_cid"] * source["DIAS"]
    peso_pond = (
        source.groupby("MAT")
        .apply(lambda g: g["peso_x_dias"].sum() / g["DIAS"].sum() if g["DIAS"].sum() > 0 else 1.0)
        .reset_index()
        .rename(columns={0: "peso_cid_ponderado"})
    )

    diversidade_cid = source.groupby("MAT")["grupo_cid"].nunique().reset_index(name="diversidade_cid")
    source["cid_cronico"] = (source["peso_cid"] >= 3.0).astype(int)
    tem_cronico = source.groupby("MAT")["cid_cronico"].max().reset_index(name="tem_cid_cronico")

    feat = freq.merge(dias, on="MAT", how="right")
    feat = feat.merge(ultimo, on="MAT", how="right")
    feat = feat.merge(primeiro, on="MAT", how="right")
    feat = feat.merge(freq_6m, on="MAT", how="left")
    feat = feat.merge(freq_3m, on="MAT", how="left")
    feat = feat.merge(dias_6m, on="MAT", how="left")
    feat = feat.merge(peso_max_cid, on="MAT", how="left")
    feat = feat.merge(peso_pond, on="MAT", how="left")
    feat = feat.merge(diversidade_cid, on="MAT", how="left")
    feat = feat.merge(tem_cronico, on="MAT", how="left")

    # Garantir que todos os empregados apareçam
    feat = todos_empregados.merge(feat, on="MAT", how="left").fillna({
        "dias_desde_ultimo": (ref_date - df["DATA"].min()).days,
        "total_atestados": 0,
        "dias_afastados": 0,
        "atestados_6m": 0,
        "atestados_3m": 0,
        "dias_afastados_6m": 0,
        "freq_mensal": 0,
        "media_dias_atestado": 0,
        "tendencia_recente": 0,
        "score_recencia": 0,
        "peso_cid_max": 0,
        "peso_cid_ponderado": 0,
        "diversidade_cid": 0,
        "tem_cid_cronico": 0,
    })

    feat["dias_desde_ultimo"] = (datetime.now() - feat["data_ultimo"]).dt.days
    feat["meses_historico"]   = ((feat["data_ultimo"] - feat["data_primeiro"]).dt.days / 30).clip(lower=1).fillna(1)
    feat["freq_mensal"]       = feat["total_atestados"] / feat["meses_historico"]
    feat["media_dias_atestado"] = feat["dias_afastados"] / feat["total_atestados"].replace(0,1)
    feat["tendencia_recente"] = feat["atestados_6m"] / (feat["total_atestados"] + 1)
    feat["score_recencia"]    = 1 / (feat["dias_desde_ultimo"] + 1)

    return feat

features = build_features(historico, data_corte)

# ── Merge com target futuro ───────────────────────────────────────────────────
target_real = (
    futuro.groupby("MAT")
    .agg(atestados_futuros=("MAT", "count"), dias_futuros=("DIAS", "sum"))
    .reset_index()
)
features = features.merge(target_real, on="MAT", how="left").fillna({
    "atestados_futuros": 0,
    "dias_futuros": 0
})

# ── Grupo CID principal
grupo_predominante = (
    historico.groupby("MAT")
    .apply(lambda g: g.loc[g["peso_cid"].idxmax(), "grupo_cid"])
    .reset_index(name="grupo_cid_principal")
)
features = features.merge(grupo_predominante, on="MAT", how="left")
features["grupo_cid_principal"] = features["grupo_cid_principal"].fillna("Não informado")

# ── Modelo XGBoost ───────────────────────────────────────────────────────────
feature_cols = [
    "dias_desde_ultimo", "total_atestados", "dias_afastados",
    "atestados_6m", "atestados_3m", "dias_afastados_6m",
    "freq_mensal", "media_dias_atestado", "tendencia_recente",
    "score_recencia", "peso_cid_max", "peso_cid_ponderado",
    "diversidade_cid", "tem_cid_cronico",
]

X = features[feature_cols]
y = features["atestados_futuros"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(X_train, y_train)

pred_raw = model.predict(X)
scaler_final = MinMaxScaler(feature_range=(0, 100))
features["score_risco"] = scaler_final.fit_transform(pred_raw.reshape(-1,1)).flatten().round(1)
features["atestados_previstos"] = pred_raw.clip(min=0).round(1)

def classificar_risco(score):
    if score >= 70:   return "🔴 Alto"
    elif score >= 40: return "🟡 Médio"
    else:             return "🟢 Baixo"

features["nivel_risco"] = features["score_risco"].apply(classificar_risco)

# ── Ranking final ─────────────────────────────────────────────────────────────
ranking = features[[
    "MAT", "score_risco", "nivel_risco", "atestados_previstos",
    "grupo_cid_principal", "total_atestados", "dias_afastados",
    "atestados_6m", "dias_desde_ultimo", "peso_cid_max"
]].copy()

ranking["dias_afastados"] = ranking["dias_afastados"].fillna(0).astype(int)
ranking = ranking.sort_values("score_risco", ascending=False).reset_index(drop=True)
ranking.index = ranking.index + 1
ranking.index.name = "Posição"

ranking = ranking.rename(columns={
    "MAT": "Empregado",
    "score_risco": "Score",
    "nivel_risco": "Nível de risco",
    "atestados_previstos": "Previstos (90d)",
    "grupo_cid_principal": "Grupo CID",
    "total_atestados": "Total atestados",
    "dias_afastados": "Dias afastados",
    "atestados_6m": "Atestados (6m)",
    "dias_desde_ultimo": "Dias s/ atestado",
    "peso_cid_max": "Peso CID",
})

# --- BOTÃO DE EXPORTAÇÃO ---
with st.sidebar:
    st.divider()
    st.subheader("Exportar Dados")
    try:
        pdf_bytes = gerar_pdf(ranking)
        st.download_button(
            label="📄 Baixar Score em PDF",
            data=pdf_bytes,
            file_name=f"ranking_absenteismo_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Erro ao gerar PDF: {e}")

# ── Exibição ──────────────────────────────────────────────────────────────────
st.subheader("Score de Absenteísmo — Previsão para os próximos 90 dias")
st.info(
    f"🔬 **Metodologia:** features construídas com dados anteriores a "
    f"**{data_corte.strftime('%d/%m/%Y')}**. O modelo XGBoost prevê quantos "
    f"atestados cada empregado terá nos próximos **{JANELA_DIAS} dias**. "
    f"O score (0–100) é derivado diretamente dessa previsão."
)

col1, col2, col3 = st.columns(3)
col1.metric("🔴 Alto risco",  (features["nivel_risco"] == "🔴 Alto").sum())
col2.metric("🟡 Médio risco", (features["nivel_risco"] == "🟡 Médio").sum())
col3.metric("🟢 Baixo risco", (features["nivel_risco"] == "🟢 Baixo").sum())

st.divider()
altura_tabela = (len(ranking) + 1) * 35 + 10
st.dataframe(
    ranking.style
        .background_gradient(subset=["Score"], cmap="RdYlGn_r")
        .format({"Score": "{:.2f}", "Previstos (90d)": "{:.2f}", "Peso CID": "{:.2f}"}),
    use_container_width=True,
    height=altura_tabela,
)
