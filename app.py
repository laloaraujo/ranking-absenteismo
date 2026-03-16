import streamlit as st
import pandas as pd
import numpy as np
import glob
import tempfile
import os
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings

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

df[["grupo_cid", "peso_cid"]] = df["CID"].apply(
    lambda c: pd.Series(get_cid_info(c))
)

hoje = df["DATA"].max()

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
historico  = df[df["DATA"] < data_corte].copy()
futuro     = df[(df["DATA"] >= data_corte) & (df["DATA"] <= hoje)].copy()

if historico.empty:
    st.error("❌ Não há dados históricos suficientes para treinar o modelo.")
    st.stop()

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

    source = source.copy()
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

    feat = freq.merge(dias,           on="MAT")
    feat = feat.merge(ultimo,         on="MAT")
    feat = feat.merge(primeiro,       on="MAT")
    feat = feat.merge(freq_6m,        on="MAT", how="left")
    feat = feat.merge(freq_3m,        on="MAT", how="left")
    feat = feat.merge(dias_6m,        on="MAT", how="left")
    feat = feat.merge(peso_max_cid,   on="MAT", how="left")
    feat = feat.merge(peso_pond,      on="MAT", how="left")
    feat = feat.merge(diversidade_cid,on="MAT", how="left")
    feat = feat.merge(tem_cronico,    on="MAT", how="left")
    feat = feat.fillna(0)

    feat["dias_desde_ultimo"]   = (ref_date - feat["data_ultimo"]).dt.days
    feat["meses_historico"]     = ((feat["data_ultimo"] - feat["data_primeiro"]).dt.days / 30).clip(lower=1)
    feat["freq_mensal"]         = feat["total_atestados"] / feat["meses_historico"]
    feat["media_dias_atestado"] = feat["dias_afastados"]  / feat["total_atestados"]
    feat["tendencia_recente"]   = feat["atestados_6m"] / (feat["total_atestados"] + 1)
    feat["score_recencia"]      = 1 / (feat["dias_desde_ultimo"] + 1)

    return feat

features = build_features(historico, data_corte)

target_real = (
    futuro.groupby("MAT")
    .agg(atestados_futuros=("MAT", "count"), dias_futuros=("DIAS", "sum"))
    .reset_index()
)

features = features.merge(target_real, on="MAT", how="left")
features["atestados_futuros"] = features["atestados_futuros"].fillna(0)
features["dias_futuros"]      = features["dias_futuros"].fillna(0)

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
features["score_risco"] = (
    scaler_final.fit_transform(pred_raw.reshape(-1, 1)).flatten().round(1)
)
features["atestados_previstos"] = pred_raw.clip(min=0).round(1)

def classificar_risco(score):
    if score >= 70:   return "🔴 Alto"
    elif score >= 40: return "🟡 Médio"
    else:             return "🟢 Baixo"

features["nivel_risco"] = features["score_risco"].apply(classificar_risco)

grupo_predominante = (
    historico.groupby("MAT")
    .apply(lambda g: g.loc[g["peso_cid"].idxmax(), "grupo_cid"])
    .reset_index(name="grupo_cid_principal")
)
features = features.merge(grupo_predominante, on="MAT", how="left")

# ── Montar ranking final ───────────────────────────────────────────────────────
ranking = features[[
    "MAT", "score_risco", "nivel_risco", "atestados_previstos",
    "grupo_cid_principal", "total_atestados", "dias_afastados",
    "atestados_6m", "dias_desde_ultimo", "peso_cid_max",
]].copy()

ranking = ranking.sort_values("score_risco", ascending=False).reset_index(drop=True)
ranking.index = ranking.index + 1
ranking.index.name = "Posição"

ranking = ranking.rename(columns={
    "MAT":                 "Empregado",
    "score_risco":         "Score",
    "nivel_risco":         "Nível de risco",
    "atestados_previstos": "Previstos (90d)",
    "grupo_cid_principal": "Grupo CID",
    "total_atestados":     "Total atestados",
    "dias_afastados":      "Dias afastados",
    "atestados_6m":        "Atestados (6m)",
    "dias_desde_ultimo":   "Dias s/ atestado",
    "peso_cid_max":        "Peso CID",
})

# ── Exibição ──────────────────────────────────────────────────────────────────
st.subheader("Ranking de Absenteísmo — Previsão para os próximos 90 dias")

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

# Tabela sem scroll: altura calculada para mostrar todas as linhas
altura_tabela = (len(ranking) + 1) * 35 + 10

st.dataframe(
    ranking.style
        .background_gradient(subset=["Score"], cmap="RdYlGn_r")
        .format({"Score": "{:.1f}", "Previstos (90d)": "{:.1f}"}),
    use_container_width=True,
    height=altura_tabela,
)

# ── Exportar PDF ──────────────────────────────────────────────────────────────
st.divider()

def gerar_pdf(df_ranking: pd.DataFrame) -> bytes | None:
    try:
        from weasyprint import HTML
    except ImportError:
        return None

    gerado_em = datetime.now().strftime("%d/%m/%Y %H:%M")

    def cor_linha(nivel):
        if "Alto"  in nivel: return "#fff0f0"
        if "Médio" in nivel: return "#fffbea"
        return "#f0fff4"

    def badge(nivel):
        if "Alto"  in nivel:
            return '<span style="background:#ef4444;color:#fff;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;">Alto</span>'
        if "Médio" in nivel:
            return '<span style="background:#f59e0b;color:#fff;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;">Médio</span>'
        return '<span style="background:#22c55e;color:#fff;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;">Baixo</span>'

    linhas_html = ""
    for pos, row in df_ranking.iterrows():
        bg = cor_linha(row["Nível de risco"])
        linhas_html += f"""
        <tr style="background:{bg};">
            <td style="text-align:center;">{pos}</td>
            <td style="text-align:center;font-weight:600;">{row['Empregado']}</td>
            <td style="text-align:center;font-weight:700;">{row['Score']:.1f}</td>
            <td style="text-align:center;">{badge(row['Nível de risco'])}</td>
            <td style="text-align:center;">{row['Previstos (90d)']:.1f}</td>
            <td>{row['Grupo CID']}</td>
            <td style="text-align:center;">{int(row['Total atestados'])}</td>
            <td style="text-align:center;">{int(row['Dias afastados'])}</td>
            <td style="text-align:center;">{int(row['Atestados (6m)'])}</td>
            <td style="text-align:center;">{int(row['Dias s/ atestado'])}</td>
            <td style="text-align:center;">{row['Peso CID']:.1f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<style>
    @page {{ size: A4 landscape; margin: 1.5cm; }}
    body {{ font-family: Arial, sans-serif; font-size: 10px; color: #1e293b; }}
    h1 {{ font-size: 15px; color: #1e3a5f; margin-bottom: 2px; }}
    .sub {{ font-size: 9px; color: #64748b; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    thead tr {{ background: #1e3a5f; color: white; }}
    th {{ padding: 6px 8px; text-align: center; font-size: 9px; font-weight: 700; letter-spacing: 0.03em; }}
    td {{ padding: 5px 8px; border-bottom: 1px solid #e2e8f0; font-size: 9px; }}
    .footer {{ margin-top: 12px; font-size: 8px; color: #94a3b8; text-align: right; }}
</style>
</head>
<body>
    <h1>Score de Risco de Absenteísmo — Próximos {JANELA_DIAS} dias</h1>
    <p class="sub">
        Gerado em {gerado_em} &nbsp;|&nbsp;
        Base de dados até {hoje.strftime('%d/%m/%Y')} &nbsp;|&nbsp;
        Modelo: XGBoost &nbsp;|&nbsp;
        Total de empregados: {len(df_ranking)}
    </p>
    <table>
        <thead>
            <tr>
                <th>#</th><th>Empregado</th><th>Score</th><th>Nível</th>
                <th>Previstos (90d)</th><th>Grupo CID</th>
                <th>Total atestados</th><th>Dias afastados</th>
                <th>Atestados (6m)</th><th>Dias s/ atestado</th><th>Peso CID</th>
            </tr>
        </thead>
        <tbody>
            {linhas_html}
        </tbody>
    </table>
    <p class="footer">Jorge Eduardo de Araujo Oliveira — Análise ML · Afastamentos</p>
</body>
</html>"""

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        HTML(string=html).write_pdf(f.name)
        pdf_bytes = open(f.name, "rb").read()
    os.unlink(f.name)
    return pdf_bytes


col_btn, col_info = st.columns([1, 5])
with col_btn:
    if st.button("📄 Exportar PDF", use_container_width=True):
        with st.spinner("Gerando PDF..."):
            pdf_bytes = gerar_pdf(ranking)

        if pdf_bytes is None:
            st.error(
                "Biblioteca `weasyprint` não instalada.\n\n"
                "Execute no terminal: `pip install weasyprint`"
            )
        else:
            nome = f"score_absenteismo_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                label="⬇️ Baixar PDF",
                data=pdf_bytes,
                file_name=nome,
                mime="application/pdf",
                use_container_width=True,
            )
with col_info:
    st.caption(
        "PDF gerado em A4 paisagem com todos os empregados, colorido por nível de risco. "
        "Requer `pip install weasyprint`."
    )
