import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import shap
import warnings
from fpdf import FPDF

warnings.filterwarnings("ignore")

# ── st.set_page_config DEVE ser a primeira chamada Streamlit ──────────────────
st.set_page_config(
    page_title="Score de Risco de Absenteísmo",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Login ─────────────────────────────────────────────────────────────────────
if "logado" not in st.session_state:
    st.session_state.logado = False

if not st.session_state.logado:
    col_login, _, _ = st.columns([1, 1, 1])
    with col_login:
        st.title("🔒 Acesso Restrito")
        usuario = st.text_input("Usuário")
        senha   = st.text_input("Senha", type="password")
        if st.button("Entrar", use_container_width=True):
            usuario_correto = st.secrets.get("usuario", "rhli")      if hasattr(st, "secrets") else "rhli"
            senha_correta   = st.secrets.get("senha",   "Rhli@2026")  if hasattr(st, "secrets") else "Rhli@2026"
            if usuario == usuario_correto and senha == senha_correta:
                st.session_state.logado = True
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos.")
    st.stop()

if st.sidebar.button("Sair"):
    st.session_state.logado = False
    st.rerun()

# ── Estilos ───────────────────────────────────────────────────────────────────
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
    <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Python · Numpy · Pandas · Streamlit · XGBoost · Scikit-Learn · SHAP</p>
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
    cid   = str(cid).strip().upper()
    letra = cid[0] if cid else "?"
    info  = GRUPO_CID.get(letra, {"grupo": "Outro", "peso": 1.0})
    return info["grupo"], info["peso"]

# ── Nomes amigáveis para exibição SHAP ───────────────────────────────────────
FEATURE_LABELS = {
    "dias_desde_ultimo":    "Dias desde último atestado",
    "total_atestados":      "Total de atestados",
    "dias_afastados":       "Total de dias afastados",
    "atestados_6m":         "Atestados (últimos 6 meses)",
    "atestados_3m":         "Atestados (últimos 3 meses)",
    "dias_afastados_6m":    "Dias afastados (6 meses)",
    "freq_mensal":          "Frequência mensal de atestados",
    "media_dias_atestado":  "Média de dias por atestado",
    "tendencia_recente":    "Tendência recente (6m/total)",
    "score_recencia":       "Score de recência",
    "peso_cid_max":         "Peso máximo CID",
    "peso_cid_ponderado":   "Peso CID ponderado por dias",
    "diversidade_cid":      "Diversidade de CIDs",
    "tem_cid_cronico":      "Tem CID crônico (peso >= 3)",
}

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
df["MAT"]  = df["MAT"].astype(str).str.zfill(6)
df["DATA"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["DATA"])
df = df.sort_values(["MAT", "DATA"])
df["CID"]  = df["CID"].astype(str).str.strip().str.upper()
df[["grupo_cid", "peso_cid"]] = df["CID"].apply(lambda c: pd.Series(get_cid_info(c)))

ultimo_mes = pd.Timestamp.now().replace(day=1) - pd.Timedelta(days=1)
hoje = ultimo_mes.normalize()

# ── Métricas principais ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Total de empregados", df["MAT"].nunique())
col2.metric("Total de atestados",  len(df))
col3.metric("Dias afastados",      int(df["DIAS"].sum()))

# ── Janela temporal ───────────────────────────────────────────────────────────
JANELA_IDEAL = 90
span_dias    = (hoje - df["DATA"].min()).days
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

# ── Histórico e futuro ────────────────────────────────────────────────────────
todos_empregados = pd.DataFrame(df["MAT"].unique(), columns=["MAT"])
historico = df[df["DATA"] < data_corte].copy()
historico = todos_empregados.merge(historico, on="MAT", how="left")
historico["DIAS"]      = historico["DIAS"].fillna(0)
historico["CID"]       = historico["CID"].fillna("Z")
historico["DATA"]      = historico["DATA"].fillna(df["DATA"].min())
historico["peso_cid"]  = historico.apply(lambda row: get_cid_info(row["CID"])[1], axis=1)
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

    feat = freq.merge(dias,     on="MAT", how="right")
    feat = feat.merge(ultimo,   on="MAT", how="right")
    feat = feat.merge(primeiro, on="MAT", how="right")
    feat = feat.merge(freq_6m,  on="MAT", how="left")
    feat = feat.merge(freq_3m,  on="MAT", how="left")
    feat = feat.merge(dias_6m,  on="MAT", how="left")
    feat = feat.merge(peso_max_cid,    on="MAT", how="left")
    feat = feat.merge(peso_pond,       on="MAT", how="left")
    feat = feat.merge(diversidade_cid, on="MAT", how="left")
    feat = feat.merge(tem_cronico,     on="MAT", how="left")

    feat = todos_empregados.merge(feat, on="MAT", how="left")

    ref_now = pd.Timestamp.now()
    feat["dias_desde_ultimo"] = (ref_now - feat["data_ultimo"]).dt.days

    feat = feat.fillna({
        "dias_desde_ultimo":   (ref_now - df["DATA"].min()).days,
        "total_atestados":     0,
        "dias_afastados":      0,
        "atestados_6m":        0,
        "atestados_3m":        0,
        "dias_afastados_6m":   0,
        "freq_mensal":         0,
        "media_dias_atestado": 0,
        "tendencia_recente":   0,
        "score_recencia":      0,
        "peso_cid_max":        0,
        "peso_cid_ponderado":  0,
        "diversidade_cid":     0,
        "tem_cid_cronico":     0,
    })

    feat["meses_historico"]     = ((feat["data_ultimo"] - feat["data_primeiro"]).dt.days / 30).clip(lower=1).fillna(1)
    feat["freq_mensal"]         = feat["total_atestados"] / feat["meses_historico"]
    feat["media_dias_atestado"] = feat["dias_afastados"]  / feat["total_atestados"].replace(0, 1)
    feat["tendencia_recente"]   = feat["atestados_6m"]    / (feat["total_atestados"] + 1)
    feat["score_recencia"]      = 1 / (feat["dias_desde_ultimo"] + 1)

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
    "dias_futuros":      0,
})

# ── Grupo CID principal ───────────────────────────────────────────────────────
grupo_predominante = (
    historico.groupby("MAT")
    .apply(lambda g: g.loc[g["peso_cid"].idxmax(), "grupo_cid"])
    .reset_index(name="grupo_cid_principal")
)
features = features.merge(grupo_predominante, on="MAT", how="left")
features["grupo_cid_principal"] = features["grupo_cid_principal"].fillna("Não informado")

# ── Modelo XGBoost ────────────────────────────────────────────────────────────
feature_cols = [
    "dias_desde_ultimo", "total_atestados", "dias_afastados",
    "atestados_6m", "atestados_3m", "dias_afastados_6m",
    "freq_mensal", "media_dias_atestado", "tendencia_recente",
    "score_recencia", "peso_cid_max", "peso_cid_ponderado",
    "diversidade_cid", "tem_cid_cronico",
]

X = features[feature_cols].fillna(0)
y = features["atestados_futuros"].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)

st.sidebar.metric("MAE do modelo", f"{mae:.2f}")

# ── Predição sem vazamento ────────────────────────────────────────────────────
pred_full = np.zeros(len(X))
pred_full[X_train.index] = model.predict(X_train)
pred_full[X_test.index]  = y_pred_test

features["atestados_previstos"] = np.clip(pred_full, a_min=0, a_max=None).round(2)

# ── Score e classificação ─────────────────────────────────────────────────────
features["score_risco"] = (
    features["atestados_previstos"].rank(pct=True) * 100
).round(1)

def classificar_risco(score):
    if score >= 75:
        return "🔴 Alto"
    elif score >= 40:
        return "🟡 Médio"
    else:
        return "🟢 Baixo"

features["nivel_risco"] = features["score_risco"].apply(classificar_risco)

# ── Ranking ───────────────────────────────────────────────────────────────────
ranking = features[[
    "MAT", "score_risco", "nivel_risco", "atestados_previstos",
    "grupo_cid_principal", "total_atestados", "dias_afastados",
    "atestados_6m", "dias_desde_ultimo", "peso_cid_max",
]].sort_values("score_risco", ascending=False).reset_index(drop=True)

ranking.index += 1

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

# ── SHAP: calcular e cachear ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Calculando explicações SHAP…")
def calcular_shap(_model, X_data):
    explainer   = shap.TreeExplainer(_model)
    shap_values = explainer(X_data)
    return shap_values

X_labeled   = X.rename(columns=FEATURE_LABELS)
shap_values = calcular_shap(model, X_labeled)

# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE GERAÇÃO DE PDF
# ═══════════════════════════════════════════════════════════════════════════════

# ── PDF 1: Score de todos os empregados ──────────────────────────────────────
def gerar_pdf_ranking(df_ranking, features):
    AZUL_ESCURO = (30,  58,  95)
    CINZA_TEXTO = (60,  60,  60)
    BRANCO      = (255, 255, 255)

    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(*AZUL_ESCURO)
    pdf.rect(0, 0, 297, 22, 'F')
    pdf.set_text_color(*BRANCO)
    pdf.set_font("Arial", "B", 13)
    pdf.set_xy(10, 4)
    pdf.cell(0, 8, "Score de Risco de Absenteísmo — Listagem Geral", ln=True)
    pdf.set_font("Arial", "", 8)
    pdf.set_xy(10, 13)
    pdf.cell(0, 6,
        f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  "
        f"Ordenado por Score (maior para menor)", ln=True)

    n_alto  = (features["nivel_risco"] == "🔴 Alto").sum()
    n_medio = (features["nivel_risco"] == "🟡 Médio").sum()
    n_baixo = (features["nivel_risco"] == "🟢 Baixo").sum()

    pdf.set_xy(10, 26)
    pdf.set_font("Arial", "B", 8)
    for label, val, cor in [
        ("Alto Risco",  n_alto,  (200, 50,  50)),
        ("Médio Risco", n_medio, (170, 130, 10)),
        ("Baixo Risco", n_baixo, (30,  130, 60)),
    ]:
        pdf.set_fill_color(*cor)
        pdf.set_text_color(*BRANCO)
        pdf.cell(38, 7, f"{label}: {val}", border=0, fill=True, align='C')
        pdf.set_x(pdf.get_x() + 3)
    pdf.set_text_color(*CINZA_TEXTO)
    pdf.ln(11)

    cols   = ["Pos", "Matrícula", "Score", "Risco",    "Prev.(90d)", "Grupo CID", "Total", "Dias", "6m",  "S/Ates.", "Peso"]
    widths = [10,    24,          18,      22,          20,           56,          16,      16,     14,    20,        14]
    aligns = ['C',   'C',         'C',     'C',         'C',          'L',         'C',     'C',    'C',   'C',       'C']

    pdf.set_font("Arial", "B", 7)
    pdf.set_fill_color(*AZUL_ESCURO)
    pdf.set_text_color(*BRANCO)
    for col, w, a in zip(cols, widths, aligns):
        pdf.cell(w, 6, col, border=1, fill=True, align=a)
    pdf.ln()

    pdf.set_font("Arial", "", 6.5)
    for pos, row in df_ranking.iterrows():
        nivel_txt = (str(row["Nível de risco"])
                     .replace("🔴 ","").replace("🟡 ","").replace("🟢 ",""))
        if   "Alto"  in nivel_txt: pdf.set_fill_color(255, 230, 230)
        elif "Médio" in nivel_txt: pdf.set_fill_color(255, 255, 220)
        else:                      pdf.set_fill_color(230, 255, 230)
        pdf.set_text_color(*CINZA_TEXTO)

        vals = [
            str(pos),
            str(row["Empregado"]),
            f"{float(row['Score']):.1f}",
            nivel_txt,
            f"{float(row['Previstos (90d)']):.2f}",
            str(row["Grupo CID"])[:28],
            str(int(row["Total atestados"])),
            str(int(row["Dias afastados"])),
            str(int(row["Atestados (6m)"])),
            str(int(row["Dias s/ atestado"])),
            f"{float(row['Peso CID']):.1f}",
        ]
        for v, w, a in zip(vals, widths, aligns):
            pdf.cell(w, 5, v, border=1, fill=True, align=a)
        pdf.ln()

    return bytes(pdf.output())


# ── PDF 2: Relatório SHAP ─────────────────────────────────────────────────────
def gerar_pdf_shap(features, X_labeled, shap_values):
    import tempfile, os

    feat_names = X_labeled.columns.tolist()
    mean_shap  = np.abs(shap_values.values).mean(axis=0)

    AZUL_ESCURO = (30,  58,  95)
    CINZA_TEXTO = (60,  60,  60)
    BRANCO      = (255, 255, 255)

    def cabecalho(pdf, titulo, subtitulo=""):
        pdf.set_fill_color(*AZUL_ESCURO)
        pdf.rect(0, 0, 297, 22, 'F')
        pdf.set_text_color(*BRANCO)
        pdf.set_font("Arial", "B", 13)
        pdf.set_xy(10, 4)
        pdf.cell(0, 8, titulo, ln=True)
        if subtitulo:
            pdf.set_font("Arial", "", 8)
            pdf.set_xy(10, 13)
            pdf.cell(0, 6, subtitulo, ln=True)
        pdf.set_text_color(*CINZA_TEXTO)

    def salvar_fig(fig):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return tmp.name

    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Página 1: Importância Global ─────────────────────────────────────────
    pdf.add_page()
    cabecalho(pdf,
        "Explicabilidade SHAP — Importância Global das Features",
        f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  Média |SHAP| de todos os empregados"
    )

    pdf.set_xy(10, 26)
    pdf.set_font("Arial", "", 8)
    pdf.multi_cell(277, 5,
        "O SHAP (SHapley Additive exPlanations) decompoe a previsão mostrando o quanto cada variável "
        "contribuiu para aumentar ou reduzir o risco. Quanto maior a barra, mais relevante é a feature "
        "para o modelo globalmente.", ln=True)

    df_imp = (pd.DataFrame({"Feature": feat_names, "Importância": mean_shap})
              .sort_values("Importância", ascending=True))

    fig, ax = plt.subplots(figsize=(11, 5))
    cores_bar = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(df_imp)))
    bars = ax.barh(df_imp["Feature"], df_imp["Importância"],
                   color=cores_bar, edgecolor="none", height=0.65)
    ax.set_xlabel("Importância média |SHAP|", fontsize=9)
    ax.set_title("Contribuição Global das Features", fontsize=11, fontweight="bold")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, df_imp["Importância"]):
        ax.text(val + max(df_imp["Importância"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=7.5)
    fig.tight_layout()
    img1 = salvar_fig(fig)
    pdf.image(img1, x=10, y=42, w=277)
    os.unlink(img1)

    # ── Página 2: Beeswarm ───────────────────────────────────────────────────
    pdf.add_page()
    cabecalho(pdf,
        "Explicabilidade SHAP — Distribuição dos Valores por Feature (Beeswarm)",
        "Cada ponto = 1 empregado  |  Vermelho = feature alta  |  Azul = feature baixa"
    )

    pdf.set_xy(10, 26)
    pdf.set_font("Arial", "", 8)
    pdf.multi_cell(277, 5,
        "A posição horizontal mostra se a feature aumentou (direita/vermelho) ou diminuiu "
        "(esquerda/azul) o risco previsto para cada empregado individualmente.", ln=True)

    order  = np.argsort(mean_shap)
    sv_ord = shap_values.values[:, order]
    fv_ord = X_labeled.values[:, order]
    fn_ord = [feat_names[i] for i in order]

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    sc = None
    for i, (sv, fv) in enumerate(zip(sv_ord.T, fv_ord.T)):
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        jitter  = np.random.uniform(-0.3, 0.3, size=len(sv))
        sc = ax2.scatter(sv, i + jitter, c=fv_norm, cmap="RdBu_r",
                         vmin=0, vmax=1, s=14, alpha=0.6, linewidths=0)
    ax2.axvline(0, color="#555", linewidth=0.8, linestyle="--")
    ax2.set_yticks(range(len(fn_ord)))
    ax2.set_yticklabels(fn_ord, fontsize=8)
    ax2.set_xlabel("Valor SHAP", fontsize=9)
    ax2.set_title("Beeswarm — Distribuição dos SHAP Values", fontsize=11, fontweight="bold")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.3)
    if sc:
        cbar = fig2.colorbar(sc, ax=ax2, pad=0.01, fraction=0.015)
        cbar.set_label("Valor da feature\n(azul=baixo · vermelho=alto)", fontsize=7)
    fig2.tight_layout()
    img2 = salvar_fig(fig2)
    pdf.image(img2, x=10, y=42, w=277)
    os.unlink(img2)

    # ── Páginas 3+: Análise individual Top 20 ────────────────────────────────
    top20 = features.nlargest(20, "score_risco")

    for _, emp_row in top20.iterrows():
        pdf.add_page()

        nivel = emp_row["nivel_risco"]
        if   "Alto"  in nivel: cor_h = (180, 30,  30)
        elif "Médio" in nivel: cor_h = (170, 120, 10)
        else:                  cor_h = (30,  130, 60)

        nivel_txt = nivel.replace("🔴 ","").replace("🟡 ","").replace("🟢 ","")
        mat       = emp_row["MAT"]
        score     = emp_row["score_risco"]
        previstos = emp_row["atestados_previstos"]

        pdf.set_fill_color(*cor_h)
        pdf.rect(0, 0, 297, 22, 'F')
        pdf.set_text_color(*BRANCO)
        pdf.set_font("Arial", "B", 12)
        pdf.set_xy(10, 4)
        pdf.cell(0, 8,
            f"SHAP Individual  |  Matrícula: {mat}  |  Score: {score:.1f}/100  |  "
            f"Previstos (90d): {previstos:.2f}  |  Risco: {nivel_txt}", ln=True)
        pdf.set_text_color(*CINZA_TEXTO)

        idx_emp = features.index[features["MAT"] == mat].tolist()[0]
        sv_emp  = shap_values.values[idx_emp]
        fv_emp  = X_labeled.iloc[idx_emp].values

        df_emp = (pd.DataFrame({"Label": feat_names, "SHAP": sv_emp, "Valor": fv_emp})
                  .sort_values("SHAP", key=abs, ascending=True).tail(14))

        cores_emp = ["#d73027" if v > 0 else "#4575b4" for v in df_emp["SHAP"]]
        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        bars3 = ax3.barh(df_emp["Label"], df_emp["SHAP"],
                         color=cores_emp, edgecolor="none", height=0.65)
        ax3.axvline(0, color="#555", linewidth=0.8, linestyle="--")
        ax3.set_xlabel("Contribuição SHAP", fontsize=9)
        ax3.set_title(f"Matrícula {mat} — Contribuição de cada feature",
                      fontsize=10, fontweight="bold")
        ax3.spines[["top","right","left"]].set_visible(False)
        ax3.tick_params(axis="y", labelsize=8)
        ax3.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax3.set_axisbelow(True)
        for bar, row in zip(bars3, df_emp.itertuples()):
            xp   = bar.get_width()
            sign = "+" if xp >= 0 else ""
            ax3.text(xp + (0.002 if xp >= 0 else -0.002),
                     bar.get_y() + bar.get_height() / 2,
                     f"{sign}{xp:.3f}  (val={row.Valor:.2f})",
                     va="center", ha="left" if xp >= 0 else "right",
                     fontsize=7, color="#222")
        legenda = [
            mpatches.Patch(color="#d73027", label="Aumenta o risco (SHAP > 0)"),
            mpatches.Patch(color="#4575b4", label="Reduz o risco (SHAP < 0)"),
        ]
        ax3.legend(handles=legenda, fontsize=7, loc="lower right")
        ax3.set_xlim(df_emp["SHAP"].min() * 1.35, df_emp["SHAP"].max() * 1.35)
        fig3.tight_layout()
        img3 = salvar_fig(fig3)
        pdf.image(img3, x=10, y=25, w=190)
        os.unlink(img3)

        # Tabela à direita do gráfico
        df_det = (pd.DataFrame({
            "Feature": feat_names,
            "Valor":   fv_emp,
            "SHAP":    sv_emp,
        }).sort_values("SHAP", key=abs, ascending=False).reset_index(drop=True))

        pdf.set_xy(205, 25)
        pdf.set_font("Arial", "B", 6.5)
        pdf.set_fill_color(*AZUL_ESCURO)
        pdf.set_text_color(*BRANCO)
        for col, w in [("Feature", 52), ("Valor", 16), ("SHAP", 18)]:
            pdf.cell(w, 5, col, border=1, fill=True, align='C')
        pdf.ln()

        pdf.set_font("Arial", "", 6)
        for _, drow in df_det.iterrows():
            sv = drow["SHAP"]
            if   sv >  0.01: pdf.set_fill_color(255, 230, 230)
            elif sv < -0.01: pdf.set_fill_color(220, 235, 255)
            else:            pdf.set_fill_color(245, 245, 245)
            pdf.set_text_color(*CINZA_TEXTO)
            sign = "+" if sv >= 0 else ""
            pdf.set_x(205)
            pdf.cell(52, 4.5, str(drow["Feature"])[:28], border=1, fill=True)
            pdf.cell(16, 4.5, f"{drow['Valor']:.2f}",    border=1, fill=True, align='C')
            pdf.cell(18, 4.5, f"{sign}{sv:.4f}",          border=1, fill=True, align='C')
            pdf.ln()

    return bytes(pdf.output())


# ── PDF 3: Curva de Risco de Recorrência (hiperbólica) ────────────────────────
def gerar_pdf_curva_risco(features):
    import tempfile, os

    AZUL_ESCURO = (30,  58,  95)
    CINZA_TEXTO = (60,  60,  60)
    BRANCO      = (255, 255, 255)

    df_plot = features[["score_risco", "atestados_previstos", "nivel_risco"]].copy()
    df_plot = df_plot.sort_values("score_risco").reset_index(drop=True)

    x = df_plot["score_risco"].values
    y = df_plot["atestados_previstos"].values

    janela   = max(3, len(df_plot) // 8)
    y_smooth = pd.Series(y).rolling(window=janela, center=True, min_periods=1).mean().values

    cores_nivel = {
        "🔴 Alto":  "#d73027",
        "🟡 Médio": "#f5a623",
        "🟢 Baixo": "#2ecc71",
    }
    cores_pts = df_plot["nivel_risco"].map(cores_nivel).fillna("#aaa")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axvspan(0,  40,  alpha=0.07, color="#2ecc71")
    ax.axvspan(40, 75,  alpha=0.07, color="#f5a623")
    ax.axvspan(75, 100, alpha=0.07, color="#d73027")

    for xv, cor in [(40, "#f5a623"), (75, "#d73027")]:
        ax.axvline(xv, color=cor, linewidth=1.2, linestyle="--", alpha=0.7)

    y_max = max(y) if len(y) > 0 else 1
    ax.text(20, y_max * 0.97, "Baixo Risco",  ha="center", fontsize=9,
            color="#27ae60", fontweight="bold", alpha=0.8)
    ax.text(57, y_max * 0.97, "Médio Risco",  ha="center", fontsize=9,
            color="#e67e22", fontweight="bold", alpha=0.8)
    ax.text(87, y_max * 0.97, "Alto Risco",   ha="center", fontsize=9,
            color="#c0392b", fontweight="bold", alpha=0.8)

    ax.scatter(x, y, c=cores_pts, s=28, alpha=0.55, zorder=3, linewidths=0)
    ax.plot(x, y_smooth, color="#1e3a5f", linewidth=2.5, zorder=4,
            label="Tendência (curva de risco)")

    top5 = df_plot.nlargest(5, "score_risco")
    for _, r in top5.iterrows():
        ax.annotate(
            f"  {int(r['score_risco'])}",
            (r["score_risco"], r["atestados_previstos"]),
            fontsize=7, color="#c0392b", va="center"
        )

    ax.set_xlabel("Score de Risco (percentil 0–100)", fontsize=10)
    ax.set_ylabel("Atestados Previstos (90 dias)", fontsize=10)
    ax.set_title("Curva de Risco de Recorrência de Absenteísmo",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    legenda = [
        mpatches.Patch(color="#2ecc71", alpha=0.5, label="Baixo Risco  (score < 40)"),
        mpatches.Patch(color="#f5a623", alpha=0.5, label="Médio Risco  (40 <= score < 75)"),
        mpatches.Patch(color="#d73027", alpha=0.5, label="Alto Risco   (score >= 75)"),
        plt.Line2D([0],[0], color="#1e3a5f", linewidth=2.5, label="Tendência (curva de risco)"),
    ]
    ax.legend(handles=legenda, fontsize=8, loc="upper left", framealpha=0.85)
    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    img_path = tmp.name

    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(*AZUL_ESCURO)
    pdf.rect(0, 0, 297, 22, 'F')
    pdf.set_text_color(*BRANCO)
    pdf.set_font("Arial", "B", 13)
    pdf.set_xy(10, 4)
    pdf.cell(0, 8, "Curva de Risco de Recorrência de Absenteísmo", ln=True)
    pdf.set_font("Arial", "", 8)
    pdf.set_xy(10, 13)
    pdf.cell(0, 6,
        f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  "
        f"Score vs Atestados Previstos — modelo XGBoost", ln=True)

    pdf.set_text_color(*CINZA_TEXTO)
    pdf.set_xy(10, 26)
    pdf.set_font("Arial", "", 8)
    pdf.multi_cell(277, 5,
        "O gráfico exibe a curva de risco de recorrência de absenteísmo: cada ponto representa um "
        "empregado, posicionado pelo seu score de risco (eixo X) e pelo número de atestados previstos "
        "para os próximos 90 dias (eixo Y). A curva azul é a tendência suavizada, revelando o formato "
        "hiperbólico característico — poucos empregados concentram a maior parte do risco.", ln=True)

    pdf.image(img_path, x=10, y=44, w=277)
    os.unlink(img_path)

    n_alto  = (features["nivel_risco"] == "🔴 Alto").sum()
    n_medio = (features["nivel_risco"] == "🟡 Médio").sum()
    n_baixo = (features["nivel_risco"] == "🟢 Baixo").sum()
    prev_alto  = features[features["nivel_risco"] == "🔴 Alto" ]["atestados_previstos"].sum()
    prev_medio = features[features["nivel_risco"] == "🟡 Médio"]["atestados_previstos"].sum()
    prev_baixo = features[features["nivel_risco"] == "🟢 Baixo"]["atestados_previstos"].sum()
    total_prev = prev_alto + prev_medio + prev_baixo + 1e-9

    pdf.set_xy(10, 168)
    pdf.set_font("Arial", "B", 7)
    pdf.set_fill_color(*AZUL_ESCURO)
    pdf.set_text_color(*BRANCO)
    for col, w in [("Faixa de Risco", 50), ("Empregados", 35), ("Atestados Previstos", 45), ("% do Total Previsto", 45)]:
        pdf.cell(w, 6, col, border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font("Arial", "", 7)
    for label, n, prev, cor_fill in [
        ("Alto  (score >= 75)",      n_alto,  prev_alto,  (255, 230, 230)),
        ("Médio (40 <= score < 75)", n_medio, prev_medio, (255, 255, 220)),
        ("Baixo (score < 40)",       n_baixo, prev_baixo, (230, 255, 230)),
    ]:
        pdf.set_fill_color(*cor_fill)
        pdf.set_text_color(*CINZA_TEXTO)
        pdf.cell(50, 5, label,                            border=1, fill=True)
        pdf.cell(35, 5, str(n),                           border=1, fill=True, align='C')
        pdf.cell(45, 5, f"{prev:.1f}",                    border=1, fill=True, align='C')
        pdf.cell(45, 5, f"{prev/total_prev*100:.1f}%",    border=1, fill=True, align='C')
        pdf.ln()

    return bytes(pdf.output())


# ── PDF 4: Real vs Previsto ───────────────────────────────────────────────────
def gerar_pdf_real_vs_previsto(features, X_test, y_test, y_pred_test, mae):
    import tempfile, os

    AZUL_ESCURO = (30,  58,  95)
    CINZA_TEXTO = (60,  60,  60)
    BRANCO      = (255, 255, 255)

    y_real   = y_test.values
    y_prev   = y_pred_test
    residuos = y_real - y_prev

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 1: Scatter Real vs Previsto
    ax1  = axes[0]
    vmax = max(y_real.max(), y_prev.max()) * 1.05
    cores_pts = ["#d73027" if abs(r) > 1 else "#4575b4" for r in residuos]
    ax1.scatter(y_real, y_prev, c=cores_pts, s=30, alpha=0.6, linewidths=0, zorder=3)
    ax1.plot([0, vmax], [0, vmax], color="#1e3a5f", linewidth=1.5,
             linestyle="--", label="Ajuste perfeito", zorder=4)
    ax1.set_xlabel("Atestados Reais", fontsize=9)
    ax1.set_ylabel("Atestados Previstos", fontsize=9)
    ax1.set_title("Real vs Previsto", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, vmax); ax1.set_ylim(0, vmax)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)
    legenda1 = [
        plt.Line2D([0],[0], color="#1e3a5f", linewidth=1.5, linestyle="--", label="Ajuste perfeito"),
        mpatches.Patch(color="#d73027", alpha=0.7, label="Erro > 1 atestado"),
        mpatches.Patch(color="#4575b4", alpha=0.7, label="Erro <= 1 atestado"),
    ]
    ax1.legend(handles=legenda1, fontsize=7, loc="upper left", framealpha=0.85)
    ax1.text(vmax * 0.98, vmax * 0.05, f"MAE = {mae:.3f}",
             ha="right", fontsize=8, color="#1e3a5f", fontweight="bold")

    # Subplot 2: Resíduos
    ax2 = axes[1]
    res_sorted = sorted(residuos)
    cores_res  = ["#d73027" if r > 0 else "#4575b4" for r in res_sorted]
    ax2.bar(range(len(res_sorted)), res_sorted, color=cores_res, edgecolor="none", width=0.8)
    ax2.axhline(0, color="#555", linewidth=1.0, linestyle="--")
    ax2.set_xlabel("Empregados (ordenados por resíduo)", fontsize=9)
    ax2.set_ylabel("Resíduo (Real - Previsto)", fontsize=9)
    ax2.set_title("Distribuição dos Resíduos", fontsize=10, fontweight="bold")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.text(len(residuos) * 0.98, max(residuos) * 0.92,
             f"Média: {residuos.mean():.3f}\nDP: {residuos.std():.3f}",
             ha="right", fontsize=7.5, color="#333",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Subplot 3: Histograma dos erros absolutos
    ax3      = axes[2]
    erros_abs = np.abs(residuos)
    n_bins   = min(20, len(erros_abs) // 2 + 1)
    ax3.hist(erros_abs, bins=n_bins, color="#4575b4", edgecolor="white",
             linewidth=0.5, alpha=0.85)
    ax3.axvline(mae, color="#d73027", linewidth=1.8, linestyle="--",
                label=f"MAE = {mae:.3f}")
    ax3.axvline(np.median(erros_abs), color="#f5a623", linewidth=1.5,
                linestyle=":", label=f"Mediana = {np.median(erros_abs):.3f}")
    ax3.set_xlabel("Erro Absoluto", fontsize=9)
    ax3.set_ylabel("Frequência", fontsize=9)
    ax3.set_title("Histograma dos Erros Absolutos", fontsize=10, fontweight="bold")
    ax3.spines[["top","right"]].set_visible(False)
    ax3.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=7.5, framealpha=0.85)
    pct_mae1 = (erros_abs <= 1).mean() * 100
    ax3.text(erros_abs.max() * 0.98, ax3.get_ylim()[1] * 0.85,
             f"{pct_mae1:.1f}% com\nerro <= 1",
             ha="right", fontsize=7.5, color="#27ae60", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Ajuste do Modelo — Dados Reais vs Previstos (conjunto de teste)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    img_path = tmp.name

    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(*AZUL_ESCURO)
    pdf.rect(0, 0, 297, 22, 'F')
    pdf.set_text_color(*BRANCO)
    pdf.set_font("Arial", "B", 13)
    pdf.set_xy(10, 4)
    pdf.cell(0, 8, "Ajuste do Modelo — Dados Reais vs Previstos", ln=True)
    pdf.set_font("Arial", "", 8)
    pdf.set_xy(10, 13)
    pdf.cell(0, 6,
        f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  "
        f"XGBoost · Conjunto de teste (20%)  |  MAE = {mae:.3f}", ln=True)

    pdf.set_text_color(*CINZA_TEXTO)
    pdf.set_xy(10, 26)
    pdf.set_font("Arial", "", 8)
    pdf.multi_cell(277, 5,
        "Os gráficos avaliam o desempenho do modelo XGBoost no conjunto de teste (20% dos dados). "
        "À esquerda: dispersão entre valores reais e previstos — quanto mais próximos da diagonal "
        "tracejada, melhor o ajuste. Ao centro: resíduos ordenados, mostrando a simetria dos erros. "
        "À direita: histograma dos erros absolutos com MAE e mediana destacados.", ln=True)

    pdf.image(img_path, x=8, y=44, w=281)
    os.unlink(img_path)

    n_test   = len(y_real)
    rmse     = np.sqrt(np.mean(residuos ** 2))
    pct_zero = (erros_abs == 0).mean() * 100
    pct_mae1 = (erros_abs <= 1).mean() * 100
    bias     = residuos.mean()

    pdf.set_xy(10, 168)
    pdf.set_font("Arial", "B", 7)
    pdf.set_fill_color(*AZUL_ESCURO)
    pdf.set_text_color(*BRANCO)
    for col, w in [("Métrica", 70), ("Valor", 35), ("Interpretação", 120)]:
        pdf.cell(w, 6, col, border=1, fill=True, align='C')
    pdf.ln()

    metricas = [
        ("MAE (Erro Absoluto Médio)",      f"{mae:.4f}",   "Erro médio de previsão em número de atestados"),
        ("RMSE (Raiz do Erro Quadrático)", f"{rmse:.4f}",  "Penaliza mais os grandes erros; quanto menor, melhor"),
        ("Bias (erro sistemático)",        f"{bias:+.4f}", "Próximo de zero indica ausência de viés sistemático"),
        ("Acurácia (erro = 0)",            f"{pct_zero:.1f}%", "% de empregados com previsão exata"),
        ("Precisão (erro <= 1 atestado)",  f"{pct_mae1:.1f}%", "% de empregados com erro de no máximo 1 atestado"),
        ("Tamanho do conjunto de teste",   str(n_test),    "Empregados usados para avaliação (20% do total)"),
    ]

    pdf.set_font("Arial", "", 7)
    for i, (metrica, valor, interp) in enumerate(metricas):
        cor_fill = (245, 247, 250) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*cor_fill)
        pdf.set_text_color(*CINZA_TEXTO)
        pdf.cell(70,  5, metrica, border=1, fill=True)
        pdf.cell(35,  5, valor,   border=1, fill=True, align='C')
        pdf.cell(120, 5, interp,  border=1, fill=True)
        pdf.ln()

    return bytes(pdf.output())


# ── Botões na sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("Exportar Dados")

    try:
        pdf_ranking = gerar_pdf_ranking(ranking, features)
        st.download_button(
            label="📄 Score Geral (PDF)",
            data=pdf_ranking,
            file_name=f"score_absenteismo_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Erro ao gerar PDF de score: {e}")

    try:
        pdf_shap = gerar_pdf_shap(features, X_labeled, shap_values)
        st.download_button(
            label="🔍 Relatório SHAP (PDF)",
            data=pdf_shap,
            file_name=f"relatorio_shap_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Erro ao gerar PDF SHAP: {e}")

    try:
        pdf_curva = gerar_pdf_curva_risco(features)
        st.download_button(
            label="📈 Curva de Risco (PDF)",
            data=pdf_curva,
            file_name=f"curva_risco_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Erro ao gerar PDF de curva de risco: {e}")

    try:
        pdf_ajuste = gerar_pdf_real_vs_previsto(features, X_test, y_test, y_pred_test, mae)
        st.download_button(
            label="📊 Real vs Previsto (PDF)",
            data=pdf_ajuste,
            file_name=f"real_vs_previsto_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Erro ao gerar PDF real vs previsto: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — Ranking
# ═══════════════════════════════════════════════════════════════════════════════
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

altura_tabela = min((len(ranking) + 1) * 35 + 10, 800)
st.dataframe(
    ranking.style
        .background_gradient(subset=["Score"], cmap="RdYlGn_r")
        .format({
            "Score": "{:.2f}",
            "Previstos (90d)": "{:.2f}",
            "Peso CID": "{:.2f}",
            "Dias afastados": "{:.0f}",
            "Total atestados": "{:.0f}",
            "Atestados (6m)": "{:.0f}",
            "Dias s/ atestado": "{:.0f}",
        }),
    use_container_width=True,
    height=altura_tabela,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — Análise SHAP
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("🔍 Explicabilidade do Modelo — SHAP")
st.markdown(
    "O SHAP (SHapley Additive exPlanations) decompõe a previsão de cada empregado "
    "mostrando **quanto cada variável contribuiu** para aumentar ou reduzir o risco previsto. "
    "Valores positivos (vermelho) elevam o risco; negativos (azul) reduzem."
)

tab1, tab2, tab3 = st.tabs([
    "📊 Importância Global",
    "🌐 Impacto por Feature (Beeswarm)",
    "👤 Análise Individual",
])

# ── Tab 1: Importância média global ──────────────────────────────────────────
with tab1:
    st.subheader("Importância Média das Features (|SHAP|)")
    st.caption(
        "Média do valor absoluto dos SHAP values de todos os empregados. "
        "Quanto maior, mais relevante a feature é para o modelo globalmente."
    )

    feat_names = X_labeled.columns.tolist()
    mean_shap  = np.abs(shap_values.values).mean(axis=0)

    df_importance = (
        pd.DataFrame({"Feature": feat_names, "Importância": mean_shap})
        .sort_values("Importância", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(df_importance)))
    bars   = ax.barh(df_importance["Feature"], df_importance["Importância"],
                     color=colors, edgecolor="none", height=0.65)

    ax.set_xlabel("Importância média |SHAP|", fontsize=10)
    ax.set_title("Contribuição Global das Features", fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, df_importance["Importância"]):
        ax.text(
            val + max(df_importance["Importância"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8, color="#333"
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Tab 2: Beeswarm ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Impacto de Cada Feature em Todos os Empregados")
    st.caption(
        "Cada ponto é um empregado. A posição horizontal mostra se a feature **aumentou** "
        "(direita, vermelho) ou **diminuiu** (esquerda, azul) o risco previsto. "
        "A cor indica o valor real da feature: vermelho = alto, azul = baixo."
    )

    order          = np.argsort(mean_shap)
    shap_vals_ord  = shap_values.values[:, order]
    feat_vals_ord  = X_labeled.values[:, order]
    feat_names_ord = [feat_names[i] for i in order]

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for i, (sv, fv) in enumerate(zip(shap_vals_ord.T, feat_vals_ord.T)):
        fv_min, fv_max = fv.min(), fv.max()
        fv_norm = (fv - fv_min) / (fv_max - fv_min + 1e-9)
        jitter  = np.random.uniform(-0.3, 0.3, size=len(sv))
        sc = ax2.scatter(
            sv, i + jitter,
            c=fv_norm, cmap="RdBu_r", vmin=0, vmax=1,
            s=14, alpha=0.6, linewidths=0,
        )

    ax2.axvline(0, color="#555", linewidth=0.8, linestyle="--")
    ax2.set_yticks(range(len(feat_names_ord)))
    ax2.set_yticklabels(feat_names_ord, fontsize=8)
    ax2.set_xlabel("Valor SHAP (impacto na previsão)", fontsize=10)
    ax2.set_title("Beeswarm — Distribuição dos SHAP Values", fontsize=12,
                  fontweight="bold", pad=10)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)

    cbar = fig2.colorbar(sc, ax=ax2, pad=0.01, fraction=0.015)
    cbar.set_label("Valor da feature\n(azul=baixo · vermelho=alto)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ── Tab 3: Análise individual ─────────────────────────────────────────────────
with tab3:
    st.subheader("Explicação Individual por Empregado")
    st.caption(
        "Selecione um empregado para ver quais features mais contribuíram "
        "para o score de risco **dele especificamente**."
    )

    lista_empregados = features["MAT"].tolist()
    empregado_sel = st.selectbox(
        "Selecione a matrícula do empregado:",
        options=lista_empregados,
        format_func=lambda m: (
            f"{m}  "
            f"(Score: {features.loc[features['MAT']==m, 'score_risco'].values[0]:.1f}"
            f"  |  {features.loc[features['MAT']==m, 'nivel_risco'].values[0]})"
        )
    )

    idx_emp   = features.index[features["MAT"] == empregado_sel].tolist()[0]
    sv_emp    = shap_values.values[idx_emp]
    fv_emp    = X_labeled.iloc[idx_emp].values
    pred_emp  = features.loc[idx_emp, "atestados_previstos"]
    score_emp = features.loc[idx_emp, "score_risco"]
    risco_emp = features.loc[idx_emp, "nivel_risco"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Score de Risco",        f"{score_emp:.1f} / 100")
    c2.metric("Previsão de Atestados", f"{pred_emp:.2f}")
    c3.metric("Nível de Risco",        risco_emp)

    st.markdown("---")

    df_emp = pd.DataFrame({
        "Feature": feat_names,
        "SHAP":    sv_emp,
        "Valor":   fv_emp,
        "Label":   X_labeled.columns.tolist(),
    }).sort_values("SHAP", key=abs, ascending=True).tail(14)

    cores = ["#d73027" if v > 0 else "#4575b4" for v in df_emp["SHAP"]]

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    bars3 = ax3.barh(df_emp["Label"], df_emp["SHAP"],
                     color=cores, edgecolor="none", height=0.65)

    ax3.axvline(0, color="#555", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("Contribuição SHAP (impacto na previsão)", fontsize=10)
    ax3.set_title(
        f"Matrícula {empregado_sel} — Contribuição de cada feature",
        fontsize=12, fontweight="bold", pad=10
    )
    ax3.spines[["top", "right", "left"]].set_visible(False)
    ax3.tick_params(axis="y", labelsize=8)
    ax3.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax3.set_axisbelow(True)

    for bar, row in zip(bars3, df_emp.itertuples()):
        x_pos = bar.get_width()
        sign  = "+" if x_pos >= 0 else ""
        label = f"{sign}{x_pos:.3f}  (val={row.Valor:.2f})"
        ax3.text(
            x_pos + (0.003 if x_pos >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            label, va="center",
            ha="left" if x_pos >= 0 else "right",
            fontsize=7.5, color="#222"
        )

    legenda = [
        mpatches.Patch(color="#d73027", label="Aumenta o risco (SHAP > 0)"),
        mpatches.Patch(color="#4575b4", label="Reduz o risco (SHAP < 0)"),
    ]
    ax3.legend(handles=legenda, fontsize=8, loc="lower right")
    ax3.set_xlim(df_emp["SHAP"].min() * 1.35, df_emp["SHAP"].max() * 1.35)

    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown("##### Detalhamento por feature")
    df_detalhe = pd.DataFrame({
        "Feature":           X_labeled.columns.tolist(),
        "Valor Real":        X_labeled.iloc[idx_emp].values,
        "Contribuição SHAP": sv_emp,
    }).sort_values("Contribuição SHAP", key=abs, ascending=False).reset_index(drop=True)

    st.dataframe(
        df_detalhe.style
            .format({"Valor Real": "{:.3f}", "Contribuição SHAP": "{:+.4f}"})
            .background_gradient(subset=["Contribuição SHAP"], cmap="RdBu_r"),
        use_container_width=True,
        height=460,
    )
