"""Dashboard do projeto de previsão de tarifas Uber.

Execução (a partir da pasta raiz do projeto):
    streamlit run app/dashboard.py
"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================================================
# Configuração
# ======================================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PASTA_DADOS = BASE_DIR / "data"
PASTA_MODELOS = BASE_DIR / "app"

st.set_page_config(page_title="Previsão de Tarifas Uber", layout="wide")


AMARELO = "#FFD400"
FUNDO = "#0F1115"
SUPERFICIE = "#171A21"
BORDA = "#2A2F3A"
TEXTO = "#E8E9EC"
TEXTO_2 = "#A3A9B5"
TEXTO_3 = "#6F7684"
CORES_MODELOS = {
    "Dummy (média)": "#5B616E",
    "Regressão Linear": "#3987E5",
    "Random Forest": "#C98500",
    "SVR (RBF ajustado)": "#199E70",
}
AZUL, AMBAR, VERDE, CINZA = "#3987E5", "#C98500", "#199E70", "#5B616E"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }}
    .stApp {{ background: {FUNDO}; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1320px; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* Cabeçalho */
    .hero {{
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, #1B1F27 0%, #13161C 60%);
        border: 1px solid {BORDA}; border-radius: 18px;
        padding: 30px 34px 28px; margin-bottom: 18px;
    }}
    .hero::after {{
        content: ""; position: absolute; right: -80px; top: -80px; width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(255,212,0,.18) 0%, rgba(255,212,0,0) 70%);
    }}
    .hero .eyebrow {{
        color: {AMARELO}; font-size: .74rem; font-weight: 700;
        letter-spacing: .14em; text-transform: uppercase; margin-bottom: 8px;
    }}
    .hero h1 {{
        color: {TEXTO}; font-size: 2.1rem; font-weight: 800; letter-spacing: -.025em;
        line-height: 1.15; margin: 0 0 8px; padding: 0;
    }}
    .hero p {{ color: {TEXTO_2}; font-size: .98rem; margin: 0; max-width: 760px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
    .chip {{
        font-size: .78rem; font-weight: 600; color: {TEXTO};
        background: rgba(255,255,255,.05); border: 1px solid {BORDA};
        padding: 6px 12px; border-radius: 999px;
    }}
    .chip b {{ color: {AMARELO}; font-weight: 700; }}

    /* Abas */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background: {SUPERFICIE}; border: 1px solid {BORDA};
        border-radius: 12px; padding: 5px; margin-bottom: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 42px; padding: 0 18px; border-radius: 8px;
        color: {TEXTO_2}; font-weight: 600; font-size: .9rem;
    }}
    .stTabs [aria-selected="true"] {{ background: {AMARELO} !important; color: #111 !important; }}
    .stTabs [aria-selected="true"] p {{ color: #111 !important; }}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

    /* Títulos de seção */
    .sec {{ margin: 26px 0 12px; }}
    .sec h3 {{ color: {TEXTO}; font-size: 1.12rem; font-weight: 700; margin: 0; padding: 0; letter-spacing: -.01em; }}
    .sec p {{ color: {TEXTO_2}; font-size: .86rem; margin: 3px 0 0; }}

    /* Cartões de indicador */
    .kpis {{ display: grid; gap: 12px; margin: 4px 0 6px; }}
    .kpi {{
        background: {SUPERFICIE}; border: 1px solid {BORDA}; border-radius: 14px;
        padding: 16px 18px 15px; position: relative;
    }}
    .kpi::before {{
        content: ""; position: absolute; left: 0; top: 16px; bottom: 16px; width: 3px;
        border-radius: 0 3px 3px 0; background: var(--acc, {AMARELO});
    }}
    .kpi .lbl {{ color: {TEXTO_2}; font-size: .74rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }}
    .kpi .val {{ color: {TEXTO}; font-size: 1.75rem; font-weight: 800; letter-spacing: -.02em; margin-top: 6px; line-height: 1.1; }}
    .kpi .sub {{ color: {TEXTO_3}; font-size: .78rem; margin-top: 5px; }}

    /* Cartões de modelo */
    .mcard {{
        background: {SUPERFICIE}; border: 1px solid {BORDA}; border-radius: 14px;
        padding: 16px 18px; height: 100%; border-top: 3px solid var(--acc);
    }}
    .mcard.best {{ border-color: {AMARELO}; box-shadow: 0 0 0 1px {AMARELO} inset; border-top-width: 3px; }}
    .mcard .nome {{ color: {TEXTO}; font-weight: 700; font-size: .98rem; display: flex; justify-content: space-between; align-items: center; }}
    .badge {{ background: {AMARELO}; color: #111; font-size: .66rem; font-weight: 800; letter-spacing: .06em;
              text-transform: uppercase; padding: 3px 8px; border-radius: 999px; }}
    .mcard .r2 {{ color: {TEXTO}; font-size: 2rem; font-weight: 800; letter-spacing: -.02em; margin: 10px 0 2px; }}
    .mcard .r2 span {{ color: {TEXTO_3}; font-size: .8rem; font-weight: 600; margin-left: 4px; }}
    .mcard table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: .84rem; }}
    .mcard td {{ padding: 5px 0; border-top: 1px solid {BORDA}; color: {TEXTO_2}; }}
    .mcard td:last-child {{ text-align: right; color: {TEXTO}; font-weight: 600; font-variant-numeric: tabular-nums; }}

    /* Etapas do projeto */
    .steps {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }}
    .step {{ background: {SUPERFICIE}; border: 1px solid {BORDA}; border-radius: 14px; padding: 14px 16px; }}
    .step .n {{ width: 24px; height: 24px; border-radius: 50%; background: {AMARELO}; color: #111;
                font-weight: 800; font-size: .78rem; display: flex; align-items: center; justify-content: center; }}
    .step .t {{ color: {TEXTO}; font-weight: 700; font-size: .9rem; margin-top: 10px; }}
    .step .d {{ color: {TEXTO_2}; font-size: .8rem; margin-top: 4px; line-height: 1.45; }}

    /* Comparação de regimes */
    .vs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}

    /* Destaques */
    .callout {{
        background: linear-gradient(135deg, rgba(255,212,0,.10), rgba(255,212,0,.03));
        border: 1px solid rgba(255,212,0,.35); border-radius: 14px; padding: 18px 20px; margin-top: 18px;
    }}
    .callout .t {{ color: {AMARELO}; font-weight: 800; font-size: .78rem; letter-spacing: .1em; text-transform: uppercase; }}
    .callout .b {{ color: {TEXTO}; font-size: .95rem; margin-top: 6px; line-height: 1.55; }}
    .note {{ color: {TEXTO_3}; font-size: .8rem; margin-top: 4px; }}

    /* Resultado do simulador */
    .result {{
        background: linear-gradient(135deg, #1E222B, #15181E); border: 1px solid {AMARELO};
        border-radius: 18px; padding: 24px 28px; display: grid;
        grid-template-columns: 1.3fr 1fr 1fr; gap: 18px; align-items: center; margin-top: 8px;
    }}
    .result .lbl {{ color: {TEXTO_2}; font-size: .74rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }}
    .result .big {{ color: {AMARELO}; font-size: 3rem; font-weight: 800; letter-spacing: -.03em; line-height: 1; margin-top: 6px; }}
    .result .mid {{ color: {TEXTO}; font-size: 1.5rem; font-weight: 700; margin-top: 6px; }}
    .result .hint {{ color: {TEXTO_3}; font-size: .8rem; margin-top: 6px; }}

    /* Componentes nativos */
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {BORDA} !important; border-radius: 14px !important; background: {SUPERFICIE}; }}
    [data-testid="stDataFrame"] {{ border: 1px solid {BORDA}; border-radius: 12px; overflow: hidden; }}
    .stButton > button, .stButton > button[kind] {{
        background: {AMARELO}; color: #111; font-weight: 800; border: none; border-radius: 10px;
        padding: .7rem 2.4rem; font-size: 1rem; transition: transform .08s ease, filter .15s ease;
    }}
    .stButton > button:hover {{ filter: brightness(.93); color: #111; }}
    .stButton > button:active {{ transform: scale(.98); }}
    .stButton > button p {{ color: #111; font-weight: 800; }}
    [data-testid="stImage"] img {{ border-radius: 12px; }}

    @media (max-width: 900px) {{
        .steps {{ grid-template-columns: 1fr 1fr; }}
        .result {{ grid-template-columns: 1fr; }}
        .hero h1 {{ font-size: 1.6rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Estilo dos gráficos (Altair), aplicado em cada gráfico
def estilizar(chart):
    return (
        chart.configure(background="transparent", font="Inter, system-ui, sans-serif",
                        locale={"number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["US$ ", ""]}})
        .configure_view(stroke="transparent")
        .configure_axis(labelColor=TEXTO_2, titleColor=TEXTO_2, labelFontSize=12, titleFontSize=12,
                        titleFontWeight=600, gridColor="#232833", domainColor=BORDA, tickColor=BORDA,
                        labelPadding=6, titlePadding=10)
        .configure_legend(labelColor=TEXTO_2, titleColor=TEXTO_2, labelFontSize=12, titleFontSize=12,
                          orient="top", symbolType="circle")
    )

# ======================================================================================
# Constantes e funções
# ======================================================================================
AEROPORTOS = {"JFK": (-73.7781, 40.6413), "LGA": (-73.8740, 40.7769), "EWR": (-74.1745, 40.6895)}

LOCAIS = {
    "Times Square": (-73.9855, 40.7580),
    "Empire State Building": (-73.9857, 40.7484),
    "Central Park": (-73.9654, 40.7829),
    "Grand Central Terminal": (-73.9772, 40.7527),
    "Wall Street": (-74.0089, 40.7061),
    "Ponte do Brooklyn": (-73.9969, 40.7061),
    "Greenwich Village": (-74.0027, 40.7336),
    "Chinatown": (-73.9970, 40.7158),
    "Upper West Side": (-73.9680, 40.7850),
    "Harlem": (-73.9465, 40.8116),
    "Aeroporto JFK": (-73.7781, 40.6413),
    "Aeroporto LaGuardia": (-73.8740, 40.7769),
    "Aeroporto Newark": (-74.1745, 40.6895),
}

DIAS = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

COLUNAS_MODELO = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "passenger_count", "distancia_km", "ano", "mes", "dia_semana", "hora", "eh_aeroporto",
]

DERIVADAS = {"distancia_km", "eh_aeroporto", "ano", "mes", "dia_semana", "hora"}


def haversine(lon1, lat1, lon2, lat2):
    """Distância em km entre dois pontos da superfície da Terra."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def eh_corrida_de_aeroporto(lon_o, lat_o, lon_d, lat_d):
    for lon, lat in AEROPORTOS.values():
        if haversine(lon_o, lat_o, lon, lat) < 2 or haversine(lon_d, lat_d, lon, lat) < 2:
            return 1
    return 0


def br(valor, casas=2):
    """Número no formato brasileiro (vírgula decimal, ponto de milhar)."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data
def carregar_csv(nome):
    return pd.read_csv(PASTA_DADOS / nome)


@st.cache_resource
def carregar_modelo():
    return joblib.load(PASTA_MODELOS / "modelo_rf.pkl"), joblib.load(PASTA_MODELOS / "scaler.pkl")


def secao(titulo, subtitulo=""):
    sub = f"<p>{subtitulo}</p>" if subtitulo else ""
    st.markdown(f'<div class="sec"><h3>{titulo}</h3>{sub}</div>', unsafe_allow_html=True)


def kpis(itens, colunas=None):
    colunas = colunas or len(itens)
    html = "".join(
        f'<div class="kpi" style="--acc:{i.get("cor", AMARELO)}"><div class="lbl">{i["rotulo"]}</div>'
        f'<div class="val">{i["valor"]}</div><div class="sub">{i.get("sub", "")}</div></div>'
        for i in itens
    )
    st.markdown(f'<div class="kpis" style="grid-template-columns:repeat({colunas},1fr)">{html}</div>',
                unsafe_allow_html=True)


def grafico(chart, altura=320):
    st.altair_chart(estilizar(chart.properties(height=altura)), theme=None, width="stretch")


def formatar(df, casas=None):
    """Formata as colunas numéricas no padrão brasileiro."""
    padrao = {
        "R2 treino": 4, "R2 teste": 4, "Diferença R2": 4, "MAE": 2, "RMSE": 2, "MAPE (%)": 2,
        "Tempo de treino (s)": 2, "Tempo de previsão (s)": 2, "Corridas no treino": 0, "Corridas": 0,
        "R2 médio": 4, "Desvio": 4, "Tempo (s)": 0,
    }
    padrao.update(casas or {})
    return df.style.format({c: (lambda v, n=n: br(v, n)) for c, n in padrao.items() if c in df.columns})


# ======================================================================================
# Dados
# ======================================================================================
df = carregar_csv("uber_tratado.csv")
exp1 = carregar_csv("metricas_exp1.csv")
exp2 = carregar_csv("metricas_exp2.csv")
cv = carregar_csv("validacao_cruzada.csv")
modelo, scaler = carregar_modelo()

rf1 = exp1[exp1["Modelo"] == "Random Forest"].iloc[0]
lr1 = exp1[exp1["Modelo"] == "Regressão Linear"].iloc[0]
dummy1 = exp1[exp1["Modelo"] == "Dummy (média)"].iloc[0]

# ======================================================================================
# Cabeçalho
# ======================================================================================
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Aprendizagem de Máquina · Desafio do Oráculo Preditivo · Grupo 03</div>
      <h1>Previsão de Tarifas de Corridas Uber</h1>
      <p>Comparação de Regressão Linear, Random Forest e SVR na previsão do valor de corridas em Nova York
      (Uber Fares Dataset, 2009 a 2015).</p>
      <div class="chips">
        <span class="chip">Modelo recomendado: <b>Random Forest</b></span>
        <span class="chip">R² no teste: <b>{br(rf1['R2 teste'], 4)}</b></span>
        <span class="chip">Erro médio: <b>US$ {br(rf1['MAE'])}</b> por corrida</span>
        <span class="chip">{br(len(df), 0)} corridas analisadas</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "Visão Geral", "Análise Exploratória", "Ajuste dos Modelos", "Comparação de Modelos", "Simulador de Tarifa",
])

# ======================================================================================
# 1. Visão geral
# ======================================================================================
with aba1:
    kpis([
        {"rotulo": "Corridas após a limpeza", "valor": br(len(df), 0), "sub": "de 200.000 (4,13% removidas)"},
        {"rotulo": "Tarifa média", "valor": f"US$ {br(df['fare_amount'].mean())}",
         "sub": f"mediana US$ {br(df['fare_amount'].median())}"},
        {"rotulo": "Distância média", "valor": f"{br(df['distancia_km'].mean())} km", "sub": "em linha reta (Haversine)"},
        {"rotulo": "Corridas de aeroporto", "valor": f"{br(df['eh_aeroporto'].mean() * 100, 1)}%",
         "sub": f"{br(df['eh_aeroporto'].sum(), 0)} corridas"},
    ])

    secao("Etapas do projeto")
    etapas = [
        ("Dados", "200.000 corridas e 9 colunas do Uber Fares Dataset."),
        ("Limpeza", "Tarifas impossíveis, passageiros inválidos, coordenadas fora de NY e distâncias < 100 m."),
        ("Features", "Distância (Haversine), ano, mês, dia da semana, hora e indicador de aeroporto."),
        ("Modelagem", "Dummy, Regressão Linear, Random Forest e SVR, com ajuste por validação cruzada."),
        ("Avaliação", "R², MAE, RMSE e MAPE em 38.347 corridas de teste, em dois experimentos."),
    ]
    st.markdown(
        '<div class="steps">' + "".join(
            f'<div class="step"><div class="n">{i}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
            for i, (t, d) in enumerate(etapas, 1)) + "</div>",
        unsafe_allow_html=True,
    )

    secao("Distribuição das tarifas",
          "Assimetria à direita: a maioria das corridas é barata, com uma cauda de corridas longas e de aeroporto.")
    with st.container(border=True):
        media, mediana = df["fare_amount"].mean(), df["fare_amount"].median()
        hist = (
            alt.Chart(df[["fare_amount"]])
            .mark_bar(color=AMBAR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("fare_amount:Q", bin=alt.Bin(step=2, extent=[0, 100]), title="Tarifa (US$)"),
                y=alt.Y("count():Q", title="Corridas"),
                tooltip=[alt.Tooltip("fare_amount:Q", bin=alt.Bin(step=2, extent=[0, 100]), title="Faixa (US$)"),
                         alt.Tooltip("count():Q", title="Corridas", format=",")],
            )
        )
        linhas = pd.DataFrame({"valor": [mediana, media], "rotulo": [f"Mediana US$ {br(mediana)}", f"Média US$ {br(media)}"]})
        regra = alt.Chart(linhas).mark_rule(color=TEXTO, strokeDash=[4, 4], strokeWidth=1.5).encode(x="valor:Q")
        texto = alt.Chart(linhas).mark_text(align="left", dx=6, dy=-6, color=TEXTO, fontWeight=600).encode(
            x="valor:Q", y=alt.value(14 + 0), text="rotulo:N")
        texto2 = texto.transform_filter(alt.datum.rotulo == linhas["rotulo"][1]).encode(y=alt.value(32))
        texto1 = texto.transform_filter(alt.datum.rotulo == linhas["rotulo"][0]).mark_text(align="right", dx=-6, dy=-6, color=TEXTO, fontWeight=600)
        grafico(hist + regra + texto1 + texto2, 300)

    secao("Dois regimes de preço",
          "Corridas de aeroporto seguem outra lógica de preço: o JFK tem tarifa fixa para Manhattan.")
    reg = df.groupby("eh_aeroporto").agg(n=("fare_amount", "size"), tarifa=("fare_amount", "mean"),
                                         dist=("distancia_km", "mean"))
    reg["km"] = reg["tarifa"] / reg["dist"]
    col1, col2 = st.columns(2)
    for col, idx, nome, cor in [(col1, 0, "Demais corridas", AZUL), (col2, 1, "Corridas de aeroporto", AMBAR)]:
        with col:
            r = reg.loc[idx]
            st.markdown(f'<div class="note" style="margin:0 0 6px;font-weight:600;color:{TEXTO_2}">{nome} · '
                        f'{br(r["n"], 0)} corridas</div>', unsafe_allow_html=True)
            kpis([
                {"rotulo": "Tarifa média", "valor": f"US$ {br(r['tarifa'])}", "cor": cor},
                {"rotulo": "Distância média", "valor": f"{br(r['dist'])} km", "cor": cor},
                {"rotulo": "Custo por km", "valor": f"US$ {br(r['km'])}", "cor": cor},
            ])

# ======================================================================================
# 2. Análise exploratória
# ======================================================================================
with aba2:
    matriz = df.corr()
    corr = (matriz["fare_amount"].drop("fare_amount").rename("r").reset_index()
            .rename(columns={"index": "Variável"}))
    corr["Origem"] = np.where(corr["Variável"].isin(DERIVADAS), "Criada no projeto", "Original do dataset")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        secao("Correlação com a tarifa", "A distância criada chega a 0,90; a melhor variável original, a 0,44.")
        with st.container(border=True):
            barras = alt.Chart(corr).mark_bar(cornerRadiusEnd=3, height=16).encode(
                y=alt.Y("Variável:N", sort="-x", title=None),
                x=alt.X("r:Q", title="Correlação de Pearson", scale=alt.Scale(domain=[-0.3, 1])),
                color=alt.Color("Origem:N", scale=alt.Scale(domain=["Criada no projeto", "Original do dataset"],
                                                             range=[AMBAR, AZUL]), title=None),
                tooltip=["Variável", "Origem", alt.Tooltip("r:Q", title="Correlação", format=".2f")],
            )
            rotulos = alt.Chart(corr).mark_text(align="left", dx=5, color=TEXTO_2, fontSize=11).encode(
                y=alt.Y("Variável:N", sort="-x"), x=alt.X("r:Q"),
                text=alt.Text("r:Q", format=".2f"),
            ).transform_filter(alt.datum.r >= 0)
            grafico(barras + rotulos, 380)
    with col2:
        secao("Tarifa média por hora do dia", "Correlação quase nula, mas efeito real: a relação é cíclica, não linear.")
        with st.container(border=True):
            hora = df.groupby("hora")["fare_amount"].mean().reset_index()
            base = alt.Chart(hora).encode(
                x=alt.X("hora:O", title="Hora do dia", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("fare_amount:Q", title="Tarifa média (US$)", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("fare_amount:Q", title="Tarifa média", format="$.2f")],
            )
            grafico(base.mark_line(color=AMBAR, strokeWidth=2.5)
                    + base.mark_circle(color=AMBAR, size=60, stroke=SUPERFICIE, strokeWidth=2), 380)

    secao("Matriz de correlação", "Azul: correlação positiva. Vermelho: negativa. Cinza: próxima de zero.")
    with st.container(border=True):
        m = matriz.reset_index().melt(id_vars="index", var_name="b", value_name="r").rename(columns={"index": "a"})
        ordem = list(matriz.columns)
        calor = alt.Chart(m).mark_rect(stroke=SUPERFICIE, strokeWidth=2, cornerRadius=3).encode(
            x=alt.X("b:N", sort=ordem, title=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("a:N", sort=ordem, title=None),
            color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#D95926", "#2A2F3A", "#3987E5"]),
                            legend=alt.Legend(title="Correlação", orient="right")),
            tooltip=[alt.Tooltip("a:N", title="Variável"), alt.Tooltip("b:N", title="com"),
                     alt.Tooltip("r:Q", title="Correlação", format=".2f")],
        )
        valores = alt.Chart(m).mark_text(fontSize=10.5, fontWeight=600).encode(
            x=alt.X("b:N", sort=ordem), y=alt.Y("a:N", sort=ordem), text=alt.Text("r:Q", format=".2f"),
            color=alt.condition("abs(datum.r) > 0.45", alt.value("#FFFFFF"), alt.value(TEXTO_2)),
        )
        grafico(calor + valores, 520)

    secao("Distância × tarifa", "Amostra de 3.000 corridas. As corridas de aeroporto formam faixas horizontais de tarifa fixa.")
    with st.container(border=True):
        amostra = df.sample(3000, random_state=42).assign(
            Tipo=lambda d: np.where(d["eh_aeroporto"] == 1, "Aeroporto", "Demais corridas"))
        disp = alt.Chart(amostra).mark_circle(size=26, opacity=0.65, stroke=SUPERFICIE, strokeWidth=0.6).encode(
            x=alt.X("distancia_km:Q", title="Distância (km)"),
            y=alt.Y("fare_amount:Q", title="Tarifa (US$)"),
            color=alt.Color("Tipo:N", scale=alt.Scale(domain=["Demais corridas", "Aeroporto"], range=[AZUL, AMBAR]),
                            title=None),
            tooltip=[alt.Tooltip("distancia_km:Q", title="Distância (km)", format=".2f"),
                     alt.Tooltip("fare_amount:Q", title="Tarifa", format="$.2f"), "Tipo:N"],
        )
        grafico(disp, 400)

# ======================================================================================
# 3. Ajuste dos modelos
# ======================================================================================
with aba3:
    st.markdown(
        f'<p class="note" style="font-size:.9rem;margin-top:10px">Ajuste com validação cruzada de 5 partições '
        f'sobre a amostra comum de 20.000 corridas. O conjunto de teste não participa do ajuste.</p>',
        unsafe_allow_html=True,
    )

    kernels = carregar_csv("comparacao_kernels.csv")
    secao("SVR — escolha do kernel", "Mesmas condições: 20.000 corridas, C = 100, epsilon = 0,1.")
    k_lin = kernels[kernels["Modelo"].str.contains("Linear")].iloc[0]
    k_rbf = kernels[kernels["Modelo"].str.contains("RBF")].iloc[0]
    col1, col2 = st.columns(2)
    for col, k, nome, cor in [(col1, k_lin, "Kernel linear", CINZA), (col2, k_rbf, "Kernel RBF (escolhido)", VERDE)]:
        with col:
            st.markdown(f'<div class="note" style="margin:0 0 6px;font-weight:600;color:{TEXTO_2}">{nome}</div>',
                        unsafe_allow_html=True)
            kpis([
                {"rotulo": "R² no teste", "valor": br(k["R2 teste"], 4), "cor": cor},
                {"rotulo": "MAE", "valor": f"US$ {br(k['MAE'])}", "cor": cor},
                {"rotulo": "Tempo de treino", "valor": f"{br(k['Tempo de treino (s)'], 0)} s", "cor": cor},
            ])
    st.markdown('<div class="note">O kernel linear fica equivalente à Regressão Linear. O RBF capta as relações '
                'não lineares, com melhor resultado e menos tempo.</div>', unsafe_allow_html=True)

    def ponto_com_desvio(dados, rotulo, escolhido, titulo_y):
        dados = dados.copy()
        dados["Escolhido"] = np.where(dados["Config"] == escolhido, "Configuração adotada", "Outras")
        dados["min"] = dados["R2 médio"] - dados["Desvio"]
        dados["max"] = dados["R2 médio"] + dados["Desvio"]
        ordem = dados.sort_values("R2 médio", ascending=False)["Config"].tolist()
        cor = alt.Color("Escolhido:N", scale=alt.Scale(domain=["Configuração adotada", "Outras"],
                                                         range=[AMBAR, "#7C8392"]), title=None)
        y = alt.Y("Config:N", sort=ordem, title=titulo_y)
        regra = alt.Chart(dados).mark_rule(strokeWidth=2, opacity=0.6).encode(
            y=y, x=alt.X("min:Q", title="R² médio ± desvio (validação cruzada)", scale=alt.Scale(zero=False), axis=alt.Axis(format=".2f", tickCount=6)),
            x2="max:Q", color=cor)
        pontos = alt.Chart(dados).mark_circle(size=110, opacity=1, stroke=SUPERFICIE, strokeWidth=2).encode(
            y=y, x="R2 médio:Q", color=cor,
            tooltip=[alt.Tooltip("Config:N", title=rotulo), alt.Tooltip("R2 médio:Q", format=".4f"),
                     alt.Tooltip("Desvio:Q", format=".4f")])
        return regra + pontos

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        secao("SVR — ajuste de C e gamma", "Melhor: C = 100 e gamma = 0,01.")
        with st.container(border=True):
            svr = carregar_csv("ajuste_svr.csv")
            svr["Config"] = "C = " + svr["C"].astype(str) + " · gamma = " + svr["gamma"].astype(str)
            grafico(ponto_com_desvio(svr, "Configuração", "C = 100 · gamma = 0.01", None), 300)
    with col2:
        secao("Random Forest — GridSearchCV", "Adotada: max_depth = 20 e min_samples_leaf = 5.")
        with st.container(border=True):
            rf = carregar_csv("ajuste_rf.csv")
            rf["Config"] = "profundidade " + rf["max_depth"].astype(str) + " · folha " + rf["min_samples_leaf"].astype(str)
            grafico(ponto_com_desvio(rf, "Configuração", "profundidade 20 · folha 5", None), 300)

    st.markdown(
        '<div class="callout"><div class="t">Leitura</div><div class="b">No SVR, o gamma menor fez cada '
        'previsão considerar mais vizinhos e quase eliminou o overfitting. No Random Forest, a profundidade 20 '
        'empatou com a profundidade livre e foi mantida por gerar árvores menores (parcimônia).</div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Ver tabelas completas"):
        st.dataframe(formatar(kernels[["Modelo", "R2 treino", "R2 teste", "MAE", "RMSE", "MAPE (%)",
                                       "Tempo de treino (s)", "Tempo de previsão (s)"]]),
                     width="stretch", hide_index=True)
        st.dataframe(formatar(svr.drop(columns="Config")),
                     width="stretch", hide_index=True)
        st.dataframe(formatar(rf.drop(columns="Config"), {"R2 médio": 6}),
                     width="stretch", hide_index=True)

# ======================================================================================
# 4. Comparação de modelos
# ======================================================================================
with aba4:
    escolha = st.radio(
        "Experimento",
        ["Experimento 1 · máximo de dados viável", "Experimento 2 · mesmas 20.000 corridas"],
        horizontal=True, label_visibility="collapsed",
    )
    e1 = escolha.startswith("Experimento 1")
    tabela = exp1 if e1 else exp2
    nome_exp = "Experimento 1" if e1 else "Experimento 2"
    st.markdown(
        '<p class="note" style="font-size:.88rem">' +
        ("Regressão Linear e Random Forest com 153.388 corridas; SVR com 20.000 (limite computacional). "
         if e1 else "Os três modelos treinados com as mesmas 20.000 corridas. ") +
        "Todos avaliados nas mesmas 38.347 corridas de teste.</p>",
        unsafe_allow_html=True,
    )

    modelos = tabela[tabela["Modelo"] != "Dummy (média)"].copy()
    dummy = tabela[tabela["Modelo"] == "Dummy (média)"].iloc[0]
    melhor = modelos.loc[modelos["R2 teste"].idxmax(), "Modelo"]

    cols = st.columns(4)
    for col, (_, r) in zip(cols, tabela.iterrows()):
        cor = CORES_MODELOS.get(r["Modelo"], CINZA)
        best = "best" if r["Modelo"] == melhor else ""
        badge = '<span class="badge">Melhor R²</span>' if best else ""
        nome = r["Modelo"].replace(" (RBF ajustado)", "").replace(" (média)", " (referência)")
        col.markdown(
            f'<div class="mcard {best}" style="--acc:{cor}"><div class="nome">{nome}{badge}</div>'
            f'<div class="r2">{br(abs(r["R2 teste"]) if abs(r["R2 teste"]) < 5e-5 else r["R2 teste"], 4)}<span>R²</span></div>'
            f'<table><tr><td>MAE</td><td>US$ {br(r["MAE"])}</td></tr>'
            f'<tr><td>RMSE</td><td>US$ {br(r["RMSE"])}</td></tr>'
            f'<tr><td>MAPE</td><td>{br(r["MAPE (%)"])}%</td></tr>'
            f'<tr><td>Treino</td><td>{br(r["Corridas no treino"], 0)} corridas</td></tr></table></div>',
            unsafe_allow_html=True,
        )

    cor_modelo = alt.Color("Modelo:N", scale=alt.Scale(domain=list(CORES_MODELOS), range=list(CORES_MODELOS.values())),
                           legend=None)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        secao("R² no teste", "Quanto da variação das tarifas o modelo explica (máximo 1).")
        with st.container(border=True):
            b = alt.Chart(modelos).encode(
                y=alt.Y("Modelo:N", sort="-x", title=None),
                x=alt.X("R2 teste:Q", title="R²", scale=alt.Scale(domain=[0, 1])),
                tooltip=["Modelo", alt.Tooltip("R2 teste:Q", format=".4f")],
            )
            grafico(b.mark_bar(cornerRadiusEnd=4, height=26).encode(color=cor_modelo)
                    + b.mark_text(align="left", dx=6, color=TEXTO, fontWeight=700).encode(text=alt.Text("R2 teste:Q", format=".4f")),
                    200)
    with col2:
        secao("Erro médio absoluto (MAE)", f"Em dólares por corrida. Linha tracejada: Dummy, US$ {br(dummy['MAE'])}.")
        with st.container(border=True):
            b = alt.Chart(modelos).encode(
                y=alt.Y("Modelo:N", sort="x", title=None),
                x=alt.X("MAE:Q", title="US$", scale=alt.Scale(domain=[0, max(6.5, dummy["MAE"] + 0.5)])),
                tooltip=["Modelo", alt.Tooltip("MAE:Q", format="$.2f")],
            )
            ref = alt.Chart(pd.DataFrame({"x": [dummy["MAE"]]})).mark_rule(color=TEXTO_3, strokeDash=[4, 4]).encode(x="x:Q")
            grafico(b.mark_bar(cornerRadiusEnd=4, height=26).encode(color=cor_modelo)
                    + b.mark_text(align="left", dx=6, color=TEXTO, fontWeight=700).encode(text=alt.Text("MAE:Q", format="$.2f"))
                    + ref, 200)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        secao("Overfitting: R² no treino × no teste", "Quanto maior a distância entre os pontos, maior o overfitting.")
        with st.container(border=True):
            longo = modelos.melt(id_vars="Modelo", value_vars=["R2 treino", "R2 teste"], var_name="Conjunto", value_name="R2")
            longo["Conjunto"] = longo["Conjunto"].map({"R2 treino": "Treino", "R2 teste": "Teste"})
            y = alt.Y("Modelo:N", title=None, sort=list(CORES_MODELOS))
            linha = alt.Chart(longo).mark_line(color="#4A5160", strokeWidth=3).encode(
                y=y, x=alt.X("R2:Q", title="R²", scale=alt.Scale(domain=[0.82, 0.95]), axis=alt.Axis(format=".2f", tickCount=6)), detail="Modelo:N")
            pontos = alt.Chart(longo).mark_point(size=130, filled=True, opacity=1).encode(
                y=y, x="R2:Q",
                shape=alt.Shape("Conjunto:N", scale=alt.Scale(domain=["Treino", "Teste"], range=["circle", "diamond"]), title=None),
                color=alt.Color("Conjunto:N", scale=alt.Scale(domain=["Treino", "Teste"], range=[TEXTO_2, AMARELO]), title=None),
                tooltip=["Modelo", "Conjunto", alt.Tooltip("R2:Q", format=".4f")],
            )
            grafico(linha + pontos, 220)
        dif = modelos.set_index("Modelo")["Diferença R2"]
        st.markdown('<div class="note">Diferença treino − teste: ' +
                    " · ".join(f"{m.replace(' (RBF ajustado)', '')} {d:+.4f}".replace(".", ",") for m, d in dif.items()) +
                    ". Abaixo de 0,10 é considerado aceitável.</div>", unsafe_allow_html=True)
    with col2:
        secao("Custo computacional", "Tempos medidos na máquina que executou o notebook; servem para comparação relativa.")
        st.dataframe(
            formatar(modelos[["Modelo", "Corridas no treino", "Tempo de treino (s)", "Tempo de previsão (s)"]]),
            width="stretch", hide_index=True,
        )
        st.markdown('<div class="note">O SVR é o mais lento para prever: compara cada corrida nova com os vetores '
                    'de suporte.</div>', unsafe_allow_html=True)

    secao("Validação cruzada (5 partições)",
          "Cada ponto é uma partição; o traço é a média. Mostra se o desempenho é estável em divisões diferentes dos dados.")
    with st.container(border=True):
        cvx = cv[cv["Experimento"] == nome_exp].copy()
        cvx["Modelo"] = cvx["Modelo"].replace({"SVR": "SVR (RBF ajustado)"})
        partes = cvx.melt(id_vars=["Modelo"], value_vars=[f"Parte {i}" for i in range(1, 6)], var_name="Partição", value_name="R2")
        y = alt.Y("Modelo:N", title=None, sort=list(CORES_MODELOS))
        pts = alt.Chart(partes).mark_circle(size=90, opacity=0.85, stroke=SUPERFICIE, strokeWidth=1.5).encode(
            y=y, x=alt.X("R2:Q", title="R² por partição", scale=alt.Scale(zero=False), axis=alt.Axis(format=".2f", tickCount=8)), color=cor_modelo,
            tooltip=["Modelo", "Partição", alt.Tooltip("R2:Q", format=".4f")])
        med = alt.Chart(cvx).mark_tick(thickness=3, size=30, color=TEXTO).encode(
            y=y, x="Média:Q", tooltip=["Modelo", alt.Tooltip("Média:Q", format=".4f"), alt.Tooltip("Desvio:Q", format=".4f")])
        grafico(pts + med, 200)

    secao("Erro médio por faixa de preço real (Experimento 1)",
          "O erro cresce com o preço (heterocedasticidade). O SVR só erra menos nas corridas de até US$ 10.")
    with st.container(border=True):
        faixa = carregar_csv("erro_por_faixa.csv")
        col_faixa = faixa.columns[0]
        ordem_faixas = faixa[col_faixa].tolist()
        fl = faixa.melt(id_vars=[col_faixa, "Corridas"], var_name="Modelo", value_name="MAE")
        fl["Faixa"] = fl[col_faixa]
        barras = alt.Chart(fl).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Faixa:N", sort=ordem_faixas, title="Tarifa real (US$)", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Modelo:N", sort=list(CORES_MODELOS)),
            y=alt.Y("MAE:Q", title="MAE (US$)"),
            color=alt.Color("Modelo:N", scale=alt.Scale(domain=list(CORES_MODELOS)[1:], range=list(CORES_MODELOS.values())[1:]),
                            title=None),
            tooltip=["Faixa", "Modelo", alt.Tooltip("MAE:Q", format="$.2f"), alt.Tooltip("Corridas:Q", format=",")],
        )
        grafico(barras, 340)

    secao("Valores reais × preditos e resíduos (Experimento 1)")
    col1, col2 = st.columns(2, gap="medium")
    col1.image(str(PASTA_DADOS / "real_vs_predito.png"), width="stretch")
    col2.image(str(PASTA_DADOS / "residuos.png"), width="stretch")

    st.markdown(
        '<div class="callout"><div class="t">Veredito</div><div class="b"><b>Random Forest.</b> Maior R² e menor '
        'RMSE nos dois experimentos e maior média na validação cruzada. Erra em média '
        f'US$ {br(rf1["MAE"])} por corrida, contra US$ {br(lr1["MAE"])} da Regressão Linear. O SVR empata na precisão '
        'nas mesmas 20.000 corridas, mas treina 7 vezes mais devagar e prevê cerca de 240 vezes mais devagar.</div></div>',
        unsafe_allow_html=True,
    )

# ======================================================================================
# 5. Simulador
# ======================================================================================
with aba5:
    st.markdown('<p class="note" style="font-size:.9rem;margin-top:10px">Previsão feita pelo Random Forest '
                'treinado com a base completa (Experimento 1).</p>', unsafe_allow_html=True)

    with st.container(border=True):
        modo = st.radio("Como informar a corrida", ["Locais conhecidos", "Coordenadas"], horizontal=True)
        col1, col2 = st.columns(2, gap="medium")
        if modo == "Locais conhecidos":
            origem = col1.selectbox("Origem", list(LOCAIS), index=0)
            destino = col2.selectbox("Destino", list(LOCAIS), index=10)
            lon_o, lat_o = LOCAIS[origem]
            lon_d, lat_d = LOCAIS[destino]
        else:
            with col1:
                lon_o = st.number_input("Longitude de origem", value=-73.9857, format="%.4f")
                lat_o = st.number_input("Latitude de origem", value=40.7484, format="%.4f")
            with col2:
                lon_d = st.number_input("Longitude de destino", value=-73.9855, format="%.4f")
                lat_d = st.number_input("Latitude de destino", value=40.7580, format="%.4f")

        c1, c2, c3, c4, c5 = st.columns(5)
        passageiros = c1.number_input("Passageiros", 1, 6, 1)
        hora = c2.number_input("Hora", 0, 23, 14)
        dia_semana = c3.selectbox("Dia da semana", range(7), format_func=lambda d: DIAS[d], index=2)
        mes = c4.selectbox("Mês", range(1, 13), index=5)
        ano = c5.selectbox("Ano", range(2009, 2016), index=6)
        prever = st.button("Prever tarifa")

    fora_ny = not all([-74.3 <= lon_o <= -73.7, 40.5 <= lat_o <= 41.0, -74.3 <= lon_d <= -73.7, 40.5 <= lat_d <= 41.0])

    if prever:
        distancia = float(haversine(lon_o, lat_o, lon_d, lat_d))
        aeroporto = eh_corrida_de_aeroporto(lon_o, lat_o, lon_d, lat_d)
        if fora_ny:
            st.warning("Coordenadas fora da região de Nova York usada no treinamento (longitude −74,3 a −73,7; "
                       "latitude 40,5 a 41,0). A previsão não é confiável.")
        if distancia < 0.1:
            st.warning("Distância menor que 100 m. Corridas assim foram removidas na limpeza.")

        entrada = pd.DataFrame([{
            "pickup_longitude": lon_o, "pickup_latitude": lat_o, "dropoff_longitude": lon_d, "dropoff_latitude": lat_d,
            "passenger_count": passageiros, "distancia_km": distancia, "ano": ano, "mes": mes,
            "dia_semana": dia_semana, "hora": hora, "eh_aeroporto": aeroporto,
        }])[COLUNAS_MODELO]
        previsao = float(modelo.predict(scaler.transform(entrada))[0])

        st.markdown(
            f'<div class="result">'
            f'<div><div class="lbl">Tarifa prevista</div><div class="big">US$ {br(previsao)}</div>'
            f'<div class="hint">Erro médio do modelo: ± US$ {br(rf1["MAE"])}</div></div>'
            f'<div><div class="lbl">Distância em linha reta</div><div class="mid">{br(distancia)} km</div></div>'
            f'<div><div class="lbl">Corrida de aeroporto</div><div class="mid">{"Sim" if aeroporto else "Não"}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not fora_ny:
            pontos = pd.DataFrame({"lat": [lat_o, lat_d], "lon": [lon_o, lon_d],
                                   "cor": ["#3987E5", "#FFD400"], "tam": [160, 160]})
            st.map(pontos, latitude="lat", longitude="lon", color="cor", size="tam", height=320)
            st.markdown('<div class="note">Azul: origem. Amarelo: destino.</div>', unsafe_allow_html=True)

    secao("Sobre o modelo", "Random Forest · 100 árvores · max_depth = 20 · min_samples_leaf = 5")
    kpis([
        {"rotulo": "R² no teste", "valor": br(rf1["R2 teste"], 4), "sub": "explica 88,8% da variação"},
        {"rotulo": "MAE", "valor": f"US$ {br(rf1['MAE'])}", "sub": "erro médio por corrida"},
        {"rotulo": "RMSE", "valor": f"US$ {br(rf1['RMSE'])}", "sub": "penaliza erros grandes"},
        {"rotulo": "MAPE", "valor": f"{br(rf1['MAPE (%)'])}%", "sub": "erro percentual médio"},
    ])
