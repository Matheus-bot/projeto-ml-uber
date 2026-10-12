"""Dashboard do projeto de previsão de tarifas Uber.

Execução (a partir da pasta raiz do projeto):
    streamlit run app/dashboard.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PASTA_DADOS = BASE_DIR / "data"
PASTA_MODELOS = BASE_DIR / "app"

st.set_page_config(page_title="Previsão de Tarifas Uber", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; max-width: 1400px; }
        h1 { font-weight: 700; letter-spacing: -0.02em; }
        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-left: 4px solid #FFD400;
            border-radius: 10px;
            padding: 14px 18px;
        }
        [data-testid="stMetricLabel"] { font-size: 0.8rem; opacity: 0.75; }
        .stButton button {
            background-color: #FFD400; color: #000000; font-weight: 700;
            border: none; border-radius: 8px; padding: 0.6rem 2.5rem;
        }
        .stButton button:hover { background-color: #E6BF00; color: #000000; }
    </style>
    """,
    unsafe_allow_html=True,
)

AEROPORTOS = {
    "JFK": (-73.7781, 40.6413),
    "LGA": (-73.8740, 40.7769),
    "EWR": (-74.1745, 40.6895),
}

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


@st.cache_data
def carregar_csv(nome):
    return pd.read_csv(PASTA_DADOS / nome)


@st.cache_resource
def carregar_modelo():
    return joblib.load(PASTA_MODELOS / "modelo_rf.pkl"), joblib.load(PASTA_MODELOS / "scaler.pkl")


def formatar_metricas(df):
    formatos = {
        "R2 treino": "{:.4f}", "R2 teste": "{:.4f}", "Diferença R2": "{:+.4f}",
        "MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}",
        "Tempo de treino (s)": "{:.2f}", "Tempo de previsão (s)": "{:.2f}",
        "Corridas no treino": "{:,.0f}",
    }
    return df.style.format({k: v for k, v in formatos.items() if k in df.columns})


# --------------------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------------------
df = carregar_csv("uber_tratado.csv")
exp1 = carregar_csv("metricas_exp1.csv")
exp2 = carregar_csv("metricas_exp2.csv")
cv = carregar_csv("validacao_cruzada.csv")
modelo, scaler = carregar_modelo()

rf_exp1 = exp1[exp1["Modelo"] == "Random Forest"].iloc[0]

st.title("Previsão de Tarifas de Corridas Uber")
st.caption(
    "Aprendizagem de Máquina — Desafio do Oráculo Preditivo — Grupo 03 · "
    "Uber Fares Dataset, Nova York, 2009 a 2015"
)

aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "Visão Geral", "Análise Exploratória", "Ajuste dos Modelos",
    "Comparação de Modelos", "Simulador de Tarifa",
])

# --------------------------------------------------------------------------------------
# 1. Visão geral
# --------------------------------------------------------------------------------------
with aba1:
    st.header("Visão geral da base")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Corridas após a limpeza", f"{len(df):,}".replace(",", "."))
    c2.metric("Tarifa média", f"US$ {df['fare_amount'].mean():.2f}")
    c3.metric("Tarifa mediana", f"US$ {df['fare_amount'].median():.2f}")
    c4.metric("Distância média", f"{df['distancia_km'].mean():.2f} km")
    c5.metric("Corridas de aeroporto", f"{df['eh_aeroporto'].mean() * 100:.1f}%")

    st.subheader("Fluxo do projeto")
    st.markdown(
        """
        | Etapa | Resultado |
        |---|---|
        | Base original | 200.000 corridas, 9 colunas |
        | Limpeza (tarifas impossíveis, passageiros, coordenadas fora de NY, distância < 100 m) | 191.735 corridas (4,13% removidas) |
        | Engenharia de features | distância (Haversine), ano, mês, dia da semana, hora, indicador de aeroporto |
        | Divisão treino/teste | 80% / 20% (153.388 / 38.347), `random_state = 42` |
        | Modelos | Dummy (referência), Regressão Linear, Random Forest, SVR |
        """
    )

    st.subheader("Distribuição das tarifas")
    valores, limites = np.histogram(df["fare_amount"], bins=60)
    st.bar_chart(pd.DataFrame({"Tarifa (US$)": limites[:-1].round(1), "Corridas": valores}),
                 x="Tarifa (US$)", y="Corridas")
    st.info(
        f"A tarifa tem assimetria à direita: média de US$ {df['fare_amount'].mean():.2f} contra mediana "
        f"de US$ {df['fare_amount'].median():.2f}. A maioria das corridas é barata, com uma cauda de "
        "corridas longas e de aeroporto."
    )

    st.subheader("Dois regimes de preço")
    regimes = df.groupby("eh_aeroporto").agg(
        Corridas=("fare_amount", "size"),
        **{"Tarifa média (US$)": ("fare_amount", "mean"), "Distância média (km)": ("distancia_km", "mean")},
    )
    regimes["Custo por km (US$)"] = regimes["Tarifa média (US$)"] / regimes["Distância média (km)"]
    regimes.index = ["Não aeroporto", "Aeroporto"]
    st.dataframe(regimes.style.format({"Corridas": "{:,.0f}", "Tarifa média (US$)": "{:.2f}",
                                       "Distância média (km)": "{:.2f}", "Custo por km (US$)": "{:.2f}"}),
                 width="stretch")

# --------------------------------------------------------------------------------------
# 2. Análise exploratória
# --------------------------------------------------------------------------------------
with aba2:
    st.header("Análise exploratória")

    matriz = df.corr()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlação com a tarifa")
        st.bar_chart(matriz["fare_amount"].drop("fare_amount").sort_values(ascending=False))
        st.caption(
            "A distância, criada pela fórmula de Haversine, tem correlação de 0,90 com a tarifa. "
            "A melhor variável original chega a 0,44."
        )
    with col2:
        st.subheader("Tarifa média por hora do dia")
        st.line_chart(df.groupby("hora")["fare_amount"].mean())
        st.caption(
            "A hora tem correlação quase nula com a tarifa, mas a tarifa média varia ao longo do dia "
            "(pico na madrugada). A relação é cíclica, não linear."
        )

    st.subheader("Matriz de correlação")
    st.dataframe(matriz.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1).format("{:.2f}"),
                 width="stretch")

    st.subheader("Distância × tarifa")
    amostra = df.sample(3000, random_state=42).assign(
        Tipo=lambda d: np.where(d["eh_aeroporto"] == 1, "Aeroporto", "Não aeroporto"))
    st.scatter_chart(amostra, x="distancia_km", y="fare_amount", color="Tipo")
    st.caption("Amostra de 3.000 corridas. As corridas de aeroporto formam faixas de tarifa fixa.")

# --------------------------------------------------------------------------------------
# 3. Ajuste dos modelos
# --------------------------------------------------------------------------------------
with aba3:
    st.header("Ajuste dos modelos")
    st.write(
        "O ajuste foi feito com validação cruzada de 5 partições sobre a amostra comum de 20.000 "
        "corridas. O conjunto de teste não participa do ajuste."
    )

    st.subheader("SVR — comparação de kernels")
    kernels = carregar_csv("comparacao_kernels.csv")
    st.dataframe(formatar_metricas(kernels[["Modelo", "R2 teste", "MAE", "RMSE", "MAPE (%)",
                                            "Tempo de treino (s)"]]),
                 width="stretch", hide_index=True)
    st.caption(
        "Mesmas condições (20.000 corridas, C = 100, epsilon = 0,1). O kernel linear fica equivalente à "
        "Regressão Linear; o RBF captura as relações não lineares."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SVR — ajuste de C e gamma")
        ajuste_svr = carregar_csv("ajuste_svr.csv")
        st.dataframe(ajuste_svr.style.format({"R2 médio": "{:.4f}", "Desvio": "{:.4f}", "Tempo (s)": "{:.0f}"}),
                     width="stretch", hide_index=True)
    with col2:
        st.subheader("Random Forest — GridSearchCV")
        ajuste_rf = carregar_csv("ajuste_rf.csv")
        st.dataframe(ajuste_rf.style.format({"R2 médio": "{:.6f}", "Desvio": "{:.4f}"}),
                     width="stretch", hide_index=True)

    st.info(
        "SVR: a melhor configuração foi C = 100 e gamma = 0,01, superior à configuração inicial "
        "(gamma = 'scale'). Random Forest: a configuração adotada (max_depth = 20, min_samples_leaf = 5) "
        "ficou empatada com a melhor e foi mantida por gerar árvores menores (parcimônia)."
    )

# --------------------------------------------------------------------------------------
# 4. Comparação de modelos
# --------------------------------------------------------------------------------------
with aba4:
    st.header("Comparação de modelos")

    experimento = st.radio(
        "Experimento",
        ["Experimento 1 — máximo de dados viável", "Experimento 2 — mesmas 20.000 corridas"],
        horizontal=True,
    )
    tabela = exp1 if experimento.startswith("Experimento 1") else exp2
    nome_exp = "Experimento 1" if experimento.startswith("Experimento 1") else "Experimento 2"

    if nome_exp == "Experimento 1":
        st.caption("Regressão Linear e Random Forest com 153.388 corridas; SVR com 20.000 (limite computacional). "
                   "Todos avaliados nas mesmas 38.347 corridas de teste.")
    else:
        st.caption("Os três modelos treinados com as mesmas 20.000 corridas e avaliados nas mesmas 38.347 corridas de teste.")

    st.subheader("Métricas no conjunto de teste")
    colunas = ["Modelo", "Corridas no treino", "R2 teste", "MAE", "RMSE", "MAPE (%)"]
    destaque = tabela[tabela["Modelo"] != "Dummy (média)"]
    st.dataframe(
        formatar_metricas(tabela[colunas])
        .highlight_max(subset=["R2 teste"], color="#FFE680", props="color: black")
        .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#FFE680", props="color: black"),
        width="stretch", hide_index=True,
    )
    st.caption("Destaque: melhor valor de cada métrica. O Dummy prevê sempre a tarifa média e serve de referência mínima.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("R² no teste")
        st.bar_chart(destaque.set_index("Modelo")["R2 teste"])
    with col2:
        st.subheader("Erro médio absoluto (US$)")
        st.bar_chart(destaque.set_index("Modelo")["MAE"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overfitting (R² treino × teste)")
        st.dataframe(formatar_metricas(destaque[["Modelo", "R2 treino", "R2 teste", "Diferença R2"]]),
                     width="stretch", hide_index=True)
    with col2:
        st.subheader("Custo computacional")
        st.dataframe(formatar_metricas(destaque[["Modelo", "Tempo de treino (s)", "Tempo de previsão (s)"]]),
                     width="stretch", hide_index=True)
        st.caption("Tempos medidos na máquina em que o notebook foi executado; servem para comparação relativa.")

    st.subheader("Validação cruzada (5 partições)")
    cv_exp = cv[cv["Experimento"] == nome_exp].drop(columns=["Experimento"])
    st.dataframe(cv_exp.style.format({c: "{:.4f}" for c in cv_exp.columns if c not in ("Modelo", "Corridas")}
                                     | {"Corridas": "{:,.0f}"}),
                 width="stretch", hide_index=True)

    st.subheader("Erro médio por faixa de preço real (Experimento 1)")
    erro_faixa = carregar_csv("erro_por_faixa.csv")
    st.dataframe(erro_faixa.style.format({c: "{:.2f}" for c in erro_faixa.columns[2:]} | {"Corridas": "{:,.0f}"}),
                 width="stretch", hide_index=True)
    st.caption(
        "O erro cresce com o preço (heterocedasticidade). O SVR erra menos nas corridas de até US$ 10, "
        "a maioria, o que explica seu melhor MAPE; o Random Forest erra menos em todas as outras faixas."
    )

    st.subheader("Valores reais × valores preditos (Experimento 1)")
    st.image(str(PASTA_DADOS / "real_vs_predito.png"), width="stretch")
    st.subheader("Resíduos (Experimento 1)")
    st.image(str(PASTA_DADOS / "residuos.png"), width="stretch")

    st.success(
        "Veredito: Random Forest. Tem o maior R² e o menor RMSE nos dois experimentos e é o único "
        "modelo de alto desempenho que aproveita a base completa. Nas mesmas 20.000 corridas, o SVR "
        "ajustado fica praticamente empatado, mas não escala para a base inteira e é o mais lento para prever."
    )

# --------------------------------------------------------------------------------------
# 5. Simulador
# --------------------------------------------------------------------------------------
with aba5:
    st.header("Simulador de tarifa")
    st.write("Previsão feita pelo Random Forest treinado com a base completa (Experimento 1).")

    modo = st.radio("Como informar a corrida", ["Locais conhecidos", "Coordenadas"], horizontal=True)

    col1, col2 = st.columns(2)
    if modo == "Locais conhecidos":
        with col1:
            origem = st.selectbox("Origem", list(LOCAIS), index=0)
        with col2:
            destino = st.selectbox("Destino", list(LOCAIS), index=10)
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

    fora_ny = not all([-74.3 <= lon_o <= -73.7, 40.5 <= lat_o <= 41.0,
                       -74.3 <= lon_d <= -73.7, 40.5 <= lat_d <= 41.0])

    if st.button("Prever tarifa"):
        distancia = float(haversine(lon_o, lat_o, lon_d, lat_d))
        aeroporto = eh_corrida_de_aeroporto(lon_o, lat_o, lon_d, lat_d)

        if fora_ny:
            st.warning("Coordenadas fora da região de Nova York usada no treinamento (longitude −74,3 a −73,7; "
                       "latitude 40,5 a 41,0). A previsão não é confiável.")
        if distancia < 0.1:
            st.warning("Distância menor que 100 m. Corridas assim foram removidas na limpeza.")

        entrada = pd.DataFrame([{
            "pickup_longitude": lon_o, "pickup_latitude": lat_o,
            "dropoff_longitude": lon_d, "dropoff_latitude": lat_d,
            "passenger_count": passageiros, "distancia_km": distancia,
            "ano": ano, "mes": mes, "dia_semana": dia_semana, "hora": hora,
            "eh_aeroporto": aeroporto,
        }])[COLUNAS_MODELO]
        previsao = float(modelo.predict(scaler.transform(entrada))[0])

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Tarifa prevista", f"US$ {previsao:.2f}")
        r2.metric("Distância em linha reta", f"{distancia:.2f} km")
        r3.metric("Corrida de aeroporto", "Sim" if aeroporto else "Não")
        st.caption(f"Erro médio do modelo no teste: ± US$ {rf_exp1['MAE']:.2f} (MAE).")

    st.divider()
    st.subheader("Sobre o modelo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (teste)", f"{rf_exp1['R2 teste']:.4f}")
    m2.metric("MAE", f"US$ {rf_exp1['MAE']:.2f}")
    m3.metric("RMSE", f"US$ {rf_exp1['RMSE']:.2f}")
    m4.metric("MAPE", f"{rf_exp1['MAPE (%)']:.2f}%")
