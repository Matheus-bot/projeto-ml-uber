import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Previsão de Tarifas Uber", layout="wide")

st.title("Previsão de Tarifas de Corridas Uber")
st.caption("Projeto de Aprendizagem de Máquina — Análise comparativa de modelos de regressão")

@st.cache_data
def carregar_dados():
    return pd.read_csv('data/uber_tratado.csv')

@st.cache_resource
def carregar_modelos():
    modelo = joblib.load('app/modelo_rf.pkl')
    scaler = joblib.load('app/scaler.pkl')
    return modelo, scaler

df = carregar_dados()
modelo, scaler = carregar_modelos()

aba1, aba2, aba3, aba4 = st.tabs([
    "Visão Geral",
    "Análise Exploratória",
    "Comparação de Modelos",
    "Simulador de Tarifa"
])

with aba1:
    st.header("Visão Geral da Base")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Corridas", f"{len(df):,}")
    col2.metric("Tarifa média", f"$ {df['fare_amount'].mean():.2f}")
    col3.metric("Distância média", f"{df['distancia_km'].mean():.2f} km")
    col4.metric("Corridas de aeroporto", f"{df['eh_aeroporto'].mean()*100:.1f}%")
    
    st.subheader("Distribuição das Tarifas")
    
    valores, bins = np.histogram(df['fare_amount'], bins=50)
    hist_df = pd.DataFrame({'Tarifa (USD)': bins[:-1].round(2), 'Frequência': valores})
    st.bar_chart(hist_df, x='Tarifa (USD)', y='Frequência')
    
    st.info(
        f"A variável alvo apresenta assimetria à direita: média de "
        f"$ {df['fare_amount'].mean():.2f} contra mediana de $ {df['fare_amount'].median():.2f}. "
        f"A maioria das corridas concentra-se em valores baixos, com cauda longa de corridas caras."
    )

with aba2:
    st.header("Análise Exploratória")
    
    st.subheader("Matriz de Correlação")
    
    matriz = df.corr()
    st.dataframe(
        matriz.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format("{:.2f}"),
        use_container_width=True
    )
    
    st.subheader("Correlação com a Tarifa")
    
    corr_alvo = matriz['fare_amount'].drop('fare_amount').sort_values(ascending=False)
    st.bar_chart(corr_alvo)
    
    st.info(
        "A variável `distancia_km`, criada por engenharia de features a partir da fórmula "
        "de Haversine, apresenta correlação de 0,90 com a tarifa — superior a qualquer "
        "variável original do dataset, cuja maior correlação é de 0,44."
    )
    
    st.subheader("Relação entre Distância e Tarifa")
    
    amostra = df.sample(3000, random_state=42)
    st.scatter_chart(amostra, x='distancia_km', y='fare_amount', color='eh_aeroporto')
    
    st.caption(
        "Amostra de 3.000 corridas. As corridas de aeroporto (em destaque) formam "
        "agrupamentos distintos, evidenciando regime tarifário próprio."
    )
with aba3:
    st.header("Comparação de Modelos")
    
    metricas = pd.read_csv('data/tabela_metricas.csv')
    
    st.subheader("Tabela Comparativa de Métricas")
    st.dataframe(
        metricas.style.highlight_max(subset=['R2'], color='#c8e6c9')
                      .highlight_min(subset=['MAE', 'MAPE (%)', 'RMSE'], color='#c8e6c9')
                      .format({'R2': '{:.4f}', 'MAE': '{:.4f}', 'MAPE (%)': '{:.2f}', 'RMSE': '{:.4f}'}),
        use_container_width=True,
        hide_index=True
    )
    
    st.caption("Células destacadas indicam o melhor resultado de cada métrica.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("R² por Modelo")
        st.bar_chart(metricas.set_index('Modelo')['R2'])
    
    with col2:
        st.subheader("Erro Médio Absoluto (USD)")
        st.bar_chart(metricas.set_index('Modelo')['MAE'])
    
    st.subheader("Valores Reais vs. Valores Preditos")
    st.image('data/real_vs_predito.png')

with aba4:
    st.header("Simulador de Tarifa")
    st.write("Informe os dados da corrida para obter a previsão do modelo Random Forest.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Origem")
        pickup_lon = st.number_input("Longitude de embarque", value=-73.9857, format="%.4f")
        pickup_lat = st.number_input("Latitude de embarque", value=40.7484, format="%.4f")
        passageiros = st.slider("Passageiros", 1, 6, 1)
    
    with col2:
        st.subheader("Destino")
        dropoff_lon = st.number_input("Longitude de desembarque", value=-73.7781, format="%.4f")
        dropoff_lat = st.number_input("Latitude de desembarque", value=40.6413, format="%.4f")
        hora = st.slider("Hora do dia", 0, 23, 14)
    
    col3, col4, col5 = st.columns(3)
    ano = col3.selectbox("Ano", [2009, 2010, 2011, 2012, 2013, 2014, 2015], index=6)
    mes = col4.selectbox("Mês", list(range(1, 13)), index=5)
    dia_semana = col5.selectbox(
        "Dia da semana",
        options=list(range(7)),
        format_func=lambda x: ['Segunda','Terça','Quarta','Quinta','Sexta','Sábado','Domingo'][x]
    )
    
    if st.button("Calcular Tarifa", type="primary"):
        
        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            dlon, dlat = lon2 - lon1, lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 6371 * 2 * np.arcsin(np.sqrt(a))
        
        distancia = haversine(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
        
        aeroportos = {'JFK': (-73.7781, 40.6413), 'LGA': (-73.8740, 40.7769), 'EWR': (-74.1745, 40.6895)}
        eh_aeroporto = 0
        for lon, lat in aeroportos.values():
            if haversine(pickup_lon, pickup_lat, lon, lat) < 2 or haversine(dropoff_lon, dropoff_lat, lon, lat) < 2:
                eh_aeroporto = 1
                break
        
        entrada = pd.DataFrame([{
            'pickup_longitude': pickup_lon,
            'pickup_latitude': pickup_lat,
            'dropoff_longitude': dropoff_lon,
            'dropoff_latitude': dropoff_lat,
            'passenger_count': passageiros,
            'distancia_km': distancia,
            'ano': ano,
            'mes': mes,
            'dia_semana': dia_semana,
            'hora': hora,
            'eh_aeroporto': eh_aeroporto
        }])
        
        entrada_esc = scaler.transform(entrada)
        previsao = modelo.predict(entrada_esc)[0]
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Tarifa prevista", f"$ {previsao:.2f}")
        c2.metric("Distância", f"{distancia:.2f} km")
        c3.metric("Aeroporto", "Sim" if eh_aeroporto else "Não")
        
        st.caption(f"Erro médio esperado do modelo: ± $ {1.65:.2f} (MAE no conjunto de teste)")