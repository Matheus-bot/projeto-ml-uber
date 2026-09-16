# Previsão de Tarifas de Corridas Uber

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

Análise comparativa entre três algoritmos de regressão supervisionada aplicados à previsão de tarifas de corridas do Uber na região metropolitana de Nova York.

Projeto desenvolvido para a disciplina de **Aprendizagem de Máquina**.

---

## Sumário

- [Visão geral](#visão-geral)
- [Resultados](#resultados)
- [O dataset](#o-dataset)
- [Metodologia](#metodologia)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Dashboard](#dashboard)
- [Conclusões](#conclusões)
- [Tecnologias](#tecnologias)
- [Autores](#autores)

---

## Visão geral

O objetivo é estimar o valor da tarifa de uma corrida a partir de suas características — origem, destino, data, horário e número de passageiros — comparando três abordagens de regressão e identificando qual apresenta melhor poder de generalização.

O principal achado do trabalho é metodológico: **as variáveis originais do dataset apresentam correlação máxima de 0,44 com a tarifa**. A variável `distancia_km`, derivada das coordenadas geográficas pela fórmula de Haversine, alcança correlação de **0,90** e responde por **87,96%** da importância preditiva do melhor modelo.

---

## Resultados

| Modelo | R² | MAE (USD) | MAPE (%) | RMSE |
| :--- | ---: | ---: | ---: | ---: |
| Regressão Linear Múltipla | 0,8451 | 2,0533 | 24,76 | 3,6198 |
| **Random Forest Regressor** | **0,8880** | **1,6506** | 20,78 | **3,0775** |
| Support Vector Regressor (SVR) | 0,8674 | 1,7065 | **19,58** | 3,3490 |

*Métricas calculadas sobre o conjunto de teste (38.347 registros nunca vistos pelos modelos).*

### Validação cruzada (5 folds)

| Modelo | Média R² | Desvio padrão | Amplitude entre folds |
| :--- | ---: | ---: | ---: |
| Regressão Linear | 0,8426 | 0,0078 | 0,0203 |
| Random Forest | 0,8876 | 0,0083 | 0,0229 |
| SVR | 0,8446 | 0,0274 | 0,0801 |

O Random Forest apresentou desempenho superior em **todos os cinco folds** — seu pior resultado (0,8722) supera o melhor resultado da Regressão Linear (0,8475).

---

## O dataset

[Uber Fares Dataset](https://www.kaggle.com/code/moham9/uber-fares-group-project/input) — 200.000 corridas realizadas em Nova York entre 2009 e 2015.

**Variável alvo:** `fare_amount` (valor da tarifa em USD)

### Tratamento aplicado

| Anomalia identificada | Registros | Decisão |
| :--- | ---: | :--- |
| Tarifa negativa ou nula | 22 | Remoção |
| Tarifa acima de USD 100 | 84 | Remoção |
| Zero passageiros | 709 | Remoção |
| 208 passageiros | 1 | Remoção |
| Coordenadas fora da região de NY | 4.419 | Remoção |
| Coordenadas ausentes | 1 | Remoção |
| Distância inferior a 100 m | 3.096 | Remoção |

**Base final:** 191.735 registros (95,87% do original).

> **Sobre o corte em USD 100.** O critério do intervalo interquartil indicaria USD 22,25 como limite superior. Optou-se por não adotá-lo: corridas com destino aos aeroportos custam legitimamente entre USD 50 e USD 70, e a aplicação do critério estatístico eliminaria integralmente esse padrão.

> **Sobre as coordenadas nulas.** Cerca de 3.800 registros apresentavam coordenadas iguais a zero — ponto localizado no Golfo da Guiné. O padrão caracteriza falha de GPS em que o sistema grava zero em vez de valor ausente, razão pela qual a verificação por `isnull()` detectou apenas um registro.

---

## Metodologia

### Engenharia de features

Seis variáveis foram derivadas dos dados brutos:

| Variável criada | Origem | Justificativa |
| :--- | :--- | :--- |
| `distancia_km` | Fórmula de Haversine sobre as 4 coordenadas | Principal determinante da tarifa |
| `ano` | `pickup_datetime` | Reajustes tarifários entre 2009 e 2015 |
| `mes` | `pickup_datetime` | Sazonalidade e fluxo turístico |
| `dia_semana` | `pickup_datetime` | Variação de demanda semanal |
| `hora` | `pickup_datetime` | Horário de pico e tarifa dinâmica |
| `eh_aeroporto` | Proximidade de JFK, LGA ou EWR (raio de 2 km) | Regime tarifário distinto |

A distância não pode ser obtida por subtração direta das coordenadas: a extensão de um grau de longitude varia com a latitude — aproximadamente 111 km no Equador, reduzindo-se em direção aos polos. A fórmula de Haversine calcula a distância sobre a superfície esférica.

**Validação empírica da variável `eh_aeroporto`:**

| Grupo | Tarifa média | Distância média | Custo por km |
| :--- | ---: | ---: | ---: |
| Urbano | USD 9,75 | 2,79 km | USD 3,49 |
| Aeroporto | USD 37,53 | 13,13 km | USD 2,86 |

O custo por quilômetro é inferior no grupo aeroporto, evidenciando **dois regimes tarifários com inclinações distintas** — situação em que a variável de distância isoladamente seria insuficiente.

### Correlação com a variável alvo

| Variável | Correlação | Origem |
| :--- | ---: | :--- |
| `distancia_km` | 0,90 | Derivada |
| `eh_aeroporto` | 0,68 | Derivada |
| `pickup_longitude` | 0,44 | Original |
| `dropoff_longitude` | 0,32 | Original |
| `pickup_latitude` | −0,22 | Original |
| `dropoff_latitude` | −0,19 | Original |
| `ano` | 0,13 | Derivada |
| `mes` | 0,03 | Derivada |
| `passenger_count` | 0,01 | Original |
| `dia_semana` | 0,00 | Derivada |
| `hora` | −0,02 | Derivada |

As variáveis `hora` e `dia_semana` apresentam correlação próxima de zero, o que **não implica ausência de poder preditivo**: o coeficiente de Pearson mede exclusivamente associações lineares, e tais variáveis possuem natureza cíclica. A análise de importância no Random Forest confirmou essa hipótese — `hora` obteve importância de 0,0082, superior à de `pickup_latitude`.

### Multicolinearidade

Verificada por Variance Inflation Factor. Todas as variáveis apresentaram VIF inferior a 2,5, muito abaixo do limiar convencional de 5.

| Variável | VIF |
| :--- | ---: |
| `eh_aeroporto` | 2,46 |
| `distancia_km` | 1,96 |
| `pickup_longitude` | 1,71 |
| `dropoff_longitude` | 1,32 |
| `dropoff_latitude` | 1,28 |
| `pickup_latitude` | 1,27 |
| Demais variáveis | ≤ 1,01 |

### Pré-processamento

| Etapa | Procedimento | Justificativa |
| :--- | :--- | :--- |
| Divisão | 80% treino / 20% teste, `random_state=42` | Reprodutibilidade e comparabilidade entre modelos |
| Escalonamento | `StandardScaler` (média 0, desvio 1) | Obrigatório para o kernel RBF do SVR |
| Ajuste do scaler | `fit_transform` apenas no treino | Prevenção de *data leakage* |
| Validação | Cross-validation com 5 partições | Verificação de estabilidade das métricas |

A padronização foi preferida à normalização (`MinMaxScaler`) pela sensibilidade desta última a valores extremos. O kernel RBF computa similaridade por distância euclidiana — sem padronização, a variável `ano` (ordem de 10³) dominaria integralmente o cálculo, anulando a contribuição das demais.

### Configuração dos modelos

| Modelo | Hiperparâmetros | Observação |
| :--- | :--- | :--- |
| `LinearRegression` | Padrão | Baseline estatístico |
| `RandomForestRegressor` | `n_estimators=100`, `max_depth=20`, `min_samples_leaf=5` | Profundidade limitada como controle de overfitting |
| `SVR` | `kernel='rbf'`, `C=100`, `gamma='scale'`, `epsilon=0.1` | Treinado em amostra de 20.000 registros |

O SVR possui complexidade entre O(n²) e O(n³), o que inviabiliza o treinamento sobre os 153.388 registros disponíveis. A avaliação, contudo, foi realizada sobre o conjunto de teste integral, preservando a comparabilidade.

---

## Estrutura do repositório

```
projeto-ml-uber/
│
├── data/
│   ├── uber.csv                  Dataset bruto (não versionado)
│   ├── uber_tratado.csv          Base após limpeza e feature engineering
│   ├── tabela_metricas.csv       Métricas comparativas dos três modelos
│   ├── real_vs_predito.png       Painel de dispersão dos três modelos
│   └── residuos.png              Análise de resíduos
│
├── notebooks/
│   └── 01_eda.ipynb              EDA, tratamento, modelagem e avaliação
│
├── app/
│   ├── dashboard.py              Aplicação Streamlit
│   ├── modelo_rf.pkl             Random Forest serializado
│   ├── modelo_lr.pkl             Regressão Linear serializada
│   └── scaler.pkl                StandardScaler ajustado no treino
│
├── .streamlit/
│   └── config.toml               Tema da aplicação
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Como executar

### Pré-requisitos

- Python 3.10 ou superior
- Git

### Instalação

```bash
git clone https://github.com/SEU-USUARIO/projeto-ml-uber.git
cd projeto-ml-uber
```

Crie e ative o ambiente virtual:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### Obter o dataset

O arquivo `uber.csv` não está versionado. Baixe pelo [Kaggle](https://www.kaggle.com/code/moham9/uber-fares-group-project/input) e coloque em `data/uber.csv`.

### Executar o notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Ou abra diretamente no VS Code selecionando o kernel `.venv`.

> Execute as células na ordem, de cima para baixo. O notebook gera os arquivos `uber_tratado.csv`, `tabela_metricas.csv`, os gráficos e os modelos serializados consumidos pelo dashboard.

### Executar o dashboard

```bash
streamlit run app/dashboard.py
```

A aplicação abre automaticamente em `http://localhost:8501`.

---

## Dashboard

Aplicação Streamlit com quatro seções:

| Aba | Conteúdo |
| :--- | :--- |
| **Visão Geral** | Indicadores da base e distribuição da variável alvo |
| **Análise Exploratória** | Matriz de correlação e relação distância × tarifa |
| **Comparação de Modelos** | Tabela de métricas, gráficos de performance e resíduos |
| **Simulador de Tarifa** | Previsão em tempo real a partir de origem, destino e horário |

O simulador permite selecionar entre treze pontos de referência de Nova York — incluindo os três aeroportos — e retorna a tarifa estimada pelo modelo Random Forest, acompanhada da distância calculada e da classificação de aeroporto.

---

## Conclusões

### Desempenho

O **Random Forest Regressor** apresentou o melhor desempenho global, superior em R², MAE e RMSE, com consistência confirmada em todos os folds da validação cruzada. A superioridade decorre da natureza dos dados: a hipótese de linearidade é violada pela existência de dois regimes tarifários, pela natureza cíclica das variáveis temporais e pela presença de interações entre features.

O **SVR** obteve o melhor MAPE (19,58%) apesar de MAE superior ao do Random Forest. A inversão decorre da tolerância epsilon, que desconsidera erros de pequena magnitude e resulta em distribuição de erro proporcionalmente mais equilibrada entre faixas de preço.

### Overfitting

| Modelo | R² treino | R² teste | Diferença |
| :--- | ---: | ---: | ---: |
| Regressão Linear | 0,8428 | 0,8451 | −0,0023 |
| Random Forest | 0,9367 | 0,8880 | +0,0487 |
| SVR | 0,9151 | 0,8674 | +0,0477 |

A Regressão Linear não apresenta overfitting — comportamento coerente com modelo de alto viés e baixa variância. Random Forest e SVR apresentam overfitting moderado e controlado; no primeiro caso, os parâmetros `max_depth` e `min_samples_leaf` são precisamente os mecanismos que restringem o crescimento das árvores.

### Análise de resíduos

Três padrões sistemáticos foram identificados:

1. **Heterocedasticidade** — a dispersão dos resíduos amplia-se conforme o valor predito aumenta, violando a premissa de homocedasticidade do Teorema de Gauss-Markov
2. **Agrupamentos em USD 45–60** — correspondentes às corridas de tarifa fixa regulamentada do aeroporto JFK, cujo valor independe do percurso e portanto não é aprendível pelos algoritmos
3. **Assimetria direcional** — os erros de maior magnitude são predominantemente de subestimação, coerente com a assimetria à direita da variável alvo

### Recomendação

A escolha do modelo não decorre exclusivamente do R². O modelo Random Forest serializado ocupa **45,9 MB** contra **737 bytes** da Regressão Linear — razão de aproximadamente 62.000 vezes.

| Cenário de uso | Modelo recomendado |
| :--- | :--- |
| Predição em lote, precisão prioritária | Random Forest |
| Alta frequência de requisições em tempo real | Regressão Linear |
| Necessidade de justificar previsões individuais | Regressão Linear |
| Ambiente com restrição de memória | Regressão Linear |

O ganho de 4,29 pontos percentuais em R² justifica o Random Forest em aplicações analíticas, mas não necessariamente em produção de alto volume, onde a Regressão Linear entrega aproximadamente 95% do desempenho a uma fração do custo computacional.

O **SVR não é recomendado** em nenhum dos cenários avaliados: posiciona-se em segundo lugar na precisão, apresenta o maior custo de treinamento, demonstrou a menor estabilidade na validação cruzada e sua complexidade impede o aproveitamento integral da base.

### Consideração metodológica

O achado mais relevante não se refere à comparação entre algoritmos, mas à etapa que a antecede. A variável `distancia_km` responde por 87,96% da importância preditiva do melhor modelo e não constava do dataset original, cuja correlação máxima com o alvo era de 0,44.

Conclui-se que, neste problema, **a qualidade da engenharia de features determinou o desempenho em grau superior ao da escolha do algoritmo**.

---

## Tecnologias

| Biblioteca | Função |
| :--- | :--- |
| pandas | Manipulação e análise de dados tabulares |
| NumPy | Operações vetorizadas e cálculo numérico |
| scikit-learn | Implementação dos algoritmos e métricas |
| matplotlib · seaborn | Visualização estatística |
| joblib | Serialização dos modelos treinados |
| Streamlit | Interface do dashboard |

---

## Autores

**Grupo 03** — Aprendizagem de Máquina

| Integrante | Responsabilidade |
| :--- | :--- |
| Erika | — |
| Felipe | — |
| Matheus | — |
| Ruylis | — |

---

## Referências

- BREIMAN, L. Random Forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.
- DRUCKER, H. et al. Support Vector Regression Machines. *Advances in Neural Information Processing Systems 9*, MIT Press, 1997.
- JAMES, G.; WITTEN, D.; HASTIE, T.; TIBSHIRANI, R. *An Introduction to Statistical Learning*. 2. ed. Springer, 2021.
- SINNOTT, R. W. Virtues of the Haversine. *Sky and Telescope*, v. 68, n. 2, p. 159, 1984.
- PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *JMLR*, v. 12, p. 2825–2830, 2011.

