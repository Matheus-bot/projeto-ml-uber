# Previsão de Tarifas de Corridas Uber

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)


Projeto da disciplina de Aprendizagem de Máquina (DSM, 5º semestre, 2026). O objetivo é prever o valor de uma corrida de Uber em Nova York a partir das coordenadas de embarque e desembarque, da data e hora e do número de passageiros, comparando três algoritmos de regressão com uma linha de base.

**Resultado:** o Random Forest Regressor foi o modelo recomendado, com R² de 0,8880 e erro médio absoluto de US$ 1,65 por corrida no conjunto de teste.

---

## Sumário

- [Problema](#problema)
- [Dados](#dados)
- [Pipeline](#pipeline)
- [Engenharia de features](#engenharia-de-features)
- [Modelagem](#modelagem)
- [Resultados](#resultados)
- [Conclusão](#conclusão)
- [Dashboard](#dashboard)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Limitações e trabalhos futuros](#limitações-e-trabalhos-futuros)
- [Equipe](#equipe)

---

## Problema

| Item | Descrição |
| --- | --- |
| Tipo | Regressão supervisionada |
| Variável alvo | `fare_amount` (tarifa em dólares) |
| Modelos comparados | DummyRegressor (linha de base), Regressão Linear Múltipla, Random Forest Regressor, Support Vector Regression (SVR) |
| Métricas | R², MAE, RMSE, MAPE |
| Validação | Divisão treino/teste 80/20 e validação cruzada com 5 partições |

---

## Dados

Base [Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset) (Kaggle): 200.000 corridas em Nova York entre 2009 e 2015, com 9 colunas.

| Coluna | Descrição | Uso |
| --- | --- | --- |
| `Unnamed: 0`, `key` | Identificadores | Descartadas |
| `fare_amount` | Tarifa em dólares | Variável alvo |
| `pickup_datetime` | Data e hora do embarque | Origem de `ano`, `mes`, `dia_semana`, `hora` |
| `pickup_longitude`, `pickup_latitude` | Coordenadas de embarque | Entrada e cálculo da distância |
| `dropoff_longitude`, `dropoff_latitude` | Coordenadas de desembarque | Entrada e cálculo da distância |
| `passenger_count` | Número de passageiros | Entrada |

### Limpeza

Foram removidos apenas valores impossíveis ou erros de registro.

| Critério | Registros removidos |
| --- | --- |
| Coordenadas fora de Nova York (longitude −74,3 a −73,7; latitude 40,5 a 41,0) | 4.419 |
| Distância menor que 100 metros | 3.096 |
| Zero passageiros | 709 |
| Tarifa acima de US$ 100 | 84 |
| Tarifa menor ou igual a zero | 22 |
| 208 passageiros | 1 |
| Valor nulo | 1 |
| **Base final** | **191.735 corridas (4,13% removido)** |

Critérios se sobrepõem; a soma da tabela é maior que o total removido.

O método IQR não foi usado para a tarifa: seu limite superior (US$ 22,25) eliminaria a maioria das corridas de aeroporto, que são legítimas. O corte em US$ 100 remove 0,042% da base.

---

## Pipeline

```
Uber_DS.csv (200.000 corridas)
        |
        v
Diagnóstico ---------> nulos, tarifas inválidas, coordenadas fora de NY, passageiros
        |
        v
Limpeza -------------> 194.831 corridas
        |
        v
Engenharia de features -> distancia_km (Haversine), eh_aeroporto, ano, mes, dia_semana, hora
        |                  remoção de distância < 100 m -> 191.735 corridas
        v
EDA -----------------> distribuição, correlação, VIF
        |
        v
Divisão 80/20 -------> treino 153.388 | teste 38.347
        |
        v
StandardScaler ------> ajustado só no treino
        |
        v
Ajuste de hiperparâmetros (GridSearchCV / validação cruzada, 20.000 corridas)
        |
        +--> Experimento 1: cada modelo com o máximo de dados viável
        +--> Experimento 2: os três modelos com as mesmas 20.000 corridas
        |
        v
Avaliação ------------> R², MAE, RMSE, MAPE, overfitting, validação cruzada, erro por faixa
        |
        v
Modelo final (Random Forest) -> app/modelo_rf.pkl -> dashboard Streamlit
```

---

## Engenharia de features

| Feature | Como foi criada | Correlação com a tarifa |
| --- | --- | --- |
| `distancia_km` | Fórmula de Haversine entre embarque e desembarque | 0,90 |
| `eh_aeroporto` | 1 se embarque ou desembarque a menos de 2 km do JFK, LaGuardia ou Newark | 0,68 |
| `ano` | Extraído de `pickup_datetime` | 0,13 |
| `mes` | Extraído de `pickup_datetime` | 0,03 |
| `dia_semana` | Extraído de `pickup_datetime` | 0,00 |
| `hora` | Extraído de `pickup_datetime` | −0,02 |

A maior correlação entre as variáveis originais era 0,44 (`pickup_longitude`). A distância criada passou a responder por 87,96% da importância do modelo final.

**Corridas de aeroporto** (10.547, 5,5% da base):

| | Aeroporto | Demais |
| --- | --- | --- |
| Tarifa média | US$ 37,53 | US$ 9,75 |
| Distância média | 13,13 km | 2,79 km |
| Custo por km | US$ 2,86 | US$ 3,49 |

**Multicolinearidade:** VIF máximo de 2,46 (`eh_aeroporto`), abaixo do limite usual de 5.

---

## Modelagem

### Desenho experimental

O SVR com a base completa não concluiu o treino em 30 minutos (custo entre O(n²) e O(n³)). Por isso foram conduzidos dois experimentos:

| | Experimento 1 | Experimento 2 |
| --- | --- | --- |
| Objetivo | Desempenho prático de cada modelo | Comparação em condições iguais |
| Treino da LR e do RF | 153.388 corridas | 20.000 corridas |
| Treino do SVR | 20.000 corridas | 20.000 corridas |
| Teste | 38.347 corridas | 38.347 corridas |

As 20.000 corridas são as primeiras linhas do conjunto de treino, já embaralhado por `train_test_split(random_state=42)`.

### Ajuste de hiperparâmetros

**Random Forest** — busca em grade, validação cruzada com 5 partições:

| `max_depth` | `min_samples_leaf` | R² médio |
| --- | --- | --- |
| None | 5 | 0,8672 |
| **20** | **5** | **0,8672 (escolhido)** |
| 20 | 1 | 0,8662 |
| None | 1 | 0,8661 |
| 10 | 5 | 0,8650 |

**SVR** — comparação de kernels (C = 100, epsilon = 0,1):

| Kernel | R² teste | MAE | Tempo de treino |
| --- | --- | --- | --- |
| Linear | 0,8404 | 1,98 | 648 s |
| **RBF** | **0,8674** | **1,71** | **205 s** |

**SVR** — ajuste de C e gamma (kernel RBF, validação cruzada com 5 partições):

| C | gamma | R² médio |
| --- | --- | --- |
| **100** | **0,01** | **0,8664 (escolhido)** |
| 1000 | 0,01 | 0,8658 |
| 10 | 0,01 | 0,8628 |
| 10 | scale | 0,8625 |
| 100 | scale | 0,8578 |
| 10 | 0,5 | 0,7498 |

### Configuração final

| Modelo | Hiperparâmetros |
| --- | --- |
| DummyRegressor | `strategy='mean'` |
| Regressão Linear | padrão |
| Random Forest | `n_estimators=100`, `max_depth=20`, `min_samples_leaf=5`, `random_state=42` |
| SVR | `kernel='rbf'`, `C=100`, `gamma=0.01`, `epsilon=0.1` |

---

## Resultados

### Experimento 1 — máximo de dados viável

| Modelo | Treino (n) | R² | MAE (US$) | RMSE (US$) | MAPE | Treino (s) | Previsão (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy | 153.388 | 0,0000 | 5,89 | 9,20 | 66,50% | — | — |
| Regressão Linear | 153.388 | 0,8451 | 2,05 | 3,62 | 24,76% | 0,13 | instantâneo |
| **Random Forest** | 153.388 | **0,8880** | **1,65** | **3,08** | 20,78% | 72,04 | 0,40 |
| SVR | 20.000 | 0,8728 | 1,72 | 3,28 | **19,93%** | 36,23 | 57,26 |

### Experimento 2 — mesmas 20.000 corridas

| Modelo | R² | MAE (US$) | RMSE (US$) | MAPE | Treino (s) |
| --- | --- | --- | --- | --- | --- |
| Dummy | −0,0001 | 5,92 | 9,20 | 67,39% | — |
| Regressão Linear | 0,8449 | 2,06 | 3,62 | 24,77% | 0,02 |
| **Random Forest** | **0,8737** | 1,79 | **3,27** | 22,36% | 5,46 |
| SVR | 0,8728 | **1,72** | 3,28 | **19,93%** | 36,23 |

### Overfitting (R² treino − R² teste)

| Modelo | Experimento 1 | Experimento 2 |
| --- | --- | --- |
| Regressão Linear | −0,0023 | −0,0034 |
| Random Forest | +0,0487 | +0,0551 |
| SVR | +0,0021 | +0,0021 |

### Validação cruzada (R² médio ± desvio-padrão, 5 partições)

| Modelo | Experimento 1 | Experimento 2 |
| --- | --- | --- |
| Regressão Linear | 0,8426 ± 0,0078 | 0,8402 ± 0,0226 |
| Random Forest | **0,8882 ± 0,0082** | **0,8672 ± 0,0188** |
| SVR | 0,8664 ± 0,0193 | 0,8664 ± 0,0193 |

### Erro por faixa de preço (MAE em US$, Experimento 1)

| Faixa da tarifa real | Corridas | Regressão Linear | Random Forest | SVR |
| --- | --- | --- | --- | --- |
| até US$ 10 | 23.934 | 1,35 | 1,11 | **1,01** |
| US$ 10 a 20 | 10.450 | 2,30 | **1,92** | 2,16 |
| US$ 20 a 30 | 2.067 | 4,52 | **3,55** | 4,19 |
| US$ 30 a 40 | 910 | 5,24 | **4,06** | 4,41 |
| US$ 40 a 60 | 912 | 7,27 | **5,09** | 5,99 |
| acima de US$ 60 | 74 | 23,53 | **12,78** | 15,02 |

O erro cresce com o valor da corrida (heterocedasticidade). O SVR tem o menor MAPE porque erra menos nas corridas baratas, que pesam mais numa métrica percentual; o Random Forest erra menos em todas as demais faixas.

### Variáveis mais relevantes

| Random Forest (importância) | | Regressão Linear (coeficiente padronizado) | |
| --- | --- | --- | --- |
| `distancia_km` | 0,8796 | `distancia_km` | 7,5394 |
| `dropoff_longitude` | 0,0306 | `eh_aeroporto` | 1,6679 |
| `ano` | 0,0242 | `ano` | 0,9758 |
| `eh_aeroporto` | 0,0156 | `dropoff_longitude` | −0,4646 |
| `dropoff_latitude` | 0,0144 | `dropoff_latitude` | −0,3046 |

O coeficiente da distância equivale a cerca de US$ 2,12 por quilômetro. Intercepto: US$ 11,28.

---

## Conclusão

**Modelo recomendado: Random Forest Regressor.**

| Critério | Regressão Linear | Random Forest | SVR |
| --- | --- | --- | --- |
| Desempenho (R², Exp. 1) | 0,8451 | **0,8880** | 0,8728 |
| Erro médio por corrida | US$ 2,05 | **US$ 1,65** | US$ 1,72 |
| Escala para a base completa | Sim | Sim | Não |
| Tempo de previsão (38.347 corridas) | instantâneo | 0,40 s | 57,26 s |
| Tamanho do modelo | 737 bytes | 45,9 MB | — |
| Interpretabilidade | Alta | Média (importância das variáveis) | Baixa |

- O Random Forest tem o maior R² e o menor RMSE nos dois experimentos e a maior média na validação cruzada. Erra 20% menos que a Regressão Linear.
- A Regressão Linear é a alternativa quando é obrigatório justificar cada previsão ou quando o hardware é muito limitado.
- O SVR empata com o Random Forest em condições iguais, mas não escala para a base completa e é o mais lento para prever.
- A engenharia de features pesou mais que a escolha do algoritmo: a distância, ausente do dataset original, responde por 88% da importância do modelo final.

---

## Dashboard

Aplicação em Streamlit com o Random Forest treinado na base completa.

| Aba | Conteúdo |
| --- | --- |
| Visão Geral | Indicadores da base e distribuição das tarifas |
| Análise Exploratória | Matriz de correlação, correlação com a tarifa, distância × tarifa |
| Ajuste dos Modelos | Busca em grade do Random Forest, comparação de kernels, ajuste de C e gamma |
| Comparação de Modelos | Métricas dos dois experimentos, overfitting, tempos, validação cruzada, erro por faixa, gráficos |
| Simulador de Tarifa | Previsão para uma corrida a partir de locais conhecidos ou coordenadas |

---

## Estrutura do repositório

```
projeto-ml-uber/
├── app/
│   ├── modelo_rf.pkl          Random Forest final (gerado pelo notebook)
│   ├── modelo_lr.pkl          Regressão Linear (gerado pelo notebook)
│   └── scaler.pkl             StandardScaler ajustado no treino
├── data/
│   ├── Uber_DS.csv            base original
│   ├── uber_tratado.csv       base limpa com as features
│   ├── metricas_exp1.csv      métricas do Experimento 1
│   ├── metricas_exp2.csv      métricas do Experimento 2
│   ├── validacao_cruzada.csv
│   ├── ajuste_rf.csv
│   ├── ajuste_svr.csv
│   ├── comparacao_kernels.csv
│   ├── erro_por_faixa.csv
│   ├── coeficientes_lr.csv
│   ├── importancias_rf.csv
│   └── *.png                  gráficos
├── notebooks/
│   └── projeto_uber.ipynb     pipeline completo
├── streamlit/
│   └── app.py                 dashboard
├── requirements.txt
└── README.md
```

---

## Como executar

Requer Python 3.11 ou mais recente.

```bash
# 1. Ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Dependências
pip install -r requirements.txt

# 3. Notebook (gera modelos, tabelas e gráficos; cerca de 1 hora por causa do SVR)
#    Abrir notebooks/projeto_uber.ipynb e executar todas as células

# 4. Dashboard (a partir da raiz do projeto)
streamlit run streamlit/app.py
```

O arquivo `data/Uber_DS.csv` deve estar presente antes de executar o notebook. O dashboard depende dos arquivos gerados pelo notebook em `data/` e `app/`.

Os arquivos `.pkl` só podem ser carregados com a mesma versão do scikit-learn que os gerou.

---

## Limitações e trabalhos futuros

| Limitação | Trabalho futuro |
| --- | --- |
| Distância em linha reta, não pelo percurso | Distância e tempo pela rota real (API de mapas) |
| Sem pedágio, trânsito ou tarifa dinâmica | Incluir essas variáveis, quando disponíveis no momento do pedido |
| Tarifa fixa do JFK não modelada explicitamente | Variável específica para a regra e para bairros de origem e destino |
| Heterocedasticidade dos erros | Transformação logarítmica da tarifa |
| SVR e ajuste feitos em amostra de 20.000 | Testar Gradient Boosting (XGBoost, LightGBM) na base completa |

---

## Tecnologias

Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, joblib, Streamlit.

---

## Equipe

- Erika A. Creatto
- Felipe F. de França
- Matheus H. de C. Rumão
- Reinaldo Gomes
- Ruylis B. Catelan