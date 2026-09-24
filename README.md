# Tech Challenge - Ciclo Completo de ML

Pipeline completo de Machine Learning para triagem preliminar de doença cardíaca, cobrindo pré-processamento, treino, avaliação, auditoria de fairness e rastreamento de experimentos com MLflow.

## Objetivo

Treinar e avaliar modelos de classificação binária para prever a presença de doença cardíaca a partir do dataset **Heart Disease** da UCI (obtido via `ucimlrepo`, id 45), comparando algoritmos, medindo métricas de desempenho e auditando o modelo final quanto a viés (fairness) antes de disponibilizá-lo.

## Pipeline

1. **Carregamento e split** (`src/preprocess.py::load_dataset`, `dividir_treino_teste`): baixa o dataset da UCI, binariza o alvo (`num >= 1`) e separa treino/teste (80/20, estratificado, `random_state=42`).
2. **Pré-processamento** (`src/preprocess.py::criar_preprocessador`): `ColumnTransformer` com um pipeline numérico (`SimpleImputer` mediana + `StandardScaler`) para `age, trestbps, chol, thalach, oldpeak` e um pipeline categórico (`SimpleImputer` mais frequente + `OneHotEncoder`) para `sex, cp, fbs, restecg, exang, slope, ca, thal`.
3. **Treino** (`src/train.py`): ajusta o pipeline completo (pré-processador + `LogisticRegression`) e calcula acurácia no teste e recall via validação cruzada (5 folds); registra tudo no MLflow e salva o modelo em `models/best_model.joblib`.
4. **Avaliação de fairness** (`src/evaluate.py`): calcula a *False Negative Rate* (FNR) por grupo do atributo sensível `sex` com `fairlearn`, verifica se a disparidade entre grupos está dentro do limiar aceitável e falha o processo (`sys.exit(1)`) caso o limiar seja ultrapassado.

No notebook `notebooks/01_EDA.ipynb` foram testados três modelos — **Logistic Regression**, **Random Forest** e **MLP** — antes de consolidar a versão final nos scripts de `src/`.

## Resultados

- Os três modelos (Logistic Regression, Random Forest e MLP) **empataram em recall** na validação cruzada (~0.77). A **Logistic Regression** foi escolhida por ser mais simples e mais estável que o MLP.
- Métricas do modelo final (Logistic Regression), conforme `models/model_card.md`:
  - Acurácia global: **0.869**
  - Classe "doente" (1): precision **0.812**, recall **0.93**, f1-score **0.867** (support 28)
  - Nota do próprio Model Card: a validação cruzada indica ~0.77, enquanto o 0.93 obtido no conjunto de teste é considerado um recorte favorável (não representativo da performance esperada em produção).
- Experimentos rastreados com **MLflow** (`mlflow.log_param`, `mlflow.log_metric`, `mlflow.sklearn.log_model`), incluindo parâmetros do modelo e as métricas de recall (CV) e acurácia (teste).

## Fairness e Model Card

O Model Card completo está em [`models/model_card.md`](models/model_card.md). Principais pontos da auditoria de fairness (atributo sensível: `sex`):

- FNR por grupo: `sex=0` → 0.14, `sex=1` → 0.04
- Disparidade observada: **0.095**, dentro do limiar aceitável de **0.10** (`verificar_disparidade` em `src/evaluate.py`)
- Limitação conhecida: a FNR mais alta no grupo feminino está baseada em apenas 7 casos reais de doença nesse grupo — o próprio Model Card trata isso como um indício a reavaliar, não uma disparidade confirmada
- Necessidade apontada de testes em populações maiores/mais diversas antes de qualquer uso real

## Estrutura

```
.
├── notebooks/
│   └── 01_EDA.ipynb        # Análise exploratória e comparação dos 3 modelos
├── src/
│   ├── preprocess.py        # Carga de dados, split e pré-processador
│   ├── train.py              # Treino do pipeline final + tracking MLflow
│   └── evaluate.py           # Auditoria de fairness (FNR por sexo)
├── tests/
│   └── test_preprocess.py    # Testes unitários (pytest)
├── models/
│   └── model_card.md         # Model Card do modelo em produção
├── .github/workflows/
│   └── ml_pipeline.yml       # CI: instala dependências e roda os testes
├── requirements.txt
└── pytest.ini
```

## Como rodar

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar o modelo (gera models/best_model.joblib e registra no MLflow)
python -m src.train

# Rodar a auditoria de fairness sobre o modelo salvo
python -m src.evaluate

# Subir a UI do MLflow para visualizar os experimentos
mlflow ui --backend-store-uri sqlite:////home/casnav/tech-challenge/mlflow.db

# Rodar os testes
pytest
```

## CI

O workflow [`.github/workflows/ml_pipeline.yml`](.github/workflows/ml_pipeline.yml) roda no GitHub Actions a cada push/pull request para `main`: configura Python 3.12, instala as dependências de `requirements.txt` e executa `pytest tests -v`.

## Licença

MIT, ver [LICENSE](LICENSE).
