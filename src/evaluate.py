import pandas as pd 
from fairlearn.metrics import MetricFrame, false_negative_rate
import sys 
from src.preprocess import load_dataset, dividir_treino_teste
import joblib

def FNR(pipeline, X_test, y_test, sensitive_features):
    """
    Calcula a False Negative Rate (FNR) para cada grupo sensível.

    Args:
        pipeline: Modelo treinado.
        X_test: Conjunto de teste das variáveis preditoras.
        y_test: Conjunto de teste da variável alvo.
        sensitive_features: Coluna do DataFrame indicando o grupo sensível (ex: sexo).

    Retorna:
        disparidade (float): Disparidade entre os grupos sensíveis.
        fnr_por_grupo (dict): FNR calculada para cada grupo sensível.
    """
    y_pred = pipeline.predict(X_test)
    sensitive_sex = sensitive_features
    
    mf = MetricFrame(
            metrics=false_negative_rate,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_sex,
        )
    fnr_por_grupo = mf.by_group
    disparidade = fnr_por_grupo.max() - fnr_por_grupo.min()

    print(f"FNR por grupo:\n{fnr_por_grupo}")
    print(f"Disparidade: {disparidade:.3f}")
    return disparidade, fnr_por_grupo
    
def verificar_disparidade(disparidade, limiar=0.10):
    """Verifica se a disparidade está dentro do limite aceitável."""
    return disparidade <= limiar

def evaluate_model(pipeline, X_test, y_test):
    LIMIAR_DISPARIDADE = 0.10
    MIN_CASOS_CONFIAVEL = 10

    sensitive_sex = X_test["sex"]
    disparidade, fnr_por_grupo = FNR(pipeline, X_test, y_test, sensitive_sex)
    # 1º: salvaguarda (informa antes de julgar)
    check = pd.DataFrame({"sex": sensitive_sex, "doente_real": y_test})
    doentes_por_grupo = check.groupby("sex")["doente_real"].sum()
    print("\nDoentes reais por grupo (base da FNR):")
    print(doentes_por_grupo, "\n")
    for grupo, n_doentes in doentes_por_grupo.items():
        if n_doentes < MIN_CASOS_CONFIAVEL:
            print(f"⚠️  Grupo sex={grupo}: só {n_doentes} doentes reais. "
                  f"FNR instável — interpretar com cautela.")

    # 2º: gate (julga por último, encerrando se reprovar)
    if verificar_disparidade(disparidade, LIMIAR_DISPARIDADE):
        print("\n✅ APROVADO: disparidade dentro do limite aceitável.")
    else:
        print("\n❌ REPROVADO: disparidade acima do limite aceitável.")
        sys.exit(1)

if __name__ == "__main__":
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = joblib.load("models/best_model.joblib")
    evaluate_model(pipeline, X_test, y_test)