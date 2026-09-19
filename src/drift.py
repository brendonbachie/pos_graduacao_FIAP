from scipy.stats import ks_2samp


def detectar_drift(dados_treino, dados_novos, alpha=0.05):
    estatistica, p_valor = ks_2samp(dados_treino, dados_novos)
    print(f"Estatística KS: {estatistica}, p-valor: {p_valor}")
    return p_valor < alpha

if __name__ == "__main__":
    from src.preprocess import load_dataset
    X, y = load_dataset()
    idade_treino = X["age"]

    # Cenário 1: sem drift (comparar treino com ele mesmo)
    print("Sem drift esperado:")
    print("Drift?", detectar_drift(idade_treino, idade_treino))

    # Cenário 2: com drift (envelhecer 10 anos)
    print("\nCom drift esperado:")
    print("Drift?", detectar_drift(idade_treino, idade_treino + 10))