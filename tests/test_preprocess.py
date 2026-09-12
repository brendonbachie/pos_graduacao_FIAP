import numpy as np
import pandas as pd
from src.preprocess import load_dataset, criar_preprocessador
from src.evaluate import verificar_disparidade


def test_preprocessor_lida_com_nulos():
    # 1. Pega dados reais (aproveita a estrutura das 13 colunas)
    X, y = load_dataset()
    amostra = X.head(5).copy()

    # 2. Injeta um NaN de propósito numa coluna numérica
    amostra.loc[amostra.index[0], "chol"] = np.nan

    # 3. Passa pelo preprocessor
    preprocessor = criar_preprocessador()
    resultado = preprocessor.fit_transform(amostra)

    # 4. Afirma que NÃO sobrou nenhum NaN na saída
    assert not np.isnan(resultado).any(), "O preprocessor deixou passar NaN!"

def test_verificar_disparidade():
    # Testa a função verificar_disparidade com diferentes valores de disparidade
    assert verificar_disparidade(0.05) == True, "Disparidade 0.05 deveria ser aceitável"
    assert verificar_disparidade(0.10) == True, "Disparidade 0.10 deveria ser aceitável"
    assert verificar_disparidade(0.15) == False, "Disparidade 0.15 não deveria ser aceitável"