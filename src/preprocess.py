from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_dataset():
    """
    Função para carregar o dataset Heart Disease do repositório UCI.
    
    Retorna:
        X (DataFrame): Variáveis preditoras.
        y_binario (Series): Variável alvo.
    """
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    y_binario = (y['num'] >= 1).astype(int)
    return X, y_binario

def dividir_treino_teste(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Função para dividir o dataset em conjuntos de treino e teste.
    
    Args:
        X (DataFrame): Variáveis preditoras.
        y (Series): Variável alvo.
        test_size (float): Proporção do conjunto de teste.
        random_state (int): Semente para reprodutibilidade.
    
    Retorna:
        X_train, X_test, y_train, y_test: Conjuntos de treino e teste.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def criar_preprocessador():
    """
    Função para pré-processar os dados.

    Args:
        None
    Retorna:
        X_train, X_test, y_train, y_test: Conjuntos de treino e teste pré.-processados.
    """
    num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
        ])


    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    return preprocessor