import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.preprocess import load_dataset, dividir_treino_teste, criar_preprocessador
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DB_PATH = RAIZ / "mlflow.db"

def train(X_train, y_train):
    "Treina o pipeline completo e retorna o modelo ajustado"
    preprocessor = criar_preprocessador()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def accuracy(pipeline, X_test, y_test):
    "Calcula a acurácia do modelo no conjunto de teste"
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    mlflow.set_experiment("heart-disease")
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = train(X_train, y_train)
    acc = accuracy(pipeline, X_test, y_test)
    print(f"Acurácia do modelo no conjunto de teste: {acc:.4f}")
    scores_lr = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="recall")
    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.log_param("modelo", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("recall_cv", scores_lr.mean())
        mlflow.log_metric("accuracy_test", acc)
        mlflow.sklearn.log_model(pipeline, name="LogisticRegression", serialization_format="pickle")
        joblib.dump(pipeline, "models/best_model.joblib")
        print("Modelo salvo em models/best_model.joblib")