from fastapi import FastAPI
import joblib
from pathlib import Path
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import json

MODEL_VERSION = "1.0.0"

"""Seta a pasta raiz do projeto para que o FastAPI consiga localizar o modelo treinado e os módulos auxiliares."""
RAIZ = Path(__file__).resolve().parent.parent

"""Inicia a API """
app = FastAPI(title="API para identificar a saúde do programa")

"""Teste para verificar se a API está funcionando normalmente"""
@app.get("/health")
def health():
    return {"status": "ok"}

"""Carrega o moedelo em uma variável, pra que não precisamos carregar toda vez que a API é chamada"""
modelo = joblib.load(RAIZ / "models" / "best_model.joblib")

class Paciente(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(paciente: Paciente):
    dados = pd.DataFrame([paciente.model_dump()])
    predicao = modelo.predict(dados)[0]
    probabilidade = modelo.predict_proba(dados)[0][1]
    log = {
        "timestamp": datetime.now().isoformat(),
        "input_features": paciente.model_dump(),
        "prediction": int(predicao),
        "model_version": MODEL_VERSION,
    }
    print(json.dumps(log))
    return {
        "predicao": int(predicao),
        "probabilidade_doenca": float(probabilidade)
    }
