import requests
import json
from pathlib import Path
from src.preprocess import load_dataset
from src.drift import detectar_drift

RAIZ = Path(__file__).resolve().parent.parent
LOG_PATH = RAIZ / "logs.jsonl"
API_URL = "http://localhost:8000/predict"

LOG_PATH.unlink(missing_ok=True)

X, y = load_dataset()
amostra = X.head(50).copy()
amostra["age"] = amostra["age"] + 10  # Introduzindo drift na idade

for _, linha in amostra.iterrows():
    paciente = linha.to_dict()
    resposta = requests.post(API_URL, json=paciente)
print(f"Pacientes enviados pra API: {len(amostra)}")

# Lê as idades que foram logadas
idades_logadas = []
with open(LOG_PATH) as f:
    for linha in f:
        registro = json.loads(linha)
        idades_logadas.append(registro["input_features"]["age"])

# Compara com as idades de treino (referência)
idade_treino = X["age"]
print("\nVerificando drift na feature 'age':")
houve_drift = detectar_drift(idade_treino, idades_logadas)
if houve_drift:
    print("🚨 ALERTA: drift detectado! Recomenda-se reavaliar/retreinar o modelo.")
else:
    print("✅ Sem drift significativo.")