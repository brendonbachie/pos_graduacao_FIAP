
# Model Card — Detecção de Doença Cardíaca (FIAP HealthCare Plus)

## Detalhes do Modelo
- Algoritmo: Logistic Regression. Empatou em recall (~0.77 na CV) com MLP e Random Forest. Escolhida por parcimônia e interpretabilidade — importante em saúde, onde a decisão precisa ser explicável.
- Versão / data: v1.0 - 12/09/2026
- Dataset de treino: Usado o dataset de Hearth Disease da UCI, como 303 pacientes

## Uso Pretendido
- Para que serve: Triagem médica preliminar para doenças cardíacas
- Para que NÃO serve / fora de escopo: Qualquer outro tipo de doença e para uso pediátrico

## Métricas de Performance
- Global: Acurácia: 0.869, Classe 1 (doente): precision: 0.812 recall: 0.93 f1-score 0.867 support: 28 
- Nota honesta sobre CV vs teste: CV dá ~0.77, o 0.93 do teste é recorte favorável

## Análise de Fairness
- Atributo sensível auditado: "sex"
- FNR por grupo: 0: 0.14, 1: 0,04
- Disparidade e limiar: Disparidade de 0,095 e aceitável 0,1

## Limitações Conhecidas
- A auditoria indicou FNR mais alta no grupo feminino (0.143 vs 0.048), mas baseada em apenas 7 doentes reais — é um indício a reavaliar, não uma disparidade confirmada.; 
- Tem-se necessidade de testes em mais populações
