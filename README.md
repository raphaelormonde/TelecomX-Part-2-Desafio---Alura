# 📡 Previsão de Churn - Telecom X

Este projeto utiliza técnicas de **Data Science** e **Machine Learning** para identificar padrões de evasão de clientes (Churn) em uma empresa de telecomunicações. O objetivo é fornecer insights acionáveis para reduzir a perda de receita e melhorar a retenção.

## 📊 Visão Geral do Dataset
O dataset original contém informações de **7.043 clientes**. Após o pré-processamento (Encoding), o conjunto de dados foi expandido para **41 variáveis** preditivas, incluindo dados demográficos, serviços assinados e informações financeiras.

### Desafios Identificados:
* **Desbalanceamento de Classes**: Apenas ~26% dos clientes haviam evadido.
* **Tipos de Contrato**: Clientes com contratos mensais apresentaram a maior taxa de saída.
* **Retenção Inicial**: A maior parte da evasão ocorre nos primeiros 20 meses de contrato (baixo *tenure*).

## 🛠️ Tecnologias Utilizadas
* **Python** (Pandas, NumPy)
* **Visualização**: Matplotlib e Seaborn
* **Machine Learning**: Scikit-learn
* **Balanceamento**: SMOTE (imbalanced-learn)

## 🚀 Etapas do Projeto

1. **Tratamento de Dados**: Limpeza de valores ausentes e conversão de tipos.
2. **Feature Engineering**: Aplicação de *One-Hot Encoding* para transformar variáveis categóricas em numéricas.
3. **Análise de Correlação**: Identificação das variáveis que mais impactam o Churn.
4. **Balanceamento (SMOTE)**: Geração de dados sintéticos para equilibrar as classes de treino (7.761 amostras de treino).
5. **Padronização**: Escalonamento de variáveis numéricas para modelos baseados em distância.
6. **Modelagem**: Treinamento de **Regressão Logística** e **Random Forest**.
