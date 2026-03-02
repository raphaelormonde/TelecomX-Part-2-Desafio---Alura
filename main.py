import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# ==============================================================================
# EXTRAÇÃO DO CSV
# ==============================================================================


df_ml = pd.read_csv('TelecomX.csv')


# ==============================================================================
# EXCLUSÃO DE COLUNAS
# ==============================================================================


# Verificando
print("\nColunas:", df_ml.columns.tolist())

#Não tem o que remover


# ==============================================================================
# ENCONDING
# ==============================================================================


# Separando variáveis binárias (Yes/No) para mapeamento direto
colunas_binarias = [
    'Churn', 'customer.Partner', 'customer.Dependents', 'phone.PhoneService', 
    'account.PaperlessBilling'
]

# Mapeando Yes para 1 e No para 0
for col in colunas_binarias:
    df_ml[col] = df_ml[col].map({'Yes': 1, 'No': 0})

# Mapeando o Gênero separadamente
df_ml['customer.gender'] = df_ml['customer.gender'].map({'Female': 1, 'Male': 0})

# Aplicando One-Hot Encoding nas colunas com múltiplas categorias
colunas_multi = [
    'phone.MultipleLines', 'internet.InternetService', 'internet.OnlineSecurity',
    'internet.OnlineBackup', 'internet.DeviceProtection', 'internet.TechSupport',
    'internet.StreamingTV', 'internet.StreamingMovies', 'account.Contract', 
    'account.PaymentMethod'
]

df_ml = pd.get_dummies(df_ml, columns=colunas_multi)

# Visualizando o novo formato dos dados
print(f"Novas dimensoes do dataset: {df_ml.shape}")
df_ml.head()

df_ml.info()


# ==============================================================================
# VERIFICAÇÃO DA PROPORÇÃO DE EVASÃO
# ==============================================================================


# Calculando a contagem e a proporção
contagem_churn = df_ml['Churn'].value_counts()
proporcao_churn = df_ml['Churn'].value_counts(normalize=True) * 100

print("Contagem por classe:")
print(contagem_churn)
print("\nProporção por classe (%):")
print(proporcao_churn)

# Visualização Gráfica
plt.figure(figsize=(8, 5))
sns.barplot(x=contagem_churn.index, y=contagem_churn.values, palette='viridis')
plt.title('Distribuição da Variável Alvo (Churn)')
plt.xlabel('Churn (0 = Não, 1 = Sim)')
plt.ylabel('Quantidade de Clientes')
plt.show()


# ==============================================================================
# SMOTE
# ==============================================================================

from imblearn.over_sampling import SMOTE

# Separando os dados
X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

# Criando o balanceador
smt = SMOTE(random_state=42)

# Aplicando o SMOTE para gerar os dados sintéticos
X_resampled, y_resampled = smt.fit_resample(X, y)

# Verificando o resultado do balanceamento
print("Antes do SMOTE (Original):")
print(y.value_counts())

print("\nDepois do SMOTE (Balanceado):")
print(y_resampled.value_counts())


# ==============================================================================
# NORMALIZAÇÃO
# ==============================================================================


from sklearn.preprocessing import StandardScaler

# Criando o escalonador
scaler = StandardScaler()

# Ajustando e transformando as colunas numéricas (Tenure, Monthly e Total)
# Dica: Normalmente escalonamos as colunas que não são 0 e 1 (binárias)
colunas_para_escalar = ['customer.tenure', 'account.Charges.Monthly', 'account.Charges.Total']

# Aplicando nos dados balanceados (X_resampled)
X_resampled[colunas_para_escalar] = scaler.fit_transform(X_resampled[colunas_para_escalar])

print("Dados normalizados com sucesso!")
X_resampled.head()


# ==============================================================================
# ANÁLISE DE CORRELAÇÃO
# ==============================================================================


# 1. Calculando a correlação de todas as variáveis com o Churn
# Usamos os dados balanceados (X_resampled e y_resampled unidos)
df_correlacao = X_resampled.copy()
df_correlacao['Churn'] = y_resampled

# 2. Criando a série de correlação apenas para o alvo
correlacao_churn = df_correlacao.corr()['Churn'].sort_values(ascending=False)

# 3. Visualizando as top correlações (positivas e negativas)
plt.figure(figsize=(10, 8))
correlacao_churn.drop('Churn').plot(kind='barh', color='skyblue')
plt.title('Correlação das Variáveis com o Churn')
plt.xlabel('Coeficiente de Correlação')
plt.ylabel('Variáveis')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# ==============================================================================
# ANÁLISES DIRECIONADAS
# ==============================================================================


# Configurando o layout para dois gráficos lado a lado
plt.figure(figsize=(16, 6))

# 1. Investigação: Tempo de contrato (Tenure) × Evasão
# O boxplot ajuda a ver onde se concentra a maioria dos clientes que saem
plt.subplot(1, 2, 1)
sns.boxplot(x='Churn', y='customer.tenure', data=df_ml, palette='coolwarm')
plt.title('Tempo de Contrato (Tenure) vs Evasão', fontsize=14)
plt.xlabel('Churn (0 = Ficou, 1 = Saiu)', fontsize=12)
plt.ylabel('Meses de Contrato', fontsize=12)

# 2. Investigação: Total gasto × Evasão
# Como o Total Gasto tem muitos valores, o boxplot revela outliers e a mediana
plt.subplot(1, 2, 2)
sns.boxplot(x='Churn', y='account.Charges.Total', data=df_ml, palette='viridis')
plt.title('Total Gasto vs Evasão', fontsize=14)
plt.xlabel('Churn (0 = Ficou, 1 = Saiu)', fontsize=12)
plt.ylabel('Total Gasto ($)', fontsize=12)

plt.tight_layout()
plt.show()


# ==============================================================================
# SEPARAÇÃO DE DADOS
# ==============================================================================


from sklearn.model_selection import train_test_split

# 1. Dividindo os dados
# test_size=0.25 define que 25% dos dados serão guardados para o teste final
# random_state=42 garante que a divisão seja sempre a mesma ao rodar o código
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=42
)

# 2. Verificando o tamanho das fatias
print(f"Total de amostras para Treino: {len(X_train)}")
print(f"Total de amostras para Teste: {len(X_test)}")


# ==============================================================================
# CRIAÇÃO DE MODELOS
# ==============================================================================


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# MODELO 1: Regressão Logística
model_log = LogisticRegression(random_state=42, max_iter=1000)
model_log.fit(X_train, y_train)
# É o modelo clássico para problemas de classificação binária (Sim/Não). Ele calcula a probabilidade de um evento ocorrer.

# MODELO 2: Random Forest
model_rf = RandomForestClassifier(random_state=42, n_estimators=100)
model_rf.fit(X_train, y_train)
# É um dos modelos mais robustos e populares. Ele cria uma "floresta" de árvores de decisão e combina os resultados para dar o veredito final, o que reduz erros e evita que o modelo "decore" os dados (overfitting).

print("Modelos treinados com sucesso!")


# ==============================================================================
# ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS
# ==============================================================================


# Variáveis mais relevantes para a Regressão Logística (Coeficientes)
# Coeficientes positivos indicam que a variável aumenta a chance de Churn
coeficientes = pd.Series(model_log.coef_[0], index=X_train.columns).sort_values(ascending=False)

# Variáveis mais relevantes para o Random Forest (Feature Importance)
# Importância baseada na redução da impureza (Gini)
importancias_rf = pd.Series(model_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# --- Visualização ---
plt.figure(figsize=(18, 8))

# Gráfico Regressão Logística
plt.subplot(1, 2, 1)
coeficientes.head(10).plot(kind='barh', color='salmon')
plt.title('Top 10 Variáveis - Regressão Logística (Coeficientes)')

# Gráfico Random Forest
plt.subplot(1, 2, 2)
importancias_rf.head(10).plot(kind='barh', color='seagreen')
plt.title('Top 10 Variáveis - Random Forest (Importância)')

plt.tight_layout()
plt.show()