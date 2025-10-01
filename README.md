# MVP 1  - Sistemas de Suporte a Decisão: Predição de Doença Cardíaca

**Discente:** Helena Silveira Ventura
**Data:** 29/09/2025

---

## 1. Definição do Problema

O objetivo deste projeto é desenvolver um Mínimo Produto Viável (MVP) para um problema de classificação, utilizando técnicas de machine learning. A tarefa consiste em prever a presença de doença cardíaca em pacientes com base em suas características clínicas e demográficas, um desafio relevante que pode auxiliar em processos de diagnóstico precoce e triagem de pacientes no sistema de saúde.

Para isso, utilizaremos o dataset "Heart Disease" do repositório da UCI. O problema será modelado como uma classificação binária, onde o objetivo é distinguir pacientes com doença cardíaca daqueles sem a condição.

### Hipótese

As características clínicas e demográficas de um paciente (como idade, tipo de dor no peito, frequência cardíaca, pressão arterial, etc.) contêm informações suficientes para prever se ele apresenta doença cardíaca.

### Dataset

- **Fonte:** UCI Machine Learning Repository
- **Link:** [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Atributos:** O dataset possui 13 atributos (features) e 1 variável alvo (presença de doença cardíaca)

### Configuração do Ambiente

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Configurações de visualização para os gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Ambiente configurado e bibliotecas importadas com sucesso!")
```

---

## 2. Preparação dos Dados

Nesta etapa, realizamos as operações de carga, limpeza e transformação dos dados para prepará-los para a modelagem.

### Operações Realizadas

1. **Carga dos Dados:** O dataset é carregado diretamente do repositório UCI Machine Learning, garantindo a reprodutibilidade do notebook.

2. **Tratamento de Dados:** A variável alvo `target` é verificada e tratada para garantir uma classificação binária consistente. Pacientes com doença cardíaca são classificados como classe 1, e os sem doença como classe 0. Valores ausentes (representados como '?') são identificados e removidos.

3. **Separação em Treino e Teste:** O dataset é dividido em 80% para treino e 20% para teste. Utilizamos a amostragem estratificada para garantir que a proporção de pacientes com e sem doença seja a mesma em ambos os conjuntos, o que é crucial para avaliar adequadamente modelos em problemas médicos.

4. **Padronização (Scaling):** Criamos uma versão padronizada dos dados (média 0, desvio padrão 1). Embora modelos baseados em árvores (como o Random Forest) não necessitem desta etapa, ela é fundamental para modelos sensíveis à escala, como a Regressão Logística, que usaremos como baseline.

### Implementação

#### 2.1. Carga dos dados

```python
# Carregando dados do repositório UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Definindo nomes das colunas
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Carregando com tratamento de valores ausentes
data = pd.read_csv(url, names=column_names, na_values='?')

# Removendo linhas com valores faltantes
data = data.dropna()

print("--- Amostra do Dataset Original ---")
display(data.head())
```

#### 2.2. Tratamento da variável alvo binária

```python
# Convertendo target para binário (0 = sem doença, 1-4 = com doença)
data['target'] = (data['target'] > 0).astype(int)

# Separando as features (X) e o alvo (y)
X = data.drop('target', axis=1)
y = data['target']

print("\n--- Distribuição das Classes (0 = Sem Doença, 1 = Com Doença) ---")
print(y.value_counts(normalize=True))
sns.countplot(x=y)
plt.title('Distribuição das Classes de Doença Cardíaca')
plt.xlabel('Classe')
plt.ylabel('Frequência')
plt.xticks([0, 1], ['Sem Doença', 'Com Doença'])
plt.show()
```

#### 2.3. Separação em Treino e Teste (Estratificada)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nFormato dos dados de treino: {X_train.shape}")
print(f"Formato dos dados de teste: {X_test.shape}")
```

#### 2.4. Padronização dos Dados

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nDados padronizados com sucesso para uso em modelos sensíveis à escala.")
```

---

## 3. Modelagem e Treinamento

Nesta seção, construímos e treinamos os modelos de machine learning. A abordagem escolhida foi:

- **Modelo Baseline (Regressão Logística):** Um modelo simples e rápido que serve como um ponto de referência. Se um modelo mais complexo não superar significativamente este baseline, sua complexidade pode não ser justificada.

- **Modelo Principal (Random Forest):** Um algoritmo de ensemble robusto e de alta performance, que não exige padronização de dados e oferece insights sobre a importância das features.

- **Análise de Feature Importance:** Após treinar o Random Forest inicial, analisamos quais atributos mais influenciaram suas decisões. Isso nos ajuda a entender o "raciocínio" do modelo e a validar se ele está focando em características clínicas relevantes.

### Implementação

#### 3.1. Modelo Baseline: Regressão Logística

```python
# Usamos os dados escalados e 'class_weight' para lidar com possível desbalanceamento
print("--- Treinando Modelo Baseline: Regressão Logística ---")
log_reg = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
print("Modelo de Regressão Logística treinado.")
```

#### 3.2. Modelo Principal: Random Forest Classifier

```python
# Não precisa de dados escalados. 'class_weight' também é usado aqui
print("\n--- Treinando Modelo Principal: Random Forest ---")
rf_initial = RandomForestClassifier(
    random_state=42, class_weight='balanced', n_estimators=100
)
rf_initial.fit(X_train, y_train)
print("Modelo Random Forest Inicial treinado.")
```

#### 3.3. Análise de Feature Importance

```python
print("\n--- Análise de Importância das Features (Random Forest) ---")
importances = rf_initial.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names, 
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Importância das Features para Prever Doença Cardíaca')
plt.xlabel('Importância Relativa')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

display(feature_importance_df)
```

---

## 4. Otimização de Hiperparâmetros

Um modelo de machine learning possui "botões" (hiperparâmetros) que controlam seu comportamento. Para extrair a máxima performance do Random Forest, utilizamos a técnica GridSearchCV, que realiza uma busca exaustiva pela melhor combinação desses hiperparâmetros.

### Justificativa

- **Validação Cruzada (cv=5):** O GridSearchCV utiliza validação cruzada para avaliar cada combinação de parâmetros. Isso fornece uma estimativa muito mais robusta da performance do modelo em dados não vistos, prevenindo a escolha de parâmetros que funcionam bem por mero acaso em uma única divisão de dados.

- **Métrica de Otimização (F1-Score):** Como estamos lidando com um problema de saúde onde tanto falsos positivos quanto falsos negativos têm custos associados, escolhemos o f1_score ponderado, que representa uma média harmônica entre precisão e recall, fornecendo uma medida de performance equilibrada e confiável para aplicações médicas.

### Implementação

```python
print("--- Otimizando o Random Forest com GridSearchCV ---")

# Definindo a grade de parâmetros para testar
# Esta grade é menor para uma execução mais rápida como exemplo
# Em um projeto real, a grade pode ser mais ampla
param_grid = {
    'n_estimators': [150, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Configurando o GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Usar todos os núcleos de CPU disponíveis
    scoring='f1_weighted',
    verbose=2
)

# Executando a busca
grid_search.fit(X_train, y_train)

print(f"\nMelhores hiperparâmetros encontrados: {grid_search.best_params_}")

# O melhor modelo já treinado com os melhores parâmetros
rf_optimized = grid_search.best_estimator_
```

---

## 5. Avaliação Final e Conclusão

Esta é a etapa final, onde avaliamos a performance dos nossos modelos no conjunto de teste — dados que eles nunca viram antes. Isso nos dá a melhor estimativa de como os modelos se comportariam em um cenário real de diagnóstico.

### Análise dos Resultados

- **Comparação de Modelos:** Comparamos a performance (Acurácia e F1-Score) dos três modelos: o baseline, o Random Forest inicial e o Random Forest otimizado.

- **Análise de Overfitting:** Verificamos se o modelo final está "decorando" os dados de treino (overfitting) ou se ele generaliza bem para novos pacientes. Fazemos isso comparando sua pontuação no conjunto de treino com a do conjunto de teste.

- **Matriz de Confusão:** Visualizamos os erros e acertos do melhor modelo para entender que tipo de erros ele mais comete (ex: diagnosticar erroneamente um paciente saudável como doente, ou vice-versa).

- **Conclusão:** A melhor solução encontrada será o modelo com a maior performance no conjunto de teste, demonstrando que a abordagem de modelagem e otimização foi bem-sucedida.

### Implementação

```python
print("--- Avaliação Final dos Modelos no Conjunto de Teste ---")

# Fazendo previsões com todos os modelos
y_pred_log_reg = log_reg.predict(X_test_scaled)
y_pred_rf_initial = rf_initial.predict(X_test)
y_pred_rf_optimized = rf_optimized.predict(X_test)
```

#### 5.1. Tabela Comparativa de Resultados

```python
results = {
    "Modelo": [
        "Regressão Logística (Baseline)",
        "Random Forest Inicial",
        "Random Forest Otimizado"
    ],
    "Acurácia": [
        accuracy_score(y_test, y_pred_log_reg),
        accuracy_score(y_test, y_pred_rf_initial),
        accuracy_score(y_test, y_pred_rf_optimized)
    ],
    "F1-Score (Ponderado)": [
        float(classification_report(y_test, y_pred_log_reg, output_dict=True)['weighted avg']['f1-score']),
        float(classification_report(y_test, y_pred_rf_initial, output_dict=True)['weighted avg']['f1-score']),
        float(classification_report(y_test, y_pred_rf_optimized, output_dict=True)['weighted avg']['f1-score'])
    ]
}

results_df = pd.DataFrame(results).set_index("Modelo")
print("\n--- Tabela Comparativa de Performance ---\n")
display(results_df.round(4))
```

#### 5.2. Relatório de Classificação Detalhado do Melhor Modelo

```python
print("\n--- Relatório de Classificação - Random Forest Otimizado ---")
print(classification_report(y_test, y_pred_rf_optimized, target_names=['Sem Doença (0)', 'Com Doença (1)']))
```

#### 5.3. Análise de Overfitting do Melhor Modelo

```python
train_score = rf_optimized.score(X_train, y_train)
test_score = rf_optimized.score(X_test, y_test)

print("\n--- Análise de Overfitting (Modelo Otimizado) ---")
print(f"Pontuação (Acurácia) no conjunto de Treino: {train_score:.4f}")
print(f"Pontuação (Acurácia) no conjunto de Teste: {test_score:.4f}")

if train_score > test_score + 0.1:
    print("\nAlerta: Diferença significativa entre treino e teste. Pode haver overfitting.")
else:
    print("\nO modelo parece ter uma boa capacidade de generalização.")
```

#### 5.4. Matriz de Confusão do Melhor Modelo

```python
print("\n--- Matriz de Confusão (Modelo Otimizado) ---")
cm = confusion_matrix(y_test, y_pred_rf_optimized)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sem Doença', 'Com Doença'], 
            yticklabels=['Sem Doença', 'Com Doença'])
plt.xlabel('Previsão do Modelo')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusão')
plt.show()
```

---

## Conclusões e Considerações Finais

Este primeiro MVP da matéria de sistemas de suporte a decisão demonstrou a aplicação prática de técnicas de machine learning para resolver um problema real de saúde pública. O projeto seguiu uma metodologia estruturada, desde a definição do problema até a avaliação final dos modelos.

### Principais Resultados

1. **Preparação de Dados Eficiente:** O tratamento adequado da variável alvo e a remoção de valores ausentes garantiram um dataset limpo e confiável para a modelagem.

2. **Comparação de Modelos:** A utilização de um modelo baseline (Regressão Logística) permitiu avaliar se a complexidade adicional do Random Forest era justificada para este problema clínico.

3. **Otimização Sistemática:** O uso do GridSearchCV com validação cruzada garantiu uma seleção robusta de hiperparâmetros, maximizando a performance do modelo final.

4. **Avaliação Abrangente:** A análise incluiu múltiplas métricas (acurácia, F1-score), verificação de overfitting e visualização da matriz de confusão, proporcionando uma visão completa da performance do modelo em um contexto médico.

### Aplicações Práticas

O modelo desenvolvido pode ser aplicado em:
- Triagem inicial de pacientes em unidades de saúde
- Apoio à decisão médica em diagnósticos
- Identificação de pacientes de alto risco para intervenção precoce
- Otimização de recursos hospitalares
- Programas de prevenção de doenças cardiovasculares

### Limitações e Considerações Éticas

É fundamental ressaltar que:
- O modelo é uma **ferramenta de apoio**, não substituindo o julgamento clínico profissional
- Decisões médicas devem sempre considerar o contexto completo do paciente
- O modelo foi treinado em um dataset específico e pode não generalizar para todas as populações
- Validação externa e aprovação regulatória são necessárias antes de uso clínico

### Próximos Passos

Para trabalhos futuros, sugere-se:
- Exploração de outros algoritmos de machine learning (XGBoost, Redes Neurais)
- Análise mais detalhada das features mais importantes e sua relevância clínica
- Coleta de dados adicionais para melhorar a generalização
- Implementação de técnicas de interpretabilidade (SHAP, LIME)
- Validação externa com dados de outras instituições
- Desenvolvimento de interface amigável para uso por profissionais de saúde
- Implementação de sistema de monitoramento contínuo do modelo

---

**Desenvolvido por:** Helena Silveira Ventura 
**Data de Conclusão:** 29/09/2025
