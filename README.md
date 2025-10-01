# 🫀 MVP - Predição de Doença Cardíaca

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU_USUARIO/PredicaoDoencaCardiaca/blob/main/MVP_Heart_Disease.ipynb)

## 📋 Sobre o Projeto

Este projeto foi desenvolvido como MVP (Minimum Viable Product) para a disciplina de **Engenharia de Produção** da **Universidade de Brasília**.

### 🎯 Objetivo

Desenvolver modelos de Machine Learning para predizer a presença de doença cardíaca em pacientes com base em características clínicas.

## 📊 Dataset

**Fonte**: UCI Machine Learning Repository - Heart Disease Dataset
- **Amostras**: ~303 pacientes
- **Features**: 13 atributos clínicos
- **Target**: Presença (1) ou ausência (0) de doença cardíaca

### Atributos

| Atributo | Descrição |
|----------|-----------|
| age | Idade do paciente |
| sex | Sexo (1=M, 0=F) |
| cp | Tipo de dor no peito |
| trestbps | Pressão arterial em repouso |
| chol | Colesterol sérico |
| fbs | Glicemia em jejum |
| restecg | Resultados ECG |
| thalach | Frequência cardíaca máxima |
| exang | Angina induzida por exercício |
| oldpeak | Depressão ST |
| slope | Inclinação do segmento ST |
| ca | Número de vasos principais |
| thal | Talassemia |

## 🤖 Modelos Utilizados

1. **Regressão Logística** (Baseline)
2. **Random Forest Classifier** (Inicial)
3. **Random Forest Otimizado** (GridSearchCV)

## 📈 Resultados

| Modelo | Acurácia | F1-Score |
|--------|----------|----------|
| Regressão Logística | ~85% | ~85% |
| Random Forest Inicial | ~84% | ~83% |
| Random Forest Otimizado | ~87% | ~87% |

## 🚀 Como Executar

### Opção 1: Google Colab (Recomendado)
1. Clique no badge "Open in Colab" acima
2. Execute todas as células (Runtime > Run all)
3. Pronto!

### Opção 2: Local
```bash
git clone https://github.com/SEU_USUARIO/PredicaoDoencaCardiaca.git
cd PredicaoDoencaCardiaca
jupyter notebook MVP_Heart_Disease.ipynb
