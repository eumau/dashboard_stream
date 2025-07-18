# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# ============================
# Carregar modelo e colunas
# ============================
modelo = joblib.load('modelo_vendas.pkl')
colunas_modelo = joblib.load('colunas_modelo.pkl')
df = pd.read_csv('vgsales.csv').dropna(subset=['Year', 'Platform', 'Genre', 'Global_Sales'])

st.set_page_config(page_title="Previsão de Vendas de Jogos", layout="wide")
st.title("🎮 Previsão de Vendas Globais de Jogos")

st.sidebar.header("🔍 Selecione os parâmetros")

# ============================
# Filtros
# ============================
anos = sorted(df['Year'].dropna().astype(int).unique())
plataformas = sorted(df['Platform'].dropna().unique())
generos = sorted(df['Genre'].dropna().unique())

ano = st.sidebar.selectbox("Ano de Lançamento", anos)
plataforma = st.sidebar.selectbox("Plataforma", plataformas)
genero = st.sidebar.selectbox("Gênero", generos)

if st.sidebar.button("🔮 Fazer Previsão"):
    entrada = pd.DataFrame([[ano, genero, plataforma]], columns=['Year', 'Genre', 'Platform'])
    entrada_dummies = pd.get_dummies(entrada)

    # Garantir que todas as colunas esperadas estão presentes
    for col in colunas_modelo:
        if col not in entrada_dummies.columns:
            entrada_dummies[col] = 0

    entrada_dummies = entrada_dummies[colunas_modelo]  # Reordenar as colunas
    previsao = modelo.predict(entrada_dummies)[0]

    st.subheader("📈 Resultado da Previsão")
    st.metric(label="Vendas Globais Previstas (em milhões)", value=f"{previsao:.2f} milhões")

    # Previsão futura
    st.subheader("🔮 Projeção de Vendas Futuras")
    anos_futuros = list(range(ano, ano + 6))
    previsoes_futuras = []

    for futuro in anos_futuros:
        entrada_futura = pd.DataFrame([[futuro, genero, plataforma]], columns=['Year', 'Genre', 'Platform'])
        entrada_futura = pd.get_dummies(entrada_futura)

        for col in colunas_modelo:
            if col not in entrada_futura.columns:
                entrada_futura[col] = 0

        entrada_futura = entrada_futura[colunas_modelo]
        pred = modelo.predict(entrada_futura)[0]
        previsoes_futuras.append(pred)

    df_futuro = pd.DataFrame({'Ano': anos_futuros, 'Previsão de Vendas': previsoes_futuras})
    fig = px.line(df_futuro, x='Ano', y='Previsão de Vendas', markers=True,
                  title="📊 Projeção de Vendas Globais para os Próximos Anos")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("🔧 Ajuste os filtros na barra lateral e clique no botão para fazer uma previsão.")
