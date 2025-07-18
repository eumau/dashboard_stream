import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ===========================
# Carregar dados e modelo
# ===========================
df = pd.read_csv("vgsales.csv")
df = df.dropna(subset=["Year", "Genre", "Platform", "Global_Sales"])

modelo = joblib.load("modelo_vendas.pkl")
colunas_modelo = joblib.load("colunas_modelo.pkl")

# ===========================
# Sidebar de navegação
# ===========================
st.sidebar.title("🎮 Navegação")
pagina = st.sidebar.radio("Ir para", ["Resumo", "Vendas por Gênero", "Vendas por Plataforma", "Previsão de Vendas"])

st.title("📊 Dashboard de Vendas de Jogos")

# ===========================
# PÁGINA: RESUMO
# ===========================
if pagina == "Resumo":
    st.subheader("📅 Vendas Globais por Ano")
    vendas_ano = df.groupby("Year")["Global_Sales"].sum().reset_index()
    fig = px.line(vendas_ano, x="Year", y="Global_Sales", markers=True, title="Vendas Globais ao Longo dos Anos")
    st.plotly_chart(fig)

    st.subheader("📚 Vendas por Gênero")
    genero = df.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(genero, x="Genre", y="Global_Sales", title="Total de Vendas por Gênero")
    st.plotly_chart(fig)

    st.subheader("🕹️ Vendas por Plataforma")
    plataforma = df.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(plataforma, x="Platform", y="Global_Sales", title="Total de Vendas por Plataforma")
    st.plotly_chart(fig)

# ===========================
# PÁGINA: VENDAS POR GÊNERO
# ===========================
elif pagina == "Vendas por Gênero":
    st.subheader("🎮 Vendas por Gênero por Ano")
    anos = sorted(df["Year"].dropna().unique())
    ano_escolhido = st.selectbox("Selecione o Ano", anos)
    dados_filtrados = df[df["Year"] == ano_escolhido]
    grafico = dados_filtrados.groupby("Genre")["Global_Sales"].sum().reset_index()
    fig = px.bar(grafico, x="Genre", y="Global_Sales", title=f"Vendas por Gênero em {int(ano_escolhido)}")
    st.plotly_chart(fig)

# ===========================
# PÁGINA: VENDAS POR PLATAFORMA
# ===========================
elif pagina == "Vendas por Plataforma":
    st.subheader("🎮 Vendas por Plataforma por Ano")
    anos = sorted(df["Year"].dropna().unique())
    ano_escolhido = st.selectbox("Selecione o Ano", anos, key="ano_plataforma")
    dados_filtrados = df[df["Year"] == ano_escolhido]
    grafico = dados_filtrados.groupby("Platform")["Global_Sales"].sum().reset_index()
    fig = px.bar(grafico, x="Platform", y="Global_Sales", title=f"Vendas por Plataforma em {int(ano_escolhido)}")
    st.plotly_chart(fig)

# ===========================
# PÁGINA: PREVISÃO
# ===========================
elif pagina == "Previsão de Vendas":
    st.subheader("📈 Prever Vendas Globais de um Jogo")

    plataformas = sorted(df["Platform"].dropna().unique())
    generos = sorted(df["Genre"].dropna().unique())
    anos = sorted(df["Year"].dropna().astype(int).unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        plataforma = st.selectbox("Plataforma", plataformas)
    with col2:
        genero = st.selectbox("Gênero", generos)
    with col3:
        ano = st.selectbox("Ano de Lançamento", anos)

    if st.button("🔮 Prever Vendas"):
        # Criar DataFrame de entrada
        entrada = pd.DataFrame([[ano, genero, plataforma]], columns=["Year", "Genre", "Platform"])
        entrada_dummy = pd.get_dummies(entrada)
        entrada_dummy = entrada_dummy.reindex(columns=colunas_modelo, fill_value=0)

        # Previsão
        previsao = modelo.predict(entrada_dummy)[0]
        st.success(f"🎯 Previsão de Vendas Globais: {previsao:.2f} milhões de unidades")

        # Exibir gráfico com valor previsto
        st.subheader("📊 Gráfico de Previsão")
        grafico_prev = pd.DataFrame({
            "Variável": ["Ano", "Plataforma", "Gênero", "Previsão"],
            "Valor": [ano, plataforma, genero, f"{previsao:.2f} milhões"]
        })
        st.write(grafico_prev)

