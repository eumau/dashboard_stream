import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ========================
# Dados e Modelo
# ========================
df = pd.read_csv('vgsales.csv')
df = df.dropna(subset=['Year'])

genres = df['Genre'].dropna().unique()
platforms = df['Platform'].dropna().unique()
anos = sorted(df['Year'].dropna().unique())

genero_mais_vendido = df.groupby('Genre')['Global_Sales'].sum().idxmax()
plataforma_mais_vendida = df.groupby('Platform')['Global_Sales'].sum().idxmax()
ano_mais_vendas = int(df.groupby('Year')['Global_Sales'].sum().idxmax())

modelo = joblib.load('modelo_vendas.pkl')
colunas_modelo = joblib.load('colunas_modelo.pkl')

# ========================
# Sidebar de Navegação
# ========================
st.sidebar.title('Navegação')
pagina = st.sidebar.radio('Ir para:', [
    'Resumo da EDA',
    'Vendas por Gênero',
    'Vendas por Plataforma',
    'Previsão de Vendas'
])

st.title('Dashboard de Vendas de Jogos')

# ========================
# Página: Resumo EDA
# ========================
if pagina == 'Resumo da EDA':
    st.header('Resumo da Análise Exploratória')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🎮 Gênero Mais Vendido", genero_mais_vendido)
    col2.metric("🕹️ Plataforma Mais Vendida", plataforma_mais_vendida)
    col3.metric("📅 Ano com Mais Vendas", ano_mais_vendas)

    st.plotly_chart(px.bar(df.groupby('Platform').size().reset_index(name='Quantidade'),
                           x='Platform', y='Quantidade', title='Distribuição de Jogos por Plataforma'))

    st.plotly_chart(px.pie(df, names='Platform', values='Global_Sales',
                           title='Distribuição das Vendas Globais por Plataforma'))

    st.plotly_chart(px.bar(df.sort_values(by='Global_Sales', ascending=False).head(10),
                           x='Name', y='Global_Sales', color='Platform',
                           title='Top 10 Jogos Mais Vendidos no Mundo'))

    st.plotly_chart(px.bar(df.groupby('Platform')['NA_Sales'].sum().reset_index(),
                           x='Platform', y='NA_Sales', title='Vendas na América do Norte por Plataforma'))

    st.plotly_chart(px.bar(df.groupby('Platform')['EU_Sales'].sum().reset_index(),
                           x='Platform', y='EU_Sales', title='Vendas na Europa por Plataforma'))

    st.plotly_chart(px.bar(df.groupby('Platform')['JP_Sales'].sum().reset_index(),
                           x='Platform', y='JP_Sales', title='Vendas no Japão por Plataforma'))

    st.plotly_chart(px.bar(df.groupby('Platform')['Other_Sales'].sum().reset_index(),
                           x='Platform', y='Other_Sales', title='Vendas em Outras Regiões por Plataforma'))

    st.plotly_chart(px.line(df.groupby('Year')['Global_Sales'].sum().reset_index(),
                            x='Year', y='Global_Sales', title='Evolução das Vendas Globais por Ano'))

    st.plotly_chart(px.bar(df, x='Platform', y='Global_Sales', color='Platform',
                           animation_frame='Year', range_y=[0, df['Global_Sales'].max()],
                           title='Evolução Anual das Vendas Globais por Plataforma'))

# ========================
# Página: Vendas por Gênero
# ========================
elif pagina == 'Vendas por Gênero':
    st.header('Vendas por Gênero')

    genero = st.selectbox("Selecione o Gênero:", genres)
    ano = st.slider("Selecione o Ano:", int(min(anos)), int(max(anos)), int(min(anos)))

    filtro = df[(df['Genre'] == genero) & (df['Year'] == ano)]
    if filtro.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        fig = px.bar(filtro, x='Name', y='Global_Sales',
                     title=f'Vendas Globais - {genero} ({ano})')
        st.plotly_chart(fig)

# ========================
# Página: Vendas por Plataforma
# ========================
elif pagina == 'Vendas por Plataforma':
    st.header('Vendas por Plataforma')

    plataforma = st.selectbox("Selecione a Plataforma:", platforms)

    filtro = df[df['Platform'] == plataforma]
    if filtro.empty:
        st.warning("Nenhum dado encontrado para a plataforma selecionada.")
    else:
        fig = px.histogram(filtro, x='Genre', y='Global_Sales', color='Genre',
                           title=f'Vendas por Gênero na Plataforma {plataforma}')
        st.plotly_chart(fig)

# ========================
# Página: Previsão de Vendas
# ========================
elif pagina == 'Previsão de Vendas':
    st.header("Prever Vendas de um Jogo")

    col1, col2 = st.columns([1, 3])

    with col1:
        ano = st.number_input("Ano de Lançamento", value=2010, min_value=1980, max_value=2030)
        plataforma = st.selectbox("Plataforma", platforms)
        genero = st.selectbox("Gênero", genres)

        if st.button("Prever Vendas"):
            entrada = pd.DataFrame([{'Year': ano, 'Platform': plataforma, 'Genre': genero}])
            entrada_dummies = pd.get_dummies(entrada)
            for col in colunas_modelo:
                if col not in entrada_dummies.columns:
                    entrada_dummies[col] = 0
            entrada_dummies = entrada_dummies[colunas_modelo]

            pred_log = modelo.predict(entrada_dummies)[0]
            pred_real = np.expm1(pred_log)

            st.success(f'Vendas Globais Estimadas: {pred_real:.2f} milhões de unidades')

            # Previsão futura
            anos_futuros = list(range(ano, ano + 11))
            preds_futuros = []
            for a in anos_futuros:
                df_futuro = pd.DataFrame([{'Year': a, 'Platform': plataforma, 'Genre': genero}])
                df_futuro_dummies = pd.get_dummies(df_futuro)
                for col in colunas_modelo:
                    if col not in df_futuro_dummies.columns:
                        df_futuro_dummies[col] = 0
                df_futuro_dummies = df_futuro_dummies[colunas_modelo]

                pred_log_fut = modelo.predict(df_futuro_dummies)[0]
                preds_futuros.append(np.expm1(pred_log_fut))

            fig = px.line(x=anos_futuros, y=preds_futuros,
                          labels={'x': 'Ano', 'y': 'Vendas Estimadas (milhões)'},
                          title=f'Previsão de Vendas Futuras para {genero} no {plataforma}')
            col2.plotly_chart(fig)

# ========================
# Executar com: `streamlit run app_streamlit.py`
# ========================
