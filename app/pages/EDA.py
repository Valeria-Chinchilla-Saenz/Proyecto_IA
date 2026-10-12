import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

st.set_page_config(page_title="Análisis de Demanda", page_icon="🚲", layout="wide")

st.title("🚲 Análisis Exploratorio de Datos (EDA)")
st.markdown("""
Esta página se presentarán los patrones históricos de demanda de alquiler de bicicletas, 
en el cual se analizarán la influencia del calendario y las condiciones climáticas.
""")

@st.cache_data
def load_data():


    df = pd.read_csv("data/raw/hour.csv")
    
   
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df
try:
    df = load_data()
    
    st.write("### Vista Previa de los Datos")
    st.dataframe(df.head())

    
    st.header("1. Patrones de Calendario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Promedio de Alquileres por Hora")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x='hr', y='cnt', estimator=np.mean, errorbar=None, ax=ax1)
        ax1.set_title("Promedio de Alquileres de Bicicletas por Hora del Día")
        ax1.set_xlabel("Hora del Día (0-23)")
        ax1.set_ylabel("Promedio de Alquileres (cnt)")
        st.pyplot(fig1)

    with col2:
        st.subheader("Promedio de Alquileres por Mes")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x='mnth', y='cnt', estimator=np.mean, errorbar=None, ax=ax2)
        ax2.set_title("Promedio de Alquileres por mes")
        ax2.set_xlabel("Meses Ene-Dic (mnth)")
        ax2.set_ylabel("Promedio de Alquileres (cnt)")
        st.pyplot(fig2)

   
    st.header("2. Factores Climáticos y Temporada")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Alquileres por Temporada")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='season', y='cnt', hue='season', palette='Set2', data=df, ax=ax3)
        ax3.set_title("Alquileres de Bicicletas por temporada")
        ax3.set_xlabel("TEMPORADAS 1-4 (season)")
        ax3.set_ylabel("Cantidad de alquileres (cnt)")
        st.pyplot(fig3)

    with col4:
   
        st.subheader("Influencia de la Situación Climática")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=df, 
            x='weathersit', 
            y='cnt', 
            hue='weathersit', 
            palette='viridis', 
            estimator=np.mean, 
            errorbar=None, 
            ax=ax4
        )
        
        ax4.set_title("Promedio de Alquileres según Situación Climática")
        ax4.set_xlabel("Clima (1:Despejado -> 4:Tormenta)")
        ax4.set_ylabel("Promedio de Alquileres (cnt)")
        
        handles, _ = ax4.get_legend_handles_labels()
        
        etiquetas_clima = [
            '1: Despejado', 
            '2: Nublado/Neblina', 
            '3: Lluvia ligera/Nieve', 
            '4: Tormenta fuerte'
        ]
        
        ax4.legend(handles=handles, labels=etiquetas_clima, title="Tipos de Clima", loc='upper right')
        
        st.pyplot(fig4)

   
    st.header("3. Variables Numéricas Continuas")
    
    st.subheader("Relación entre Temperatura y Demanda")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='temp', y='cnt', alpha=0.3, color='orange', ax=ax5)
    ax5.set_title("Relación entre la Temperatura y la Demanda de Bicicletas")
    ax5.set_xlabel("Temperatura Normalizada (0 a 1)")
    ax5.set_ylabel("Cantidad de Alquileres (cnt)")
    st.pyplot(fig5)

    st.subheader("Matriz de Correlación")
    fig6, ax6 = plt.subplots(figsize=(12, 8))
    matriz_corr_inicial = df.corr(numeric_only=True)
    sns.heatmap(
        matriz_corr_inicial,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax6
    )
    ax6.set_title("Matriz de Correlación - Variables Iniciales Completas")
    st.pyplot(fig6)

except FileNotFoundError:
    st.error(" No se encontró el archivo de datos.")

 