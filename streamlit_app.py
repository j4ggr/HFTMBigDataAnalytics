import streamlit as st
from app.pages import page_basics
from app.pages import page_preparation, page_boxplot, page_detrend, page_standartisation, page_normalization
from app.pages import page_mining, page_smoothing, page_kumulation, matrix_plot, page_korrelation
from app.pages import page_probability, page_tree, page_bayas, page_distribution
from app.pages import page_analysis, page_knn, page_log_regression, page_certainty, page_differences, page_fft, page_merit, page_kgoals

st.set_page_config(page_title="Big Data Analytics", layout="wide")


basics = [
	st.Page(page_basics, title="Einleitung"),
]

preparation = [
    st.Page(page_preparation, title="Einleitung"),
	st.Page(page_boxplot, title="Boxplot"),
 	st.Page(page_detrend, title="Detrend"),
   	st.Page(page_standartisation, title="Standardisierung"),
    st.Page(page_normalization, title="Normierung"),
]

mining = [
    st.Page(page_mining, title="Einleitung"),
    st.Page(page_smoothing, title="Glättung"),
    st.Page(page_kumulation, title="Kumulation"),
    st.Page(matrix_plot, title="Matrix Plot"),
	st.Page(page_korrelation, title="Korrelationskoeffizient"),

]

probability = [
    st.Page(page_probability, title="Einleitung"),
	st.Page(page_distribution, title="Wahrscheinlichkeitsverteilung"),
 	st.Page(page_bayas, title="Multiplikationsregel"),
  	st.Page(page_tree, title="Baumdiagramm"),
]

analysis = [
    st.Page(page_analysis, title="Einleitung"),
    st.Page(page_knn, title="k nächste Nachbarn"),
    st.Page(page_log_regression, title="Logarithmische Regresion"),
    st.Page(page_certainty, title="Bestimmtheitsmass"),
    st.Page(page_differences, title="Extremale Datenwerte aus Differenzen"),
    st.Page(page_fft, title="Frequenzanalyse (FFT)"),
    st.Page(page_merit, title="Gütekennzahl (figure of merit)"),
    st.Page(page_kgoals, title="Optimierung von Zielkonflikten"),

]


navigation = st.navigation({
    "Grundlagen": basics,
	"Datenvorbereitung": preparation,
	"Data Mining": mining,
    "Warscheinlichkeit": probability,
    "Datenanalyse": analysis,
    }
)
navigation.run()
