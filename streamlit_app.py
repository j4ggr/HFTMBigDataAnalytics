import streamlit as st
from app.i18n import select_language, t, translate_streamlit_text
from app.pages import page_basics
from app.pages import page_preparation, page_boxplot, page_detrend, page_standartisation, page_normalization
from app.pages import page_mining, page_smoothing, page_kumulation, matrix_plot, page_korrelation
from app.pages import page_probability, page_tree, page_bayas, page_distribution
from app.pages import page_analysis, page_knn, page_log_regression, page_certainty, page_differences, page_fft, page_merit, page_kgoals

st.set_page_config(page_title="Big Data Analytics", layout="wide")
select_language()
translate_streamlit_text()


basics = [
	st.Page(page_basics, title=t("Einleitung")),
]

preparation = [
    st.Page(page_preparation, title=t("Einleitung")),
	st.Page(page_boxplot, title="Boxplot"),
 	st.Page(page_detrend, title=t("Detrend")),
   	st.Page(page_standartisation, title=t("Standardisierung")),
    st.Page(page_normalization, title=t("Normierung")),
]

mining = [
    st.Page(page_mining, title=t("Einleitung")),
    st.Page(page_smoothing, title=t("Glättung")),
    st.Page(page_kumulation, title=t("Kumulation")),
    st.Page(matrix_plot, title="Matrix Plot"),
	st.Page(page_korrelation, title=t("Korrelationskoeffizient")),

]

probability = [
    st.Page(page_probability, title=t("Einleitung")),
	st.Page(page_distribution, title=t("Wahrscheinlichkeitsverteilung")),
 	st.Page(page_bayas, title=t("Multiplikationsregel")),
  	st.Page(page_tree, title=t("Baumdiagramm")),
]

analysis = [
    st.Page(page_analysis, title=t("Einleitung")),
    st.Page(page_knn, title=t("k nächste Nachbarn")),
    st.Page(page_log_regression, title=t("Logarithmische Regresion")),
    st.Page(page_certainty, title=t("Bestimmtheitsmass")),
    st.Page(page_differences, title=t("Extremale Datenwerte aus Differenzen")),
    st.Page(page_fft, title=t("Frequenzanalyse (FFT)")),
    st.Page(page_merit, title=t("Gütekennzahl (figure of merit)")),
    st.Page(page_kgoals, title=t("Optimierung von Zielkonflikten")),

]


navigation = st.navigation({
    t("Grundlagen"): basics,
	t("Datenvorbereitung"): preparation,
	"Data Mining": mining,
    t("Warscheinlichkeit"): probability,
    t("Datenanalyse"): analysis,
    }
)
navigation.run()
