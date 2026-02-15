import streamlit as st
from app.pages import page_boxplot
from app.pages import page_korrelation
from app.pages import page_wahrscheinlichkeit

st.set_page_config(page_title="Big Data Analytics", layout="wide")

pages = [
	st.Page(page_wahrscheinlichkeit, title="Wahrscheinlichkeit"),
	st.Page(page_korrelation, title="Korrelationskoeffizient"),
	st.Page(page_boxplot, title="Boxplot"),
]

navigation = st.navigation({"Analysen": pages})
navigation.run()
