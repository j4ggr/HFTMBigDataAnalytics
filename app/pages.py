import streamlit as st

from app.charts import chart_box_plot
from app.charts import chart_correlation_coefficient
from app.charts import chart_dice_probability_distribution


def page_wahrscheinlichkeit():
    st.title("Wahrscheinlichkeit")
    n_dices = st.slider("N Würfel", min_value=1, max_value=10, value=1, step=1)
    exp_samples = st.slider("Stichprobengröße (10^x)", min_value=1, max_value=6, value=2, step=1)
    kind = st.selectbox("Datenquelle", options=["Empirisch", "Theoretisch", "Beide"], index=0)

    figure = chart_dice_probability_distribution(n_dices=n_dices, exp_samples=exp_samples, kind=kind)
    st.pyplot(figure, clear_figure=True, use_container_width=True)


def page_korrelation():
    st.title("Korrelationskoeffizient")
    correlation_coefficient = st.slider("R", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)

    figure = chart_correlation_coefficient(correlation_coefficient=correlation_coefficient)
    st.pyplot(figure, clear_figure=True, use_container_width=True)


def page_boxplot():
    st.title("Boxplot")
    point = st.slider("Position", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)

    figure = chart_box_plot(point=point)
    st.pyplot(figure, clear_figure=True, use_container_width=True)