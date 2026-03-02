import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from graphviz import Digraph
from scipy.signal import detrend
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


from app.charts import chart_box_plot
from app.charts import chart_correlation_coefficient
from app.charts import chart_dice_probability_distribution

###############################################################################
# GRUNDLAGEN
###############################################################################
def page_basics():
    st.title("Big Data Analytics: Grundlagen")
    
    st.text("In der heutigen digitalen Welt entstehen Daten in einem unglaublichen Tempo - von Social-Media-Posts über Sensoren im Internet der Dinge bis zu Geschäftstransaktionen. Um diese Flut an Informationen sinnvoll zu nutzen, benötigen wir ein solides Verständnis der Grundlagen von Big Data. In diesem Kapitel lernen wir, was Big Data ausmacht, welche Technologien dahinterstecken und warum die fünf Vs (Volumen, Geschwindigkeit, Vielfalt, Verlässlichkeit und Wert) den Kern unserer Analyseprozesse bilden.")

    st.header("Die 5 Vs von Big Data")
    st.markdown("""
    1. **Volume (Volumen)**: Grosse Datenmengen  
    2. **Velocity (Geschwindigkeit)**: Echtzeit-Daten  
    3. **Variety (Vielfalt)**: Strukturiert & unstrukturiert  
    4. **Veracity (Verlässlichkeit)**: Datenqualität  
    5. **Value (Wert)**: Geschäftlicher Nutzen
    """)

    st.header("Beispiel: Datenvolumen visualisieren")
    import numpy as np
    import pandas as pd
    import altair as alt

    volumen = pd.DataFrame({
        'Datenquelle': ['Sensoren', 'Logs', 'Social Media', 'IoT', 'Datenbanken'],
        'Volumen (TB)': [50, 120, 200, 80, 150]
    })

    chart = alt.Chart(volumen).mark_bar().encode(
        x='Datenquelle',
        y='Volumen (TB)',
        tooltip=['Datenquelle', 'Volumen (TB)']
    )
    st.altair_chart(chart, use_container_width=True)


###############################################################################
# Datenvorbereitung
###############################################################################
def page_preparation():
    st.title("Datenvorbereitung")
    
    st.text("Rohdaten sind selten perfekt - sie enthalten fehlende Werte, Fehler oder unterschiedliche Formate. Bevor wir spannende Muster entdecken oder Vorhersagen treffen können, müssen die Daten sorgfältig vorbereitet werden. Datenvorbereitung ist der entscheidende Schritt, der sicherstellt, dass Analysen valide Ergebnisse liefern. In diesem Kapitel erfahren wir, wie Daten bereinigt, transformiert, integriert und reduziert werden, um sie fit für Big-Data-Analysen zu machen.")

    st.header("Beispiel: Fehlende Werte behandeln")

    # Beispiel-Daten
    data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'Alter': [25, np.nan, 30, 22, np.nan],
        'Gehalt': [50000, 60000, np.nan, 45000, 70000]
    })

    st.write("Rohdaten:")
    st.dataframe(data)

    # Fehlende Werte füllen
    data_filled = data.fillna(data.mean(numeric_only=True))

    st.write("Daten nach Imputation:")
    st.dataframe(data_filled)


def page_boxplot():
    st.title("Boxplot")
    
    st.text("Ausreisser erkennt und verschiedene Features visuell vergleicht.")

    point = st.slider("Position", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)

    figure = chart_box_plot(point=point)
    st.pyplot(figure, clear_figure=True, use_container_width=True)
    
    # Zufallsdaten erstellen
    np.random.seed(42)
    data = pd.DataFrame({
        'Umsatz': np.random.normal(50000, 10000, 100),
        'Kosten': np.random.normal(30000, 5000, 100),
        'Gewinn': np.random.normal(15000, 7000, 100)
    })

    st.header("Zusatz: Verteilung und Ausreisser visualisieren")
    st.dataframe(data.head(10))

    # Feature-Auswahl für Boxplot
    feature = st.selectbox("Feature auswählen:", data.columns)

    st.subheader(f"Boxplot für {feature}")

    # Boxplot erstellen
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(y=data[feature], ax=ax)
    ax.set_title(f"Boxplot von {feature}")
    ax.set_ylabel("Wert")
    st.pyplot(fig)

    st.subheader("Beschreibung der Verteilung")
    st.write(data[feature].describe())

    st.subheader("Ausreisser identifizieren")
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    iqr = q3 - q1
    outliers = data[(data[feature] < q1 - 1.5*iqr) | (data[feature] > q3 + 1.5*iqr)]
    st.write(outliers)


def page_detrend():
    st.title("Zeitreihe Detrend")
    
    st.text("Das zeigt, wie man den langfristigen Trend entfernt, um saisonale Muster oder Schwankungen besser sichtbar zu machen.")

    st.header("Ziel: Trend aus einer Zeitreihe entfernen")

    # Simulierte Zeitreihe erzeugen
    np.random.seed(42)
    t = np.arange(0, 100)
    trend = 0.5 * t           # linearer Trend
    seasonal = 10 * np.sin(2 * np.pi * t / 20)  # saisonale Schwankung
    noise = np.random.normal(0, 2, len(t))
    y = trend + seasonal + noise

    # Detrending durchführen
    y_detrended = detrend(y)

    # Auswahl: Original oder detrended anzeigen
    option = st.radio("Welche Daten anzeigen?", ("Original", "Detrended"))

    fig, ax = plt.subplots(figsize=(10,5))

    if option == "Original":
        ax.plot(t, y, label="Originaldaten")
        ax.plot(t, trend, label="Trend", linestyle="--", color="red")
        ax.set_title("Originalzeitreihe mit Trend")
    else:
        ax.plot(t, y_detrended, label="Detrended Daten", color="green")
        ax.set_title("Trend entfernte Zeitreihe (Detrended)")

    ax.set_xlabel("Zeit")
    ax.set_ylabel("Wert")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Daten als Tabelle")
    df = pd.DataFrame({
        "Zeit": t,
        "Original": y,
        "Detrended": y_detrended
    })
    st.dataframe(df.head(10))


def page_standartisation():
    st.title("Standardisierung von Daten")
    
    st.text("Wir sehen, wie man Daten standardisiert (Mittelwert 0, Standardabweichung 1)")

    st.header("Ziel: Daten auf vergleichbare Skala bringen")

    # Zufallsdaten erstellen
    np.random.seed(42)
    data = pd.DataFrame({
        'Umsatz': np.random.normal(50000, 10000, 100),
        'Kosten': np.random.normal(30000, 5000, 100),
        'Gewinn': np.random.normal(15000, 7000, 100)
    })

    st.subheader("Rohdaten")
    st.dataframe(data.head(10))

    # Feature-Auswahl
    features = st.multiselect("Welche Features standardisieren?", data.columns.tolist(), default=data.columns.tolist())

    if features:
        scaler = StandardScaler()
        data_scaled = data.copy()
        data_scaled[features] = scaler.fit_transform(data[features])
        
        st.subheader("Standardisierte Daten")
        st.dataframe(data_scaled.head(10))
        
        st.subheader("Visualisierung: Vorher vs. Nachher")
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        
        # Vorher
        axes[0].boxplot(data[features].values, labels=features)
        axes[0].set_title("Originaldaten")
        
        # Nachher
        axes[1].boxplot(data_scaled[features].values, labels=features)
        axes[1].set_title("Standardisierte Daten")
        
        st.pyplot(fig)
 
 
def page_normalization():
    st.title("Normierung von Daten")
    
    st.text("Normierung skaliert die Daten auf einen festen Bereich, z. B. [0,1], und ist besonders nützlich, wenn Features unterschiedliche Skalen haben und für Machine-Learning-Modelle vergleichbar sein sollen.")

    st.header("Ziel: Daten auf einen festen Bereich [0,1] skalieren")

    # Zufallsdaten erzeugen
    np.random.seed(42)
    data = pd.DataFrame({
        'Umsatz': np.random.normal(50000, 10000, 100),
        'Kosten': np.random.normal(30000, 5000, 100),
        'Gewinn': np.random.normal(15000, 7000, 100)
    })

    st.subheader("Rohdaten")
    st.dataframe(data.head(10))

    # Feature-Auswahl
    features = st.multiselect("Welche Features normieren?", data.columns.tolist(), default=data.columns.tolist())

    if features:
        scaler = MinMaxScaler()
        data_norm = data.copy()
        data_norm[features] = scaler.fit_transform(data[features])
        
        st.subheader("Normierte Daten [0,1]")
        st.dataframe(data_norm.head(10))
        
        st.subheader("Visualisierung: Vorher vs. Nachher")
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        
        # Vorher
        axes[0].boxplot(data[features].values, labels=features)
        axes[0].set_title("Originaldaten")
        
        # Nachher
        axes[1].boxplot(data_norm[features].values, labels=features)
        axes[1].set_title("Normierte Daten [0,1]")
        
        st.pyplot(fig)       


###############################################################################
# Data Mining
###############################################################################
def page_mining():
    st.title("Data Mining: Clustering")
    
    st.text("Big Data ist nur dann wertvoll, wenn wir darin Wissen entdecken. Data Mining ist die methodische Suche nach Mustern, Zusammenhängen und Trends in grossen Datenmengen. Ob Klassifikation, Clustering oder Assoziationsanalyse - in diesem Kapitel lernen wir, wie wir Daten systematisch analysieren, um Entscheidungsprozesse zu unterstützen und verborgene Zusammenhänge aufzudecken.")

    st.header("Iris-Daten clustering Beispiel")

    iris = load_iris()
    X = iris.data
    df = pd.DataFrame(X, columns=iris.feature_names)

    n_clusters = st.slider("Anzahl Cluster:", min_value=2, max_value=5, value=3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    st.write(df.head())

    st.subheader("Cluster Verteilung")
    st.bar_chart(df['Cluster'].value_counts())


def page_smoothing():
    st.title("Glättung von Zeitreihen")
    
    st.text("Glättung hilft, Rauschen zu reduzieren und Trends oder saisonale Muster deutlicher sichtbar zu machen. Ich zeige sowohl gleitenden Mittelwert (Moving Average) als auch Exponentielle Glättung.")

    st.header("Ziel: Rauschen reduzieren und Trends sichtbar machen")

    # Simulierte Zeitreihe erzeugen
    np.random.seed(42)
    t = np.arange(0, 100)
    trend = 0.3 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 20)
    noise = np.random.normal(0, 3, len(t))
    y = trend + seasonal + noise

    df = pd.DataFrame({'Zeit': t, 'Wert': y})

    st.subheader("Originaldaten")
    st.line_chart(df.set_index('Zeit'))

    # Glättungsmethode auswählen
    method = st.selectbox("Glättungsmethode wählen:", ["Gleitender Mittelwert", "Exponentielle Glättung"])

    if method == "Gleitender Mittelwert":
        window = st.slider("Fenstergrösse:", min_value=2, max_value=20, value=5)
        df['Geglättet'] = df['Wert'].rolling(window=window).mean()
        st.subheader(f"Geglättete Daten (Moving Average, Fenster={window})")
        st.line_chart(df.set_index('Zeit')[['Wert','Geglättet']])
        
    elif method == "Exponentielle Glättung":
        alpha = st.slider("Glättungsfaktor α:", min_value=0.01, max_value=1.0, value=0.3, step=0.01)
        model = ExponentialSmoothing(df['Wert'], trend=None, seasonal=None)
        df['Geglättet'] = model.fit(smoothing_level=alpha).fittedvalues
        st.subheader(f"Geglättete Daten (Exponentielle Glättung, α={alpha})")
        st.line_chart(df.set_index('Zeit')[['Wert','Geglättet']])


def page_kumulation():
    st.title("Kumulation von Daten")
    
    st.text("Kumulation bedeutet, dass man Werte aufsummiert, z. B. um einen laufenden Gesamtwert oder aggregierte Trends über die Zeit zu berechnen.")

    st.header("Ziel: Laufende Summe / kumulierte Werte berechnen")

    # Simulierte tägliche Umsätze erzeugen
    np.random.seed(42)
    tage = pd.date_range(start="2026-01-01", periods=30, freq="D")
    umsatz = np.random.randint(1000, 5000, size=len(tage))

    df = pd.DataFrame({'Datum': tage, 'Täglicher Umsatz': umsatz})

    st.subheader("Rohdaten")
    st.dataframe(df)

    # Kumulation berechnen
    df['Kumulativer Umsatz'] = df['Täglicher Umsatz'].cumsum()

    st.subheader("Kumulativer Umsatz")
    st.dataframe(df)

    # Visualisierung
    st.subheader("Visualisierung: Täglicher vs. kumulativer Umsatz")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['Datum'], df['Täglicher Umsatz'], marker='o', label="Täglicher Umsatz")
    ax.plot(df['Datum'], df['Kumulativer Umsatz'], marker='x', linestyle='--', label="Kumulativer Umsatz")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Umsatz")
    ax.set_title("Kumulation von Umsätzen")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)


def matrix_plot():
    st.title("Matrix Plot / Pairplot")
    
    st.text("Ein Matrix Plot (oft auch als Pairplot bezeichnet) ist ideal, um Zusammenhänge zwischen mehreren Variablen gleichzeitig zu visualisieren.")

    st.header("Ziel: Zusammenhänge zwischen mehreren Variablen visualisieren")

    # Zufallsdaten erstellen
    np.random.seed(42)
    data = pd.DataFrame({
        'Umsatz': np.random.normal(50000, 10000, 100),
        'Kosten': np.random.normal(30000, 5000, 100),
        'Gewinn': np.random.normal(15000, 7000, 100),
        'Mitarbeiter': np.random.randint(5, 50, 100)
    })

    st.subheader("Rohdaten")
    st.dataframe(data.head(10))

    # Features für Matrix Plot auswählen
    features = st.multiselect("Welche Features visualisieren?", data.columns.tolist(), default=data.columns.tolist())

    if len(features) >= 2:
        st.subheader("Matrix Plot")
        fig = sns.pairplot(data[features])
        st.pyplot(fig)
    else:
        st.warning("Bitte wähle mindestens 2 Features aus, um einen Matrix Plot zu erstellen.")


def page_korrelation():
    st.title("Korrelationskoeffizient")
    
    st.text("Dies ist ein zentraler Schritt in Big Data Analytics, um lineare Zusammenhänge zwischen Variablen zu erkennen.")
    
    correlation_coefficient = st.slider("R", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)

    figure = chart_correlation_coefficient(correlation_coefficient=correlation_coefficient)
    st.pyplot(figure, clear_figure=True, use_container_width=True)
    
    st.header("Zusatz: Zusammenhänge zwischen Variablen messen")

    # Zufallsdaten erstellen
    np.random.seed(42)
    data = pd.DataFrame({
        'Umsatz': np.random.normal(50000, 10000, 100),
        'Kosten': np.random.normal(30000, 5000, 100),
        'Gewinn': np.random.normal(15000, 7000, 100),
        'Mitarbeiter': np.random.randint(5, 50, 100)
    })

    st.subheader("Rohdaten")
    st.dataframe(data.head(10))

    # Feature-Auswahl
    features = st.multiselect("Welche Features korrelieren?", data.columns.tolist(), default=data.columns.tolist())

    if len(features) >= 2:
        # Korrelationsmatrix berechnen
        corr_matrix = data[features].corr()
        st.subheader("Korrelationsmatrix")
        st.dataframe(corr_matrix)
        
        # Heatmap visualisieren
        st.subheader("Heatmap der Korrelationsmatrix")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Bitte wähle mindestens 2 Features aus, um die Korrelation zu berechnen.")



###############################################################################
# Wahrscheinlichkeit
###############################################################################
def page_probability():
    st.title("Wahrscheinlichkeit")
    
    st.text("Datenanalysen sind immer mit Unsicherheit verbunden - wir können die Zukunft nicht genau vorhersagen, aber wir können Wahrscheinlichkeiten berechnen. Die Wahrscheinlichkeitsrechnung ist das Fundament, auf dem viele Big-Data-Methoden wie statistische Modelle, Risikobewertungen und Machine-Learning-Algorithmen aufbauen. In diesem Kapitel betrachten wir die Grundlagen der Wahrscheinlichkeit und wichtige Verteilungen, die uns helfen, Daten besser zu interpretieren.")

    st.header("Binomialverteilung interaktiv")

    n = st.slider("Anzahl Versuche (n):", min_value=1, max_value=50, value=10)
    p = st.slider("Erfolgswahrscheinlichkeit (p):", min_value=0.0, max_value=1.0, value=0.5)

    from scipy.stats import binom

    x = np.arange(0, n+1)
    prob = binom.pmf(x, n, p)

    st.subheader("Wahrscheinlichkeiten")
    st.dataframe(pd.DataFrame({'Erfolge': x, 'P(X=k)': prob}))

    st.subheader("Visualisierung")
    fig, ax = plt.subplots()
    ax.bar(x, prob)
    ax.set_xlabel("Anzahl Erfolge")
    ax.set_ylabel("Wahrscheinlichkeit")
    st.pyplot(fig)


def page_tree():
    st.title("🌳 Interaktives Baumdiagramm")
    st.latex(r"P(A \cap B) = P(A|B)\cdot P(B)")

    st.markdown("---")

    st.header("🔢 Wahrscheinlichkeiten eingeben")

    p_b = st.slider("P(B)", 0.0, 1.0, 0.6, 0.01)
    p_a_given_b = st.slider("P(A | B)", 0.0, 1.0, 0.8, 0.01)
    p_a_given_not_b = st.slider("P(A | ¬B)", 0.0, 1.0, 0.2, 0.01)

    p_not_b = 1 - p_b

    # Schnittwahrscheinlichkeiten
    p_ab = p_b * p_a_given_b
    p_a_not_b = p_not_b * p_a_given_not_b
    p_a_total = p_ab + p_a_not_b

    st.markdown("---")

    st.header("📊 Ergebnisse")

    st.latex(rf"P(A \cap B) = {p_a_given_b:.2f} \cdot {p_b:.2f} = {p_ab:.4f}")
    st.latex(rf"P(A) = {p_ab:.4f} + {p_a_not_b:.4f} = {p_a_total:.4f}")

    st.markdown("---")

    st.header("🌳 Baumdiagramm")

    dot = Digraph()

    dot.node("Start", "Start")
    dot.node("B", f"B\nP={p_b:.2f}")
    dot.node("notB", f"¬B\nP={p_not_b:.2f}")

    dot.node("AB", f"A\nP={p_a_given_b:.2f}\n∩={p_ab:.4f}")
    dot.node("notAB", f"¬A\nP={1-p_a_given_b:.2f}")

    dot.node("A_notB", f"A\nP={p_a_given_not_b:.2f}\n∩={p_a_not_b:.4f}")
    dot.node("notA_notB", f"¬A\nP={1-p_a_given_not_b:.2f}")

    dot.edge("Start", "B")
    dot.edge("Start", "notB")

    dot.edge("B", "AB")
    dot.edge("B", "notAB")

    dot.edge("notB", "A_notB")
    dot.edge("notB", "notA_notB")

    st.graphviz_chart(dot)

    st.markdown("---")

    st.subheader("🧠 Interpretation")

    st.write(
        f"Die Wahrscheinlichkeit, dass A und B gemeinsam eintreten, beträgt **{p_ab:.4f}**."
    )

    st.write(
        f"Die Gesamtwahrscheinlichkeit für A beträgt **{p_a_total:.4f}**."
    )


def page_bayas():
    st.title("📊 Multiplikationsregel der Wahrscheinlichkeit")
    st.markdown("### Formel:")
    st.latex(r"P(A \cap B) = P(A|B) \cdot P(B)")

    st.markdown("---")

    st.header("🔢 Eingabe")

    p_b = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    p_a_given_b = st.number_input("P(A | B)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    if st.button("Berechnen"):
        p_intersection = p_a_given_b * p_b
        
        st.markdown("## ✅ Ergebnis")
        st.latex(rf"P(A \cap B) = {p_a_given_b} \cdot {p_b}")
        st.latex(rf"P(A \cap B) = {p_intersection}")

        st.markdown("---")
        st.subheader("🧠 Interpretation")
        st.write(
            f"Die Wahrscheinlichkeit, dass A und B gemeinsam auftreten, beträgt **{p_intersection:.4f}**."
        )

    st.markdown("---")

    st.header("🔍 Unabhängigkeit prüfen (optional)")

    p_a = st.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    if st.button("Unabhängigkeit testen"):
        p_intersection = p_a_given_b * p_b
        independent_value = p_a * p_b

        st.markdown("### Vergleich:")
        st.write(f"P(A ∩ B) berechnet: {p_intersection:.4f}")
        st.write(f"P(A) · P(B): {independent_value:.4f}")

        if abs(p_intersection - independent_value) < 1e-6:
            st.success("A und B sind (nahezu) unabhängig.")
        else:
            st.warning("A und B sind abhängig.")


def page_distribution():
    st.title("Wahrscheinlichkeit")
    n_dices = st.slider("N Würfel", min_value=1, max_value=10, value=1, step=1)
    exp_samples = st.slider("Stichprobengrösse (10^x)", min_value=1, max_value=6, value=2, step=1)
    kind = st.selectbox("Datenquelle", options=["Empirisch", "Theoretisch", "Beide"], index=0)

    figure = chart_dice_probability_distribution(n_dices=n_dices, exp_samples=exp_samples, kind=kind)
    st.pyplot(figure, clear_figure=True, use_container_width=True)


###############################################################################
# Datenanalyse
###############################################################################
def page_analysis():
    st.title("Datenanalyse")
    st.text("Das Ziel von Big Data Analytics ist es, aus rohen Daten Erkenntnisse zu gewinnen. Datenanalyse verbindet Statistik, Visualisierung und maschinelles Lernen, um Trends, Zusammenhänge und Prognosen sichtbar zu machen. In diesem Kapitel geht es darum, wie wir Daten beschreiben, explorativ analysieren und daraus Entscheidungen ableiten können - von einfachen statistischen Kennzahlen bis zu aussagekräftigen Grafiken und Modellen.")

    st.header("Deskriptive Statistik Beispiel")

    # Zufallsdaten generieren
    np.random.seed(42)
    data = pd.DataFrame({
        'Umsatz': np.random.normal(50000, 10000, 100),
        'Kosten': np.random.normal(30000, 5000, 100)
    })

    st.write("Rohdaten:")
    st.dataframe(data.head())

    st.subheader("Statistik")
    st.write(data.describe())

    st.subheader("Visualisierung")
    fig, ax = plt.subplots()
    ax.scatter(data['Kosten'], data['Umsatz'])
    ax.set_xlabel("Kosten")
    ax.set_ylabel("Umsatz")
    ax.set_title("Kosten vs Umsatz")
    st.pyplot(fig)


def page_knn():
    st.title("k-Nearest Neighbors (kNN)")
    
    # Iris-Daten laden
    iris = load_iris()
    X = iris.data[:, :2]  # nur die ersten zwei Features für Visualisierung
    y = iris.target
    feature_names = iris.feature_names[:2]

    # Trainingsdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # k auswählen
    k = st.slider("Anzahl Nachbarn (k):", min_value=1, max_value=15, value=3)

    # kNN trainieren
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Interaktiver Punkt
    st.subheader("Neuen Punkt klassifizieren")
    x_input = st.slider(f"{feature_names[0]}:", float(X[:,0].min()-1), float(X[:,0].max()+1), float(X[:,0].mean()))
    y_input = st.slider(f"{feature_names[1]}:", float(X[:,1].min()-1), float(X[:,1].max()+1), float(X[:,1].mean()))

    new_point = np.array([[x_input, y_input]])
    pred_class = knn.predict(new_point)[0]
    st.write(f"Der neue Punkt wird der Klasse **{iris.target_names[pred_class]}** zugeordnet.")

    # Plot erstellen
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                        np.arange(y_min, y_max, 0.05))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8,6))
    cmap_light = mcolors.ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#00FF00', '#0000FF']

    # Entscheidungsflächen
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    # Trainingspunkte plotten
    for i, color, name in zip(range(len(iris.target_names)), cmap_bold, iris.target_names):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], X[idx, 1], c=color, label=name, edgecolor='k', s=50)

    # Neuen Punkt plotten
    ax.scatter(x_input, y_input, c='yellow', edgecolor='black', s=150, marker='X', label='Neuer Punkt')

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(f"kNN Klassifikation mit k={k}")
    ax.legend()
    st.pyplot(fig)
    
    
def page_log_regression():
    st.title("Logarithmische Regression")

    st.header("Ziel: Logarithmische Zusammenhänge modellieren")

    # Simulierte Daten
    np.random.seed(42)
    X = np.linspace(1, 20, 50).reshape(-1,1)  # X > 0, da log(X)
    y = 5 * np.log(X.flatten()) + 10 + np.random.normal(0, 1, size=X.shape[0])

    df = pd.DataFrame({'X': X.flatten(), 'y': y})
    st.subheader("Datenvorschau")
    st.dataframe(df.head(10))

    # Regression trainieren
    # Logarithmus auf X anwenden
    X_log = np.log(X)
    model = LinearRegression()
    model.fit(X_log, y)
    y_pred = model.predict(X_log)

    # Metriken
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.subheader("Modellmetriken")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Visualisierung
    st.subheader("Visualisierung: Originaldaten vs. Modell")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(X, y, color='blue', label='Originaldaten')
    ax.plot(X, y_pred, color='red', linewidth=2, label='Logarithmische Regression')
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Logarithmische Regression")
    ax.legend()
    st.pyplot(fig) 
    

def page_certainty():
    st.title("Bestimmtheitsmass (R²)")

    st.header("Ziel: Modellgüte mit R² messen")

    # Simulierte Daten
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1,1)
    y = 3 * X.flatten() + 7 + np.random.normal(0, 5, size=X.shape[0])

    df = pd.DataFrame({'X': X.flatten(), 'y': y})
    st.subheader("Datenvorschau")
    st.dataframe(df.head(10))

    # Lineare Regression trainieren
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Bestimmtheitsmass R² berechnen
    r2 = r2_score(y, y_pred)
    st.subheader("Bestimmtheitsmass")
    st.write(f"R² Score: {r2:.3f}  (0 ≤ R² ≤ 1, höher = besser)")

    # Visualisierung
    st.subheader("Visualisierung: Originaldaten vs. Modell")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(X, y, color='blue', label='Originaldaten')
    ax.plot(X, y_pred, color='red', linewidth=2, label='Lineares Modell')
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Lineare Regression mit R²")
    ax.legend()
    st.pyplot(fig)

    # Residuenvisualisierung
    st.subheader("Residuen (Fehler)")
    residuals = y - y_pred
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.scatter(X, residuals, color='purple')
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Residuen")
    ax2.set_title("Residuen der Vorhersagen")
    st.pyplot(fig2)
    

def page_differences():
    st.title("Erkennung extremer Datenwerte mit Differenzen")

    st.markdown("""
    Dieses Beispiel berechnet Vorwärts-, Rückwärts- und zentrale Differenzen und erkennt
    lokale Maxima und Minima, indem das Vorzeichen der Differenzen wechselt.
    """)

    # Simulierte Daten mit mehreren Extrema
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.normal(size=x.size)

    df = pd.DataFrame({'x': x, 'y': y})

    # Differenzen berechnen
    df['df'] = df['y'].shift(-1) - df['y']        # Vorwärtsdifferenz
    df['db'] = df['y'] - df['y'].shift(1)         # Rückwärtsdifferenz
    df['dc'] = 0.5 * (df['df'] + df['db'])        # Zentrale Differenz

    # Vorzeichenwechsel erkennen (von positiv zu negativ = lokales Maximum)
    df['df_sign_change'] = df['df'].shift(1) * df['df'] < 0
    df['db_sign_change'] = df['db'].shift(1) * df['db'] < 0
    df['dc_sign_change'] = df['dc'].shift(1) * df['dc'] < 0

    # Lokale Maxima: Differenz wechselt von positiv zu negativ
    df['local_max'] = ((df['df'].shift(1) > 0) & (df['df'] < 0)) | \
                    ((df['db'].shift(1) > 0) & (df['db'] < 0)) | \
                    ((df['dc'].shift(1) > 0) & (df['dc'] < 0))

    # Lokale Minima: Differenz wechselt von negativ zu positiv
    df['local_min'] = ((df['df'].shift(1) < 0) & (df['df'] > 0)) | \
                    ((df['db'].shift(1) < 0) & (df['db'] > 0)) | \
                    ((df['dc'].shift(1) < 0) & (df['dc'] > 0))

    st.subheader("Daten mit erkannten Extrema")
    st.dataframe(df[['x', 'y', 'local_max', 'local_min']].fillna(False))

    # Visualisierung
    st.subheader("Visualisierung")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['x'], df['y'], label='Daten')
    ax.scatter(df.loc[df['local_max'], 'x'], df.loc[df['local_max'], 'y'], color='red', label='Lokale Maxima', s=100)
    ax.scatter(df.loc[df['local_min'], 'x'], df.loc[df['local_min'], 'y'], color='green', label='Lokale Minima', s=100)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Lokale Maxima und Minima erkannt durch Differenzen")
    ax.legend()
    st.pyplot(fig) 
    

def page_fft():
    st.title("Frequenzanalyse mit FFT")

    st.text("Das Beispiel erzeugt eine Kombination aus mehreren Sinuswellen, führt die FFT durch und zeigt sowohl das Zeit- als auch das Frequenzspektrum an.")

    st.header("Ziel: Frequenzanteile eines Signals sichtbar machen")

    # Parameter für Signal
    st.sidebar.header("Signalparameter")
    freq1 = st.sidebar.slider("Frequenz 1 (Hz)", 1, 50, 5)
    freq2 = st.sidebar.slider("Frequenz 2 (Hz)", 1, 50, 15)
    freq3 = st.sidebar.slider("Frequenz 3 (Hz)", 1, 50, 30)
    ampl1 = st.sidebar.slider("Amplitude 1", 0.1, 5.0, 1.0)
    ampl2 = st.sidebar.slider("Amplitude 2", 0.1, 5.0, 0.5)
    ampl3 = st.sidebar.slider("Amplitude 3", 0.1, 5.0, 0.3)
    noise_level = st.sidebar.slider("Rauschpegel", 0.0, 1.0, 0.1)

    # Zeitachse
    fs = 200  # Abtastfrequenz in Hz
    t = np.linspace(0, 2, 2 * fs, endpoint=False)  # 2 Sekunden

    # Signal generieren
    signal = (ampl1 * np.sin(2 * np.pi * freq1 * t) +
        ampl2 * np.sin(2 * np.pi * freq2 * t) +
        ampl3 * np.sin(2 * np.pi * freq3 * t) +
        noise_level * np.random.normal(size=t.shape))

    st.subheader("Signal im Zeitbereich")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, signal)
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Zeitbereichssignal")
    st.pyplot(fig)

    # FFT berechnen
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, 1/fs)

    # Nur positive Frequenzen betrachten
    pos_mask = fft_freq >= 0
    fft_freq = fft_freq[pos_mask]
    fft_power = np.abs(fft_vals[pos_mask]) ** 2

    st.subheader("Frequenzspektrum (Leistung)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.stem(fft_freq, fft_power)
    ax2.set_xlabel("Frequenz (Hz)")
    ax2.set_ylabel("Leistung")
    ax2.set_title("Leistungsspektrum des Signals")
    ax2.set_xlim(0, fs/2)
    st.pyplot(fig2)


def page_merit():
    st.title("Gütekennzahl & Risikoanalyse – Beispiel")

    st.markdown("""
    Gütekennzahlen sind oft zeitabhängige Verhältnisse, die Entscheidungen bewerten.
    Hier ein Beispiel für das Risiko eines Ereignisses über die Zeit.

    Formeln:

    - \( T = \) mittlere Zeit bis zum Ereignis
    - \( P(t) = 1 - e^{-t/T} \) Ereigniswahrscheinlichkeit
    - Risiko \( = P(t) \times \text{Schaden} \)
    """)

    # Parameter
    T = st.slider("Mittlere Zeit bis zum Ereignis T [Jahre]", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
    damage = st.slider("Schaden [CHF]", min_value=1000, max_value=100000, value=20000, step=1000)

    # Zeitachse
    t = np.linspace(0, 7, 500)

    # Ereigniswahrscheinlichkeit berechnen
    P_t = 1 - np.exp(-t / T)

    # Risiko berechnen
    risk = P_t * damage

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, P_t, label="Ereigniswahrscheinlichkeit $P(t)$", color='green')
    ax.set_xlabel("Zeit $t$ [Jahre]")
    ax.set_ylabel("Wahrscheinlichkeit / Risiko")

    ax2 = ax.twinx()
    ax2.plot(t, risk, label="Risiko (CHF)", color='red')
    ax2.set_ylabel("Risiko [CHF]")

    # Legende kombinieren
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

    ax.grid(True)
    ax.set_title("Zeitabhängige Gütekennzahl: Ereigniswahrscheinlichkeit & Risiko")
    st.pyplot(fig)


def page_kgoals():
    st.title("Optimierung von Zielkonflikten")

    st.markdown("""
    In diesem Beispiel modellieren wir drei Zielgrössen (Kennzahlen) mit Zielkonflikten, die von der Zeit abhängen:

    - Wirtschaftlichkeit \(g_1(t) = n t \frac{P_2}{P_1}\)  
    - Qualität \(g_2(t) = 2 - e^{q t}\) (mit \(q < 0\))  
    - Produktivität \(g_3(t) = n t\)  

    Wir berechnen eine gewichtete Summe der normierten Kennzahlen und bestimmen den Zeitpunkt \(t\), an dem diese Summe maximal ist.

    Passen Sie Parameter und Gewichtungen an, um unterschiedliche Prioritäten zu simulieren.
    """)

    # Parameter Eingabe
    P1 = st.number_input("Stückpreis Werkzeug \(P_1\)", min_value=1.0, value=100.0, step=1.0, format="%.2f")
    P2 = st.number_input("Stückpreis Produkt \(P_2\)", min_value=1.0, value=300.0, step=1.0, format="%.2f")
    n = st.number_input("Produktionsrate \(n\) [Stück/Zeit]", min_value=1, value=10, step=1)
    q = st.slider("Abnahmerate der Produktionsqualität \(q\) (negativer Wert)", min_value=-0.05, max_value=-0.0001, value=-0.01, step=0.0001, format="%.5f")

    st.markdown("### Gewichtungen der Zielgrössen (müssen zusammen 1 ergeben)")
    w1 = st.slider("Gewichtung Wirtschaftlichkeit \(w_1\)", 0.0, 1.0, 0.4, step=0.05)
    w2 = st.slider("Gewichtung Qualität \(w_2\)", 0.0, 1.0, 0.4, step=0.05)
    w3 = st.slider("Gewichtung Produktivität \(w_3\)", 0.0, 1.0, 0.2, step=0.05)

    # Gewichtssumme prüfen und normalisieren falls nötig
    w_sum = w1 + w2 + w3
    if w_sum == 0:
        st.error("Die Summe der Gewichtungen darf nicht 0 sein!")
        st.stop()
    elif w_sum != 1.0:
        w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum
        st.info(f"Gewichtungen wurden automatisch normalisiert: w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}")

    # Zeitachse (einheitlich)
    t = np.linspace(0, 250, 1000)  # ausreichend fein für max-Suche

    # Kennzahlen berechnen
    g1 = n * t * (P2 / P1)         # Wirtschaftlichkeit
    g2 = 2 - np.exp(q * t)         # Qualität (nimmt ab, q < 0)
    g3 = n * t                    # Produktivität

    # Normalisierung auf [0,1] (Min-Max)
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val == 0:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    g1_norm = normalize(g1)
    g2_norm = normalize(g2)
    g3_norm = normalize(g3)

    # Gewichtete Summe (Gesamtzielwert)
    Z = w1 * g1_norm + w2 * g2_norm + w3 * g3_norm

    # Optimum bestimmen
    opt_idx = np.argmax(Z)
    opt_t = t[opt_idx]
    opt_val = Z[opt_idx]

    st.subheader(f"Optimale Zeit t = {opt_t:.2f} (max. Zielwert = {opt_val:.3f})")

    # Plot mit separaten y-Achsen für Übersichtlichkeit
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color_g1 = 'tab:purple'
    color_g2 = 'tab:green'
    color_g3 = 'tab:blue'
    color_Z = 'tab:red'

    # Zielgrössen
    ax1.plot(t, g1, label='Wirtschaftlichkeit \(g_1\)', color=color_g1)
    ax1.plot(t, g2, label='Qualität \(g_2\)', color=color_g2)
    ax1.plot(t, g3, label='Produktivität \(g_3\)', color=color_g3)
    ax1.set_xlabel('Zeit \(t\) [a.u.]')
    ax1.set_ylabel('Kennzahlen (Originalwerte)')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    # Zweite y-Achse für Gesamtzielwert
    ax2 = ax1.twinx()
    ax2.plot(t, Z, label='Gesamtzielwert \(Z\)', color=color_Z, linewidth=2)
    ax2.axvline(opt_t, color=color_Z, linestyle='--', label=f'Optimale Zeit: {opt_t:.2f}')
    ax2.set_ylabel('Gesamtzielwert \(Z\) (gewichtete Summe der normierten Kennzahlen)')
    ax2.tick_params(axis='y', colors=color_Z)
    ax2.legend(loc='upper right')

    ax1.grid(True)
    plt.title("Optimierung von Zielkonflikten: Wirtschaftlichkeit, Qualität & Produktivität")

    st.pyplot(fig)

    