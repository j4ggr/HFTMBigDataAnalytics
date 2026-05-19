import streamlit as st


LANGUAGES = {
    "de": "Deutsch",
    "en": "English",
    "fr": "Francais",
}


TRANSLATIONS = {
    "en": {
        "Sprache": "Language",
        "Einleitung": "Introduction",
        "Grundlagen": "Basics",
        "Datenvorbereitung": "Data Preparation",
        "Glättung": "Smoothing",
        "Detrend": "Detrend",
        "Standardisierung": "Standardization",
        "Normierung": "Normalization",
        "Kumulation": "Cumulation",
        "Korrelationskoeffizient": "Correlation Coefficient",
        "Warscheinlichkeit": "Probability",
        "Wahrscheinlichkeitsverteilung": "Probability Distribution",
        "Multiplikationsregel": "Multiplication Rule",
        "Baumdiagramm": "Tree Diagram",
        "Datenanalyse": "Data Analysis",
        "k nächste Nachbarn": "k-Nearest Neighbors",
        "Logarithmische Regresion": "Logarithmic Regression",
        "Bestimmtheitsmass": "Coefficient of Determination",
        "Extremale Datenwerte aus Differenzen": "Extreme Data Values from Differences",
        "Frequenzanalyse (FFT)": "Frequency Analysis (FFT)",
        "Gütekennzahl (figure of merit)": "Figure of Merit",
        "Optimierung von Zielkonflikten": "Optimization of Conflicting Goals",
        "Big Data Analytics: Grundlagen": "Big Data Analytics: Basics",
        "In der heutigen digitalen Welt entstehen Daten in einem unglaublichen Tempo - von Social-Media-Posts über Sensoren im Internet der Dinge bis zu Geschäftstransaktionen. Um diese Flut an Informationen sinnvoll zu nutzen, benötigen wir ein solides Verständnis der Grundlagen von Big Data. In diesem Kapitel lernen wir, was Big Data ausmacht, welche Technologien dahinterstecken und warum die fünf Vs (Volumen, Geschwindigkeit, Vielfalt, Verlässlichkeit und Wert) den Kern unserer Analyseprozesse bilden.": "In today's digital world, data is generated at an incredible pace, from social media posts and Internet of Things sensors to business transactions. To use this flood of information meaningfully, we need a solid understanding of the basics of big data. In this chapter, we learn what defines big data, which technologies are behind it, and why the five Vs (volume, velocity, variety, veracity, and value) form the core of our analytical processes.",
        "Die 5 Vs von Big Data": "The 5 Vs of Big Data",
        "\n    1. **Volume (Volumen)**: Grosse Datenmengen  \n    2. **Velocity (Geschwindigkeit)**: Echtzeit-Daten  \n    3. **Variety (Vielfalt)**: Strukturiert & unstrukturiert  \n    4. **Veracity (Verlässlichkeit)**: Datenqualität  \n    5. **Value (Wert)**: Geschäftlicher Nutzen\n    ": "\n    1. **Volume**: Large amounts of data  \n    2. **Velocity**: Real-time data  \n    3. **Variety**: Structured and unstructured data  \n    4. **Veracity**: Data quality  \n    5. **Value**: Business value\n    ",
        "Beispiel: Datenvolumen visualisieren": "Example: Visualizing Data Volume",
        "Datenvorbereitung": "Data Preparation",
        "Rohdaten sind selten perfekt - sie enthalten fehlende Werte, Fehler oder unterschiedliche Formate. Bevor wir spannende Muster entdecken oder Vorhersagen treffen können, müssen die Daten sorgfältig vorbereitet werden. Datenvorbereitung ist der entscheidende Schritt, der sicherstellt, dass Analysen valide Ergebnisse liefern. In diesem Kapitel erfahren wir, wie Daten bereinigt, transformiert, integriert und reduziert werden, um sie fit für Big-Data-Analysen zu machen.": "Raw data is rarely perfect; it contains missing values, errors, or different formats. Before we can discover meaningful patterns or make predictions, the data must be prepared carefully. Data preparation is the decisive step that ensures analyses deliver valid results. In this chapter, we learn how data is cleaned, transformed, integrated, and reduced to make it ready for big data analytics.",
        "Beispiel: Fehlende Werte behandeln": "Example: Handling Missing Values",
        "Rohdaten:": "Raw Data:",
        "Daten nach Imputation:": "Data After Imputation:",
        "Ausreisser erkennt und verschiedene Features visuell vergleicht.": "Identifies outliers and visually compares different features.",
        "Position": "Position",
        "Zusatz: Verteilung und Ausreisser visualisieren": "Extra: Visualize Distribution and Outliers",
        "Feature auswählen:": "Select Feature:",
        "Beschreibung der Verteilung": "Distribution Description",
        "Ausreisser identifizieren": "Identify Outliers",
        "Zeitreihe Detrend": "Time Series Detrending",
        "Das zeigt, wie man den langfristigen Trend entfernt, um saisonale Muster oder Schwankungen besser sichtbar zu machen.": "This shows how to remove the long-term trend to make seasonal patterns or fluctuations more visible.",
        "Ziel: Trend aus einer Zeitreihe entfernen": "Goal: Remove the Trend from a Time Series",
        "Welche Daten anzeigen?": "Which data should be displayed?",
        "Daten als Tabelle": "Data as Table",
        "Standardisierung von Daten": "Data Standardization",
        "Wir sehen, wie man Daten standardisiert (Mittelwert 0, Standardabweichung 1)": "We see how to standardize data (mean 0, standard deviation 1).",
        "Ziel: Daten auf vergleichbare Skala bringen": "Goal: Put Data on a Comparable Scale",
        "Rohdaten": "Raw Data",
        "Welche Features standardisieren?": "Which features should be standardized?",
        "Standardisierte Daten": "Standardized Data",
        "Visualisierung: Vorher vs. Nachher": "Visualization: Before vs. After",
        "Normierung von Daten": "Data Normalization",
        "Normierung skaliert die Daten auf einen festen Bereich, z. B. [0,1], und ist besonders nützlich, wenn Features unterschiedliche Skalen haben und für Machine-Learning-Modelle vergleichbar sein sollen.": "Normalization scales data to a fixed range, e.g. [0,1], and is especially useful when features have different scales and need to be comparable for machine learning models.",
        "Ziel: Daten auf einen festen Bereich [0,1] skalieren": "Goal: Scale Data to a Fixed Range [0,1]",
        "Welche Features normieren?": "Which features should be normalized?",
        "Normierte Daten [0,1]": "Normalized Data [0,1]",
        "Data Mining: Clustering": "Data Mining: Clustering",
        "Big Data ist nur dann wertvoll, wenn wir darin Wissen entdecken. Data Mining ist die methodische Suche nach Mustern, Zusammenhängen und Trends in grossen Datenmengen. Ob Klassifikation, Clustering oder Assoziationsanalyse - in diesem Kapitel lernen wir, wie wir Daten systematisch analysieren, um Entscheidungsprozesse zu unterstützen und verborgene Zusammenhänge aufzudecken.": "Big data is only valuable when we discover knowledge in it. Data mining is the systematic search for patterns, relationships, and trends in large datasets. Whether classification, clustering, or association analysis, in this chapter we learn how to analyze data systematically to support decision-making and reveal hidden relationships.",
        "Iris-Daten clustering Beispiel": "Iris Data Clustering Example",
        "Anzahl Cluster:": "Number of Clusters:",
        "Cluster Verteilung": "Cluster Distribution",
        "Glättung von Zeitreihen": "Time Series Smoothing",
        "Glättung hilft, Rauschen zu reduzieren und Trends oder saisonale Muster deutlicher sichtbar zu machen. Ich zeige sowohl gleitenden Mittelwert (Moving Average) als auch Exponentielle Glättung.": "Smoothing helps reduce noise and make trends or seasonal patterns more visible. I show both moving average and exponential smoothing.",
        "Ziel: Rauschen reduzieren und Trends sichtbar machen": "Goal: Reduce Noise and Make Trends Visible",
        "Originaldaten": "Original Data",
        "Glättungsmethode wählen:": "Choose Smoothing Method:",
        "Fenstergrösse:": "Window Size:",
        "Glättungsfaktor α:": "Smoothing Factor α:",
        "Kumulation von Daten": "Data Cumulation",
        "Kumulation bedeutet, dass man Werte aufsummiert, z. B. um einen laufenden Gesamtwert oder aggregierte Trends über die Zeit zu berechnen.": "Cumulation means summing values, for example to calculate a running total or aggregated trends over time.",
        "Ziel: Laufende Summe / kumulierte Werte berechnen": "Goal: Calculate Running Sums / Cumulative Values",
        "Kumulativer Umsatz": "Cumulative Revenue",
        "Visualisierung: Täglicher vs. kumulativer Umsatz": "Visualization: Daily vs. Cumulative Revenue",
        "Matrix Plot / Pairplot": "Matrix Plot / Pairplot",
        "Ein Matrix Plot (oft auch als Pairplot bezeichnet) ist ideal, um Zusammenhänge zwischen mehreren Variablen gleichzeitig zu visualisieren.": "A matrix plot, often called a pairplot, is ideal for visualizing relationships between multiple variables at the same time.",
        "Ziel: Zusammenhänge zwischen mehreren Variablen visualisieren": "Goal: Visualize Relationships Between Several Variables",
        "Welche Features visualisieren?": "Which Features Should Be Visualized?",
        "Bitte wähle mindestens 2 Features aus, um einen Matrix Plot zu erstellen.": "Please select at least 2 features to create a matrix plot.",
        "Dies ist ein zentraler Schritt in Big Data Analytics, um lineare Zusammenhänge zwischen Variablen zu erkennen.": "This is a central step in big data analytics for detecting linear relationships between variables.",
        "Zusatz: Zusammenhänge zwischen Variablen messen": "Extra: Measure Relationships Between Variables",
        "Welche Features korrelieren?": "Which Features Should Be Correlated?",
        "Korrelationsmatrix": "Correlation Matrix",
        "Heatmap der Korrelationsmatrix": "Correlation Matrix Heatmap",
        "Bitte wähle mindestens 2 Features aus, um die Korrelation zu berechnen.": "Please select at least 2 features to calculate the correlation.",
        "Wahrscheinlichkeit": "Probability",
        "Datenanalysen sind immer mit Unsicherheit verbunden - wir können die Zukunft nicht genau vorhersagen, aber wir können Wahrscheinlichkeiten berechnen. Die Wahrscheinlichkeitsrechnung ist das Fundament, auf dem viele Big-Data-Methoden wie statistische Modelle, Risikobewertungen und Machine-Learning-Algorithmen aufbauen. In diesem Kapitel betrachten wir die Grundlagen der Wahrscheinlichkeit und wichtige Verteilungen, die uns helfen, Daten besser zu interpretieren.": "Data analyses are always connected with uncertainty. We cannot predict the future exactly, but we can calculate probabilities. Probability theory is the foundation on which many big data methods, such as statistical models, risk assessments, and machine learning algorithms, are built. In this chapter, we look at the basics of probability and important distributions that help us interpret data better.",
        "Binomialverteilung interaktiv": "Interactive Binomial Distribution",
        "Anzahl Versuche (n):": "Number of Trials (n):",
        "Erfolgswahrscheinlichkeit (p):": "Success Probability (p):",
        "Wahrscheinlichkeiten": "Probabilities",
        "Visualisierung": "Visualization",
        "Interaktives Baumdiagramm": "Interactive Tree Diagram",
        "🌳 Interaktives Baumdiagramm": "Interactive Tree Diagram",
        "Wahrscheinlichkeiten eingeben": "Enter Probabilities",
        "Ergebnisse": "Results",
        "Baumdiagramm": "Tree Diagram",
        "Interpretation": "Interpretation",
        "Multiplikationsregel der Wahrscheinlichkeit": "Multiplication Rule of Probability",
        "📊 Multiplikationsregel der Wahrscheinlichkeit": "Multiplication Rule of Probability",
        "### Formel:": "### Formula:",
        "Eingabe": "Input",
        "Berechnen": "Calculate",
        "## ✅ Ergebnis": "## Result",
        "Unabhängigkeit prüfen (optional)": "Check Independence (optional)",
        "Unabhängigkeit testen": "Test Independence",
        "### Vergleich:": "### Comparison:",
        "A und B sind (nahezu) unabhängig.": "A and B are almost independent.",
        "A und B sind abhängig.": "A and B are dependent.",
        "N Würfel": "N Dice",
        "Stichprobengrösse (10^x)": "Sample Size (10^x)",
        "Datenquelle": "Data Source",
        "Das Ziel von Big Data Analytics ist es, aus rohen Daten Erkenntnisse zu gewinnen. Datenanalyse verbindet Statistik, Visualisierung und maschinelles Lernen, um Trends, Zusammenhänge und Prognosen sichtbar zu machen. In diesem Kapitel geht es darum, wie wir Daten beschreiben, explorativ analysieren und daraus Entscheidungen ableiten können - von einfachen statistischen Kennzahlen bis zu aussagekräftigen Grafiken und Modellen.": "The goal of big data analytics is to gain insights from raw data. Data analysis combines statistics, visualization, and machine learning to make trends, relationships, and forecasts visible. In this chapter, we focus on how we describe data, analyze it exploratively, and derive decisions from it, from simple statistical measures to meaningful charts and models.",
        "Deskriptive Statistik Beispiel": "Descriptive Statistics Example",
        "Statistik": "Statistics",
        "Neuen Punkt klassifizieren": "Classify New Point",
        "Logarithmische Regression": "Logarithmic Regression",
        "Ziel: Logarithmische Zusammenhänge modellieren": "Goal: Model Logarithmic Relationships",
        "Datenvorschau": "Data Preview",
        "Modellmetriken": "Model Metrics",
        "Visualisierung: Originaldaten vs. Modell": "Visualization: Original Data vs. Model",
        "Bestimmtheitsmass (R²)": "Coefficient of Determination (R²)",
        "Ziel: Modellgüte mit R² messen": "Goal: Measure Model Quality with R²",
        "Bestimmtheitsmass": "Coefficient of Determination",
        "Residuen (Fehler)": "Residuals (Errors)",
        "Erkennung extremer Datenwerte mit Differenzen": "Detecting Extreme Data Values with Differences",
        "\n    Dieses Beispiel berechnet Vorwärts-, Rückwärts- und zentrale Differenzen und erkennt\n    lokale Maxima und Minima, indem das Vorzeichen der Differenzen wechselt.\n    ": "\n    This example calculates forward, backward, and central differences and detects\n    local maxima and minima when the sign of the differences changes.\n    ",
        "Daten mit erkannten Extrema": "Data with Detected Extrema",
        "Frequenzanalyse mit FFT": "Frequency Analysis with FFT",
        "Das Beispiel erzeugt eine Kombination aus mehreren Sinuswellen, führt die FFT durch und zeigt sowohl das Zeit- als auch das Frequenzspektrum an.": "The example creates a combination of several sine waves, performs the FFT, and shows both the time and frequency spectra.",
        "Ziel: Frequenzanteile eines Signals sichtbar machen": "Goal: Make Frequency Components of a Signal Visible",
        "Signalparameter": "Signal Parameters",
        "Frequenz 1 (Hz)": "Frequency 1 (Hz)",
        "Frequenz 2 (Hz)": "Frequency 2 (Hz)",
        "Frequenz 3 (Hz)": "Frequency 3 (Hz)",
        "Amplitude 1": "Amplitude 1",
        "Amplitude 2": "Amplitude 2",
        "Amplitude 3": "Amplitude 3",
        "Rauschpegel": "Noise Level",
        "Signal im Zeitbereich": "Signal in Time Domain",
        "Frequenzspektrum (Leistung)": "Frequency Spectrum (Power)",
        "Gütekennzahl & Risikoanalyse – Beispiel": "Figure of Merit & Risk Analysis Example",
        "Mittlere Zeit bis zum Ereignis T [Jahre]": "Mean Time Until Event T [years]",
        "Schaden [CHF]": "Damage [CHF]",
        "Stückpreis Werkzeug \\(P_1\\)": "Unit Price Tool \\(P_1\\)",
        "Stückpreis Produkt \\(P_2\\)": "Unit Price Product \\(P_2\\)",
        "Produktionsrate \\(n\\) [Stück/Zeit]": "Production Rate \\(n\\) [units/time]",
        "Abnahmerate der Produktionsqualität \\(q\\) (negativer Wert)": "Decrease Rate of Production Quality \\(q\\) (negative value)",
        "### Gewichtungen der Zielgrössen (müssen zusammen 1 ergeben)": "### Weights of Target Variables (must sum to 1)",
        "Gewichtung Wirtschaftlichkeit \\(w_1\\)": "Weight Economic Efficiency \\(w_1\\)",
        "Gewichtung Qualität \\(w_2\\)": "Weight Quality \\(w_2\\)",
        "Gewichtung Produktivität \\(w_3\\)": "Weight Productivity \\(w_3\\)",
        "Die Summe der Gewichtungen darf nicht 0 sein!": "The sum of the weights must not be 0!",
        "Wahrscheinlichkeitsverteilung der Augensumme von Würfeln": "Probability Distribution of the Sum of Dice",
        "{n_samples} Würfe mit {n_dices} sechsseitigen Würfeln{sub_title_addition}": "{n_samples} rolls with {n_dices} six-sided dice{sub_title_addition}",
        " (Empirisch)": " (Empirical)",
        " (Theoretisch)": " (Theoretical)",
        " (Empirisch vs. Theoretisch)": " (Empirical vs. Theoretical)",
        "Anzahl Beobachtungen": "Number of Observations",
        "Kumulative Beobachtungen": "Cumulative Observations",
        "Kumulative Wahrscheinlichkeit": "Cumulative Probability",
        "Augensumme": "Sum of Dice",
        "Box Plot mit einem dynamischen Punkt": "Box Plot with a Dynamic Point",
    },
    "fr": {
        "Sprache": "Langue",
        "Einleitung": "Introduction",
        "Grundlagen": "Bases",
        "Datenvorbereitung": "Preparation des donnees",
        "Glättung": "Lissage",
        "Detrend": "Detendancement",
        "Standardisierung": "Standardisation",
        "Normierung": "Normalisation",
        "Kumulation": "Cumul",
        "Korrelationskoeffizient": "Coefficient de correlation",
        "Warscheinlichkeit": "Probabilite",
        "Wahrscheinlichkeitsverteilung": "Distribution de probabilite",
        "Multiplikationsregel": "Regle de multiplication",
        "Baumdiagramm": "Arbre de probabilites",
        "Datenanalyse": "Analyse des donnees",
        "k nächste Nachbarn": "k plus proches voisins",
        "Logarithmische Regresion": "Regression logarithmique",
        "Bestimmtheitsmass": "Coefficient de determination",
        "Extremale Datenwerte aus Differenzen": "Valeurs extremes a partir des differences",
        "Frequenzanalyse (FFT)": "Analyse frequentielle (FFT)",
        "Gütekennzahl (figure of merit)": "Indicateur de qualite",
        "Optimierung von Zielkonflikten": "Optimisation de conflits d'objectifs",
        "Big Data Analytics: Grundlagen": "Big Data Analytics : bases",
        "In der heutigen digitalen Welt entstehen Daten in einem unglaublichen Tempo - von Social-Media-Posts über Sensoren im Internet der Dinge bis zu Geschäftstransaktionen. Um diese Flut an Informationen sinnvoll zu nutzen, benötigen wir ein solides Verständnis der Grundlagen von Big Data. In diesem Kapitel lernen wir, was Big Data ausmacht, welche Technologien dahinterstecken und warum die fünf Vs (Volumen, Geschwindigkeit, Vielfalt, Verlässlichkeit und Wert) den Kern unserer Analyseprozesse bilden.": "Dans le monde numerique actuel, les donnees sont produites a un rythme incroyable, des publications sur les reseaux sociaux aux capteurs de l'Internet des objets en passant par les transactions commerciales. Pour utiliser ce flot d'informations de maniere pertinente, nous avons besoin d'une solide comprehension des bases du big data. Dans ce chapitre, nous apprenons ce qui caracterise le big data, quelles technologies le soutiennent et pourquoi les cinq V (volume, vitesse, variete, veracite et valeur) forment le coeur de nos processus d'analyse.",
        "Die 5 Vs von Big Data": "Les 5 V du Big Data",
        "\n    1. **Volume (Volumen)**: Grosse Datenmengen  \n    2. **Velocity (Geschwindigkeit)**: Echtzeit-Daten  \n    3. **Variety (Vielfalt)**: Strukturiert & unstrukturiert  \n    4. **Veracity (Verlässlichkeit)**: Datenqualität  \n    5. **Value (Wert)**: Geschäftlicher Nutzen\n    ": "\n    1. **Volume** : grandes quantites de donnees  \n    2. **Velocity** : donnees en temps reel  \n    3. **Variety** : donnees structurees et non structurees  \n    4. **Veracity** : qualite des donnees  \n    5. **Value** : valeur commerciale\n    ",
        "Beispiel: Datenvolumen visualisieren": "Exemple : visualiser le volume des donnees",
        "Rohdaten sind selten perfekt - sie enthalten fehlende Werte, Fehler oder unterschiedliche Formate. Bevor wir spannende Muster entdecken oder Vorhersagen treffen können, müssen die Daten sorgfältig vorbereitet werden. Datenvorbereitung ist der entscheidende Schritt, der sicherstellt, dass Analysen valide Ergebnisse liefern. In diesem Kapitel erfahren wir, wie Daten bereinigt, transformiert, integriert und reduziert werden, um sie fit für Big-Data-Analysen zu machen.": "Les donnees brutes sont rarement parfaites : elles contiennent des valeurs manquantes, des erreurs ou des formats differents. Avant de decouvrir des motifs interessants ou de faire des predictions, les donnees doivent etre preparees avec soin. La preparation des donnees est l'etape decisive qui garantit des resultats d'analyse valides. Dans ce chapitre, nous voyons comment nettoyer, transformer, integrer et reduire les donnees pour les rendre pretes pour l'analyse big data.",
        "Beispiel: Fehlende Werte behandeln": "Exemple : traiter les valeurs manquantes",
        "Rohdaten:": "Donnees brutes :",
        "Daten nach Imputation:": "Donnees apres imputation :",
        "Ausreisser erkennt und verschiedene Features visuell vergleicht.": "Identifie les valeurs aberrantes et compare visuellement differentes variables.",
        "Position": "Position",
        "Zusatz: Verteilung und Ausreisser visualisieren": "Complement : visualiser la distribution et les valeurs aberrantes",
        "Feature auswählen:": "Choisir une variable :",
        "Beschreibung der Verteilung": "Description de la distribution",
        "Ausreisser identifizieren": "Identifier les valeurs aberrantes",
        "Zeitreihe Detrend": "Detendancement d'une serie temporelle",
        "Das zeigt, wie man den langfristigen Trend entfernt, um saisonale Muster oder Schwankungen besser sichtbar zu machen.": "Cela montre comment supprimer la tendance a long terme afin de rendre les motifs saisonniers ou les fluctuations plus visibles.",
        "Ziel: Trend aus einer Zeitreihe entfernen": "Objectif : supprimer la tendance d'une serie temporelle",
        "Welche Daten anzeigen?": "Quelles donnees afficher ?",
        "Daten als Tabelle": "Donnees sous forme de tableau",
        "Standardisierung von Daten": "Standardisation des donnees",
        "Wir sehen, wie man Daten standardisiert (Mittelwert 0, Standardabweichung 1)": "Nous voyons comment standardiser les donnees (moyenne 0, ecart-type 1).",
        "Ziel: Daten auf vergleichbare Skala bringen": "Objectif : mettre les donnees sur une echelle comparable",
        "Rohdaten": "Donnees brutes",
        "Welche Features standardisieren?": "Quelles variables standardiser ?",
        "Standardisierte Daten": "Donnees standardisees",
        "Visualisierung: Vorher vs. Nachher": "Visualisation : avant vs apres",
        "Normierung von Daten": "Normalisation des donnees",
        "Normierung skaliert die Daten auf einen festen Bereich, z. B. [0,1], und ist besonders nützlich, wenn Features unterschiedliche Skalen haben und für Machine-Learning-Modelle vergleichbar sein sollen.": "La normalisation met les donnees a l'echelle dans une plage fixe, par exemple [0,1], et elle est particulierement utile lorsque les variables ont des echelles differentes et doivent etre comparables pour les modeles de machine learning.",
        "Ziel: Daten auf einen festen Bereich [0,1] skalieren": "Objectif : mettre les donnees a l'echelle dans la plage [0,1]",
        "Welche Features normieren?": "Quelles variables normaliser ?",
        "Normierte Daten [0,1]": "Donnees normalisees [0,1]",
        "Big Data ist nur dann wertvoll, wenn wir darin Wissen entdecken. Data Mining ist die methodische Suche nach Mustern, Zusammenhängen und Trends in grossen Datenmengen. Ob Klassifikation, Clustering oder Assoziationsanalyse - in diesem Kapitel lernen wir, wie wir Daten systematisch analysieren, um Entscheidungsprozesse zu unterstützen und verborgene Zusammenhänge aufzudecken.": "Le big data n'a de valeur que si nous y decouvrons des connaissances. Le data mining est la recherche methodique de motifs, de relations et de tendances dans de grands ensembles de donnees. Qu'il s'agisse de classification, de clustering ou d'analyse d'association, ce chapitre montre comment analyser systematiquement les donnees pour soutenir les decisions et reveler des relations cachees.",
        "Iris-Daten clustering Beispiel": "Exemple de clustering des donnees Iris",
        "Anzahl Cluster:": "Nombre de clusters :",
        "Cluster Verteilung": "Distribution des clusters",
        "Glättung von Zeitreihen": "Lissage de series temporelles",
        "Glättung hilft, Rauschen zu reduzieren und Trends oder saisonale Muster deutlicher sichtbar zu machen. Ich zeige sowohl gleitenden Mittelwert (Moving Average) als auch Exponentielle Glättung.": "Le lissage aide a reduire le bruit et a rendre les tendances ou motifs saisonniers plus visibles. Je montre a la fois la moyenne mobile et le lissage exponentiel.",
        "Ziel: Rauschen reduzieren und Trends sichtbar machen": "Objectif : reduire le bruit et rendre les tendances visibles",
        "Originaldaten": "Donnees originales",
        "Glättungsmethode wählen:": "Choisir la methode de lissage :",
        "Fenstergrösse:": "Taille de fenetre :",
        "Glättungsfaktor α:": "Facteur de lissage α :",
        "Kumulation von Daten": "Cumul de donnees",
        "Kumulation bedeutet, dass man Werte aufsummiert, z. B. um einen laufenden Gesamtwert oder aggregierte Trends über die Zeit zu berechnen.": "Le cumul consiste a additionner des valeurs, par exemple pour calculer un total courant ou des tendances agregees dans le temps.",
        "Ziel: Laufende Summe / kumulierte Werte berechnen": "Objectif : calculer une somme courante / des valeurs cumulees",
        "Kumulativer Umsatz": "Chiffre d'affaires cumule",
        "Visualisierung: Täglicher vs. kumulativer Umsatz": "Visualisation : chiffre d'affaires quotidien vs cumule",
        "Matrix Plot / Pairplot": "Matrix plot / pairplot",
        "Ein Matrix Plot (oft auch als Pairplot bezeichnet) ist ideal, um Zusammenhänge zwischen mehreren Variablen gleichzeitig zu visualisieren.": "Un matrix plot, souvent appele pairplot, est ideal pour visualiser simultanement les relations entre plusieurs variables.",
        "Ziel: Zusammenhänge zwischen mehreren Variablen visualisieren": "Objectif : visualiser les relations entre plusieurs variables",
        "Welche Features visualisieren?": "Quelles variables visualiser ?",
        "Bitte wähle mindestens 2 Features aus, um einen Matrix Plot zu erstellen.": "Veuillez choisir au moins 2 variables pour creer un matrix plot.",
        "Dies ist ein zentraler Schritt in Big Data Analytics, um lineare Zusammenhänge zwischen Variablen zu erkennen.": "C'est une etape centrale de l'analyse big data pour detecter des relations lineaires entre variables.",
        "Zusatz: Zusammenhänge zwischen Variablen messen": "Complement : mesurer les relations entre variables",
        "Welche Features korrelieren?": "Quelles variables correler ?",
        "Korrelationsmatrix": "Matrice de correlation",
        "Heatmap der Korrelationsmatrix": "Heatmap de la matrice de correlation",
        "Bitte wähle mindestens 2 Features aus, um die Korrelation zu berechnen.": "Veuillez choisir au moins 2 variables pour calculer la correlation.",
        "Wahrscheinlichkeit": "Probabilite",
        "Datenanalysen sind immer mit Unsicherheit verbunden - wir können die Zukunft nicht genau vorhersagen, aber wir können Wahrscheinlichkeiten berechnen. Die Wahrscheinlichkeitsrechnung ist das Fundament, auf dem viele Big-Data-Methoden wie statistische Modelle, Risikobewertungen und Machine-Learning-Algorithmen aufbauen. In diesem Kapitel betrachten wir die Grundlagen der Wahrscheinlichkeit und wichtige Verteilungen, die uns helfen, Daten besser zu interpretieren.": "Les analyses de donnees sont toujours liees a l'incertitude : nous ne pouvons pas predire l'avenir exactement, mais nous pouvons calculer des probabilites. Le calcul des probabilites est la base de nombreuses methodes big data, comme les modeles statistiques, les evaluations des risques et les algorithmes de machine learning. Dans ce chapitre, nous etudions les bases des probabilites et des distributions importantes qui nous aident a mieux interpreter les donnees.",
        "Binomialverteilung interaktiv": "Distribution binomiale interactive",
        "Anzahl Versuche (n):": "Nombre d'essais (n) :",
        "Erfolgswahrscheinlichkeit (p):": "Probabilite de succes (p) :",
        "Wahrscheinlichkeiten": "Probabilites",
        "Visualisierung": "Visualisation",
        "Interaktives Baumdiagramm": "Arbre de probabilites interactif",
        "🌳 Interaktives Baumdiagramm": "Arbre de probabilites interactif",
        "Wahrscheinlichkeiten eingeben": "Saisir les probabilites",
        "Ergebnisse": "Resultats",
        "Baumdiagramm": "Arbre de probabilites",
        "Interpretation": "Interpretation",
        "Multiplikationsregel der Wahrscheinlichkeit": "Regle de multiplication des probabilites",
        "📊 Multiplikationsregel der Wahrscheinlichkeit": "Regle de multiplication des probabilites",
        "### Formel:": "### Formule :",
        "Eingabe": "Saisie",
        "Berechnen": "Calculer",
        "## ✅ Ergebnis": "## Resultat",
        "Unabhängigkeit prüfen (optional)": "Verifier l'independance (optionnel)",
        "Unabhängigkeit testen": "Tester l'independance",
        "### Vergleich:": "### Comparaison :",
        "A und B sind (nahezu) unabhängig.": "A et B sont presque independants.",
        "A und B sind abhängig.": "A et B sont dependants.",
        "N Würfel": "N des",
        "Stichprobengrösse (10^x)": "Taille de l'echantillon (10^x)",
        "Datenquelle": "Source des donnees",
        "Das Ziel von Big Data Analytics ist es, aus rohen Daten Erkenntnisse zu gewinnen. Datenanalyse verbindet Statistik, Visualisierung und maschinelles Lernen, um Trends, Zusammenhänge und Prognosen sichtbar zu machen. In diesem Kapitel geht es darum, wie wir Daten beschreiben, explorativ analysieren und daraus Entscheidungen ableiten können - von einfachen statistischen Kennzahlen bis zu aussagekräftigen Grafiken und Modellen.": "L'objectif de l'analyse big data est d'obtenir des connaissances a partir de donnees brutes. L'analyse de donnees combine statistiques, visualisation et machine learning pour rendre visibles les tendances, relations et previsions. Dans ce chapitre, nous voyons comment decrire les donnees, les analyser de facon exploratoire et en deduire des decisions, des simples indicateurs statistiques aux graphiques et modeles parlants.",
        "Deskriptive Statistik Beispiel": "Exemple de statistique descriptive",
        "Statistik": "Statistiques",
        "Neuen Punkt klassifizieren": "Classer un nouveau point",
        "Logarithmische Regression": "Regression logarithmique",
        "Ziel: Logarithmische Zusammenhänge modellieren": "Objectif : modeliser des relations logarithmiques",
        "Datenvorschau": "Apercu des donnees",
        "Modellmetriken": "Metriques du modele",
        "Visualisierung: Originaldaten vs. Modell": "Visualisation : donnees originales vs modele",
        "Bestimmtheitsmass (R²)": "Coefficient de determination (R²)",
        "Ziel: Modellgüte mit R² messen": "Objectif : mesurer la qualite du modele avec R²",
        "Bestimmtheitsmass": "Coefficient de determination",
        "Residuen (Fehler)": "Residus (erreurs)",
        "Erkennung extremer Datenwerte mit Differenzen": "Detection de valeurs extremes avec les differences",
        "\n    Dieses Beispiel berechnet Vorwärts-, Rückwärts- und zentrale Differenzen und erkennt\n    lokale Maxima und Minima, indem das Vorzeichen der Differenzen wechselt.\n    ": "\n    Cet exemple calcule les differences avant, arriere et centrales et detecte\n    les maxima et minima locaux lorsque le signe des differences change.\n    ",
        "Daten mit erkannten Extrema": "Donnees avec extrema detectes",
        "Frequenzanalyse mit FFT": "Analyse frequentielle avec FFT",
        "Das Beispiel erzeugt eine Kombination aus mehreren Sinuswellen, führt die FFT durch und zeigt sowohl das Zeit- als auch das Frequenzspektrum an.": "L'exemple cree une combinaison de plusieurs ondes sinusoidales, applique la FFT et affiche a la fois le spectre temporel et le spectre frequentiel.",
        "Ziel: Frequenzanteile eines Signals sichtbar machen": "Objectif : rendre visibles les composantes frequentielles d'un signal",
        "Signalparameter": "Parametres du signal",
        "Frequenz 1 (Hz)": "Frequence 1 (Hz)",
        "Frequenz 2 (Hz)": "Frequence 2 (Hz)",
        "Frequenz 3 (Hz)": "Frequence 3 (Hz)",
        "Amplitude 1": "Amplitude 1",
        "Amplitude 2": "Amplitude 2",
        "Amplitude 3": "Amplitude 3",
        "Rauschpegel": "Niveau de bruit",
        "Signal im Zeitbereich": "Signal dans le domaine temporel",
        "Frequenzspektrum (Leistung)": "Spectre frequentiel (puissance)",
        "Gütekennzahl & Risikoanalyse – Beispiel": "Indicateur de qualite & analyse du risque - exemple",
        "Mittlere Zeit bis zum Ereignis T [Jahre]": "Temps moyen jusqu'a l'evenement T [annees]",
        "Schaden [CHF]": "Dommage [CHF]",
        "Stückpreis Werkzeug \\(P_1\\)": "Prix unitaire outil \\(P_1\\)",
        "Stückpreis Produkt \\(P_2\\)": "Prix unitaire produit \\(P_2\\)",
        "Produktionsrate \\(n\\) [Stück/Zeit]": "Taux de production \\(n\\) [pieces/temps]",
        "Abnahmerate der Produktionsqualität \\(q\\) (negativer Wert)": "Taux de baisse de la qualite de production \\(q\\) (valeur negative)",
        "### Gewichtungen der Zielgrössen (müssen zusammen 1 ergeben)": "### Ponderations des objectifs (la somme doit etre 1)",
        "Gewichtung Wirtschaftlichkeit \\(w_1\\)": "Ponderation rentabilite \\(w_1\\)",
        "Gewichtung Qualität \\(w_2\\)": "Ponderation qualite \\(w_2\\)",
        "Gewichtung Produktivität \\(w_3\\)": "Ponderation productivite \\(w_3\\)",
        "Die Summe der Gewichtungen darf nicht 0 sein!": "La somme des ponderations ne doit pas etre 0 !",
        "Wahrscheinlichkeitsverteilung der Augensumme von Würfeln": "Distribution de probabilite de la somme des des",
        "{n_samples} Würfe mit {n_dices} sechsseitigen Würfeln{sub_title_addition}": "{n_samples} lancers avec {n_dices} des a six faces{sub_title_addition}",
        " (Empirisch)": " (empirique)",
        " (Theoretisch)": " (theorique)",
        " (Empirisch vs. Theoretisch)": " (empirique vs theorique)",
        "Anzahl Beobachtungen": "Nombre d'observations",
        "Kumulative Beobachtungen": "Observations cumulees",
        "Kumulative Wahrscheinlichkeit": "Probabilite cumulee",
        "Augensumme": "Somme des des",
        "Box Plot mit einem dynamischen Punkt": "Box plot avec un point dynamique",
    },
}


def current_language() -> str:
    return st.session_state.get("language", "de")


def t(text: str) -> str:
    language = current_language()
    translations = TRANSLATIONS.get(language, {})
    if text in translations:
        return translations[text]

    stripped_text = text.lstrip("✅📊🔢🔍🧠🌳 ")
    if stripped_text in translations:
        return translations[stripped_text]

    if language == "en":
        if text.startswith("Boxplot für "):
            return text.replace("Boxplot für ", "Boxplot for ", 1)
        if text.startswith("Geglättete Daten "):
            return text.replace("Geglättete Daten ", "Smoothed Data ", 1).replace("Fenster", "window")
        if text.startswith("Die Wahrscheinlichkeit, dass A und B gemeinsam eintreten"):
            value = text.split("**")[1]
            return f"The probability that A and B occur together is **{value}**."
        if text.startswith("Die Wahrscheinlichkeit, dass A und B gemeinsam auftreten"):
            value = text.split("**")[1]
            return f"The probability that A and B occur together is **{value}**."
        if text.startswith("Die Gesamtwahrscheinlichkeit für A beträgt"):
            value = text.split("**")[1]
            return f"The total probability for A is **{value}**."
        if text.startswith("P(A ∩ B) berechnet:"):
            return text.replace("berechnet", "calculated")
        if text.startswith("Der neue Punkt wird der Klasse"):
            value = text.split("**")[1]
            return f"The new point is assigned to class **{value}**."
        if text.startswith("R² Score:") and "höher = besser" in text:
            return text.replace("höher = besser", "higher = better")
        if text.startswith("Gewichtungen wurden automatisch normalisiert:"):
            return text.replace("Gewichtungen wurden automatisch normalisiert:", "Weights were normalized automatically:")
        if text.startswith("Optimale Zeit t ="):
            return text.replace("Optimale Zeit", "Optimal time").replace("max. Zielwert", "max. target value")

    if language == "fr":
        if text.startswith("Boxplot für "):
            return text.replace("Boxplot für ", "Boxplot pour ", 1)
        if text.startswith("Geglättete Daten "):
            return text.replace("Geglättete Daten ", "Donnees lissees ", 1).replace("Fenster", "fenetre")
        if text.startswith("Die Wahrscheinlichkeit, dass A und B gemeinsam eintreten"):
            value = text.split("**")[1]
            return f"La probabilite que A et B se produisent ensemble est de **{value}**."
        if text.startswith("Die Wahrscheinlichkeit, dass A und B gemeinsam auftreten"):
            value = text.split("**")[1]
            return f"La probabilite que A et B se produisent ensemble est de **{value}**."
        if text.startswith("Die Gesamtwahrscheinlichkeit für A beträgt"):
            value = text.split("**")[1]
            return f"La probabilite totale de A est de **{value}**."
        if text.startswith("P(A ∩ B) berechnet:"):
            return text.replace("berechnet", "calcule")
        if text.startswith("Der neue Punkt wird der Klasse"):
            value = text.split("**")[1]
            return f"Le nouveau point est attribue a la classe **{value}**."
        if text.startswith("R² Score:") and "höher = besser" in text:
            return text.replace("höher = besser", "plus eleve = meilleur")
        if text.startswith("Gewichtungen wurden automatisch normalisiert:"):
            return text.replace("Gewichtungen wurden automatisch normalisiert:", "Les ponderations ont ete normalisees automatiquement :")
        if text.startswith("Optimale Zeit t ="):
            return text.replace("Optimale Zeit", "Temps optimal").replace("max. Zielwert", "valeur cible max.")

    return text


def select_language() -> None:
    language = st.sidebar.selectbox(
        t("Sprache"),
        options=list(LANGUAGES),
        format_func=lambda code: LANGUAGES[code],
        index=list(LANGUAGES).index(current_language()),
    )
    st.session_state["language"] = language


def translate_streamlit_text() -> None:
    if getattr(st, "_i18n_text_wrapped", False):
        return

    text_methods = [
        "title",
        "header",
        "subheader",
        "text",
        "write",
        "markdown",
        "button",
        "warning",
        "error",
        "success",
        "info",
    ]

    input_methods = [
        "slider",
        "selectbox",
        "multiselect",
        "radio",
        "number_input",
    ]

    def wrap_method(target, name):
        original = getattr(target, name)

        def wrapped(*args, **kwargs):
            if args and isinstance(args[0], str):
                args = (t(args[0]), *args[1:])
            if isinstance(kwargs.get("label"), str):
                kwargs["label"] = t(kwargs["label"])
            return original(*args, **kwargs)

        setattr(target, name, wrapped)

    for method in text_methods + input_methods:
        if hasattr(st, method):
            wrap_method(st, method)

    for method in ["header", "slider", "selectbox", "multiselect", "radio", "number_input"]:
        if hasattr(st.sidebar, method):
            wrap_method(st.sidebar, method)

    st._i18n_text_wrapped = True
