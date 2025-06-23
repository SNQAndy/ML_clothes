1 Introduction
Motivation

    >Den Struggle beim Einkaufen von neuen Klamotten, oder sogar beim entdecken von neuen Styles, kennt ein Jeder von uns.
    >Mit unserer Anwendung wollen wir ein Tool entwickeln, welches helfen soll genauere Vorstellungen und Ansprüche, in diese Richtung, formen zu können.

Research question

    >Lässt sich ein Zusammenhang zwwischen Kleidungsstücken herstellen, die uns gefallen, und ist es Möglich relativ simple vorhersagen zu treffen ob uns ähnliche Klamotten gefallen?

Goal

    >Entwicklung eines Modells zum klassifizieren von Kleidungsstücken.
    >Entwicklung einer Web-App die ähnlich wie eine Datingapp funktioniert, bloß für Klamotten.

Target audience

    >Personen jeden Alters, die sich für Kleidung interessieren.
    >Entwickler, die an Ähnlichkeitsvorhersagen interessiert sind.

How to use this repository

    Clone the repository
    Install the required packages via pip install -r requirements.txt
    Run the Jupyter notebooks to explore the data
    Run the Python scripts to train the models
    Run the Streamlit app for the full experience!
2 Related Work

    >Artikel "Clothing similarity computation based on TLAC" (https://www.emerald.com/insight/content/doi/10.1108/09556221211232856/full/html)
    

3 Methodology    
3.1 General Methodology
    
    >Suche auf Github und Gitlab nach ähnlichen Projekten
    >Suche nach Datensatz
    >Vertraut machen mit dem Datensatz
    >Wechsel des Datensatzes
    >Entwwicklung des Streamlit Grundgerüsts
    
3.2 Data Understanding and Preparation

    >1.Datensatz von kaggle, welcher aber zu klein war (https://www.kaggle.com/datasets/gustavofadel/clothes-dataset)
        >Kategorisieren der Kleidungsstückarten
        >Stückanzahl Ausgabe jeder Klasse
        
    >2.Datensatz MNIST von tensorflow Ein größerer Datensatz mit 60.000 Trainings Daten und 10.000 Testdaten als Beispiele  (https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file)
        >Kategorisieren der Kleidungsstückarten
        >Prediction von Kleidungsarten
    

3.3 Modelling and Evaluation

    >Das Ziel ist es, ein Modell zu entwickeln, das nach einer bestimmten Anzahl an "Likes" und "Dislikes" von Kleidungsstücken eine Vorhersage trifft: Es sollen Kleidungsstücke vorgeschlagen werden, die dem individuellen Geschmack der Nutzer*innen entsprechen. Dabei wird ein einfaches,         aber effektives Klassifikations- oder Empfehlungssystem verwendet. Eine Evaluation erfolgt auf Basis der Vorhersagegenauigkeit sowie der Nutzerzufriedenheit (z. B. über Testdurchläufe in der App).
    

4. Results

Akuteller Stand (Woche 23.06): Streamlit-Anwendung mit Datensatz integriert, Model zur Vorhersage um welche Klamotten es sich handelt, Ersten Prototyp zur Vorhersage von welchen Klamotten einen gefallen könnte

5 Discussion
Aktueller Probleme: 
    Swipe Funktion muss noch integriert werden, 
    Prototyp muss in die Website mit eingebunden werden,
    (aktuellen Datensatz in einen Farbigen Datensatz ändern und nach Geschlechtern trennen)... muss noch angesprochen werden
