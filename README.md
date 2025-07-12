1 Introduction
Motivation

    >Den Struggle beim Einkaufen von neuen Klamotten, oder sogar beim entdecken von neuen Styles, kennt ein Jeder von uns.
    >Mit unserer Anwendung wollen wir ein Tool entwickeln, welches helfen soll genauere Vorstellungen und Ansprüche, in diese Richtung, formen zu können.

Research question

    >Lässt sich ein Zusammenhang zwischen Kleidungsstücken herstellen, die uns gefallen, und ist es Möglich relativ simple Vorhersagen zu treffen ob uns ähnliche Klamotten gefallen?

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
    Run the Streamlit app for the lite version!
    Run it local for the full experience!
    
2 Related Work

    >Artikel "Clothing similarity computation based on TLAC" (https://www.emerald.com/insight/content/doi/10.1108/09556221211232856/full/html)
    >Git Repo https://github.com/GaParmar/img2img-turbo
    

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

    >Das Ziel ist es, ein Modell zu entwickeln, das nach einer bestimmten Anzahl an "Likes" und "Dislikes" von Kleidungsstücken eine Vorhersage trifft: Es sollen Kleidungsstücke vorgeschlagen werden, die dem individuellen Geschmack der Nutzer*innen entsprechen. Dabei wird ein einfaches, aber effektives Klassifikations- oder Empfehlungssystem verwendet. Eine Evaluation erfolgt auf Basis der Vorhersagegenauigkeit sowie der Nutzerzufriedenheit (z. B. über Testdurchläufe in der App).
    

4 Results

    >Modell ist in der Lage anhand von selbstausgewählten Bild Konzepten, neue Kleidungsstücke zu kreiren.
    >Verbesserungen der App können durch einen anderen Datensatz erzielt werden, sodass eindeutiger Bilder dem User angezeigt werden.

    >Akuteller Stand (Woche 23.06): Streamlit-Anwendung mit Datensatz integriert, Model zur Vorhersage um welche Klamotten es sich handelt, Ersten Prototyp zur Vorhersage von welchen Klamotten einen gefallen könnte

5 Discussion
Limitationen:

    >Der MNIST Datensatz ist gut zur Visuallisierung des Prinzipes der Anwendung gedacht, jedoch wäre für eine bessere UX ein Datensatz mit einer höheren Bildqualität gut.
    >Ressourcen- und Zeitlimits wurden bei der Umsetzung des Projektes erreicht.

Ethische Überlegung:

    >Datenschutz durch das Sicherstellen von nicht Verarbeitung von Daten dritter.

Transparenz:

    >Für die Bilderstellung wurde einiges an Arbeit von dem bereits genannten Repositorys "img2img-turbo" genutzt.
    >Feedback und konkrete Verbesserungsvorschläge sind jederzeit erwünscht.

6 Conclusion

    >Eine Hilfe zur "Style-Erweiterung" ist durch die App gegeben.
    >Die Anwendung kann noch verbessert werden, durch:
    
            Datensatzänderung
            Ladezeitenoptimierung
            Allgemeine Codevereinfacherung
        
    >Die Streamlitapp bietet ebenfalls ein verbessertes Erlebnis der "Stylefindung" durch Gameifizierung
    >Die Ergebnisse und Modelle sind offen für die Community und können weiterentwickelt werden
