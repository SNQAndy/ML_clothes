Fashionswipe
# 1. Introduction
Motivation

    Den Struggle beim Online-Shoppen kennt fast jeder: Unüberschaubare Auswahl, unklare Passformen und ständig wechselnde Trends machen Stilfindung schwer.
    Unser Ziel ist es, ein Tool zu entwickeln, das durch Machine Learning individuelle Stilpräferenzen erkennt und so beim Finden und Auswählen neuer Kleidung unterstützt.

Research Question

    Gibt es einen Zusammenhang zwischen den Kleidungsstücken, die wir mögen?
    Können wir auf Basis weniger Interaktionen (Likes/Dislikes) sinnvolle Vorhersagen treffen, welche anderen Kleidungsstücke uns gefallen würden?

Goal

        Entwicklung eines Klassifikations- und Empfehlungssystems für Kleidung

        Realisierung einer Web-App mit einer interaktiven Swipe-Mechanik (ähnlich einer Dating-App) zur Nutzerprofilierung

        Integration eines KI-generierten Outfit-Vorschlagssystems (Stable Diffusion)

Target Audience

        Modeinteressierte aller Altersgruppen

        Entwickler*innen, die sich für Recommendation Engines und Bildklassifikation interessieren

        Personen, die Entscheidungen bei Online-Käufen effizienter gestalten wollen

How to use this repository

1. Repository klonen
git clone https://github.com/SNQAndy/ML_clothes.git

2. Abhängigkeiten installieren
pip install -r requirements.txt

3. Daten explorieren & Modell trainieren
jupyter notebook

4. Streamlit-App starten
streamlit run fashion_swipe_cloud.py

5. Für eine lokale Erfahrung:
python fashion_swipe_cloud.py

# 2. Related Work

        Clothing similarity computation based on TLAC
        (Artikel-Link: https://www.emerald.com/insight/content/doi/10.1108/09556221211232856/full/html)

        img2img-turbo
        (GitHub-Repo: https://github.com/GaParmar/img2img-turbo)

# 3. Methodology
# 3.1 General Methodology

        Recherche nach bestehenden Projekten auf GitHub/GitLab

        Auswahl und Validierung eines geeigneten Datensatzes

        Entwicklung eines Grundgerüsts in Streamlit

        Erweiterung um Generierung und interaktive Elemente

# 3.2 Data Understanding and Preparation

        Kaggle Dataset (zu klein):
        Gustavo Fadel Dataset
        → Erste Kategorisierung und Visualisierung

        Fashion-MNIST (verwendet):
        Zalando Research
        → 70.000 Graustufenbilder, 10 Kategorien
        → 60.000 Trainingsbilder und 10.000 Testbilder
        → Skalierung (28×28 → 280×280), Normalisierung & Klartext-Label
        → Performance-Optimierung mit Lazy Loading und RAM-schonender Architektur

# 3.3 Modelling and Evaluation

        Klassifikationsmodell: ConvNet mit 4 Conv-Blöcken, 2 FC-Layern, Adam-Optimizer

        Generative Modelle: Stable Diffusion XL Turbo, RealVIS, SDXL mit Refiner (Zuerst PixToPix Turbo, wurde aber ausgeschlossen da das Modell nicht Zielführend war)
        
        Huggingface Api für leistungstärkere Modelle, vor allem bei RealVIS da es zu groß für Streamlit ist
        
        Fallback Strategie falls ein Modell ausfällt wird zum nächsten Modell gewechselt und sollte die API ausfallen, wird auf das lokal integrierte Modell Stable Diffusion XL gewechselt für eine größere Robustheit
        
        Huggingface Api für leistungstärkere Modelle, vor allem bei RealVIS, da es zu groß für Streamlit ist

        Ensemble-Strategie + Adaptive Inference

        Evaluation anhand:

            Vorhersagegenauigkeit (Style-Prediction)

            User Experience (Responsivität, Bildqualität, Ladezeiten)

# 4. Results

        Klassifikator erkennt grundlegende Stilvorlieben mit hoher Genauigkeit

        Style-Generator (Stable Diffusion) erstellt Outfits anhand Nutzerpräferenzen

        Ladezeiten der Bilder: < 0.5 Sek, RAM-Verbrauch: Ø 3.2 GB

        Erste funktionale MVP-App mit Like-Dislike-Funktion, Style-Vorschlägen & Galerie

        Prototyp unterstützt bereits einfache Style-Profilierung

# 5. Discussion
Limitationen

        Ausgangsdaten (Fashion-MNIST) haben geringe Auflösung

        Farbige, realistischere Bilder wären für die UX ideal

        Komplexität der Modelle begrenzt durch lokale Rechenleistung

Ethische Überlegungen

        Datenschutz: Keine Verarbeitung personenbezogener Daten

        Bildgenerierung basiert auf offenen Modellen → Transparenz über verwendete Quellen

Transparenz

        Weitgehende Nutzung und Anpassung bestehender Projekte (z. B. img2img-turbo)

        Feedback willkommen – wir sehen das Projekt als Open-Source-Lernplattform

# 6. Conclusion

        Fashionswipe unterstützt Nutzer*innen spielerisch bei der Erweiterung ihres Stils

        Durch Gameifizierung (Liken/Disliken, Generieren, Galerie) entsteht ein neuartiges UX-Konzept

        Potenzial für weitere Entwicklung:

            Einbindung realistischerer, farbiger Datensätze

            Integration von Live-Shop-APIs (Preis, Verfügbarkeit)

            Optimierung von Ladezeiten & Infrastruktur

        Der Quellcode ist offen und modular – ideal für Community-Collaboration

