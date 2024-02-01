# **Explorative Datenanalyse und fortgeschrittenes Feature Engineering: Entwicklung eines leistungsfähigen Diabetes-Vorhersagemodells!**


In diesem Diabetes-Projekt standen eine eingehende explorative Datenanalyse (EDA) sowie kreatives Feature-Engineering im Vordergrund, bevor das maschinelle Lernmodell entwickelt wurde. Hier ist eine Zusammenfassung der wichtigsten Schritte:

### Datenüberblick:
- Der Datensatz stammt von Pima-Indianerinnen aus einer Diabetesstudie in den USA.
- Enthält 768 Beobachtungen und 8 numerische unabhängige Variablen.
- Die Zielvariable ist "Outcome" mit 1 für positives und 0 für negatives Diabetesergebnis.
  
### Explorative Datenanalyse (EDA):
- Allgemeine Überprüfung der Datenform, Datentypen, erste Zeilen, letzte Zeilen und Quantile.
- Erfassung numerischer und kategorischer Variablen.
- Analyse kategorischer Variablen mit Countplots.
- Analyse numerischer Variablen mit Histogrammen und Deskriptivstatistiken.
- Untersuchung von Ausreißern und Anpassung der Ausreißerwerte.
- Analyse und Imputation von fehlenden Werten.
- Korrelationsmatrix und Heatmap zur Darstellung der Korrelationen zwischen Variablen.
  
### Feature Engineering:
- Schaffung neuer kategorischer Variablen basierend auf Altersgruppen, BMI-Stufen, Glukosewerten, Hautdicke und anderen Merkmalen.
- Umwandlung von Alters- und Glukosewerten in kategorische Variablen.
- Erstellung von neuen Merkmalen wie "NEW_INSULIN_SCORE", "NEW_GLUCOSE*INSULIN", usw.
- Label-Encoding für binäre kategoriale Variablen.
- One-Hot-Encoding für mehrkategoriale Variablen.
- Standardisierung der numerischen Variablen.
  
### Modellierung:
- Aufteilung der Daten in Trainings- und Testsets.
- Verwendung eines RandomForestClassifier-Modells für die Klassifikation.
- Evaluierung des Modells mit Metriken wie Genauigkeit, Recall, Präzision, F1-Score und AUC-ROC.
  
### Ergebnisse:
- Verbesserte Leistung des Modells nach Feature-Engineering und Datenpräparation.
- Verbesserte Genauigkeit (Accuracy: 0.79 im Vergleich zu 0.77 des Basis-Modells).
- Verbesserter Recall (Recall: 0.711 im Vergleich zu 0.706 des Basis-Modells).
- Verbesserte Präzision (Precision: 0.67 im Vergleich zu 0.59 des Basis-Modells).
- Verbesserter F1-Score (F1: 0.69 im Vergleich zu 0.64 des Basis-Modells).
- Verbesserte AUC-ROC (AUC: 0.77 im Vergleich zu 0.75 des Basis-Modells).
  
### Feature Importance:
- Visualisierung der wichtigsten Merkmale im Modell. Wichtige Merkmale können aus der Reihenfolge ihrer Bedeutung abgeleitet werden.
