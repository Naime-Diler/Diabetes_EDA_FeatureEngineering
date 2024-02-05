# **Explorative Datenanalyse und fortgeschrittenes Feature Engineering!** 
## **Übersicht des Diabetesprädiktionsprojekts!**


In diesem Diabetes-Projekt standen eine eingehende explorative Datenanalyse (EDA) sowie kreatives Feature-Engineering im Vordergrund, bevor das maschinelle Lernmodell entwickelt wurde. Hier ist eine Zusammenfassung der wichtigsten Schritte:

### Projektziel:
Es wird angestrebt, ein maschinelles Lernmodell zu entwickeln, das die Fähigkeit besitzt, vorherzusagen, ob Personen an Diabetes erkrankt sind, wenn ihre charakteristischen Merkmale bekannt sind.

### Datensatz:
Der zugrundeliegende Datensatz stammt von den National Institutes of Diabetes-Digestive-Kidney Diseases in den USA. Die Daten wurden im Rahmen einer Diabetesstudie an Pima-Indianerinnen im Alter von 21 Jahren und älter erhoben. Der Datensatz umfasst 768 Beobachtungen und beinhaltet 8 numerische unabhängige Variablen. Die Zielvariable ist als "Outcome" festgelegt, wobei 1 ein positives Ergebnis bei einem Diabetes-Test anzeigt und 0 ein negatives Ergebnis bedeutet.

### Schritte im Projekt:
### 1.	Explorative Datenanalyse (EDA):
•	Allgemeine Übersicht über den Datensatz (Form, Datentypen, erste Zeilen)

•	Erfassung numerischer und kategorischer Variablen

•	Analyse kategorischer Variablen

•	Analyse numerischer Variablen

•	Analyse von Ausreißern

•	Analyse von fehlenden Werten

•	Korrelationsanalyse


### 2.	Feature Engineering:
•	Neue Merkmale basierend auf bestehenden Variablen erstellt, z. B. Altersgruppen, BMI-Stufen, etc

•	Transformation von numerischen Werten in kategorische Variablen

•	Schaffung neuer, aussagekräftiger Merkmale wie Alter in Bezug auf Schwangerschaften


### 3.	Datenpräparation:
•	Kodierung von kategorischen Variablen mittels Label-Encoding und One-Hot-Encoding

•	Standardisierung numerischer Variablen

### 4.	Modellierung:
•	Aufteilung des Datensatzes in Trainings- und Testsets

•	Anwendung eines Random Forest Klassifikationsmodells

### 5.	Modellbewertung:
•	Bewertung des Modells anhand verschiedener Metriken wie Accuracy, Recall, Precision, F1-Score und AUC

### 6.	Feature Importance:
•	Analyse der wichtigsten Merkmale für die Modellentscheidungen

Das entwickelte Modell zeigt eine Verbesserung gegenüber dem Basismodell, insbesondere in den Metriken Accuracy, Recall und Precision. Feature Engineering trug dazu bei, aussagekräftige Merkmale zu schaffen und die Modellleistung zu steigern.

  

