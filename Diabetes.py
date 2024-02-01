###################################  Diabetes   ###########################



# Es wird angestrebt, ein maschinelles Lernmodell zu entwickeln, das die Fähigkeit besitzt, vorherzusagen, ob Personen
# an Diabetes erkrankt sind, wenn ihre charakteristischen Merkmale bekannt sind. Vor der eigentlichen Modellentwicklung
# ist es erforderlich, die notwendigen Schritte der Datenanalyse (EDA) und Feature-Engineering zu unternehmen. Der
# zugrundeliegende Datensatz ist Teil einer umfassenden Sammlung der National Institutes of Diabetes-Digestive-Kidney
# Diseases in den USA. Die Daten wurden im Rahmen einer Diabetesstudie an Pima-Indianerinnen im Alter von 21 Jahren und
# älter erhoben, die in Phoenix, der fünftgrößten Stadt des Bundesstaates Arizona, USA, leben. Der Datensatz umfasst 768
# Beobachtungen und beinhaltet 8 numerische unabhängige Variablen. Die Zielvariable ist als "outcome" festgelegt, wobei
# 1 ein positives Ergebnis bei einem Diabetes-Test anzeigt und 0 ein negatives Ergebnis bedeutet.


# Pregnancies: Anzahl der Schwangerschaften
# Glucose: Glukose
# BloodPressure: Blutdruck (Diastolisch). Hier bezieht sich "Diastolisch" auf den niedrigeren Druck in den Blutgefäßen
#                während der Entspannungsphase des Herzzyklus.
# SkinThickness: Hautdicke
# Insulin: Insulin
# BMI: Body Mass Index
# DiabetesPedigreeFunction: Eine Funktion, die aufgrund genetischer Merkmale und familiärer Verbindungen die
#                           Wahrscheinlichkeit für die Entwicklung von Diabetes bei Personen unserer Abstammung berechnet.
# Age: Alter (Jahre)
# Outcome: Information darüber, ob eine Person Diabetes hat oder nicht. Hat die Krankheit (1) oder nicht (0)



#######################################
# Importieren von Bibliotheken und Einstellungen (Settings)
######################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


####################################
# Datentransfer
###################################
def load_data(dataframe):
    return pd.read_csv(dataframe)


diabetes = load_data("datasets/diabetes-230423-212053.csv")
df = diabetes.copy()
df.head()


####################################################################################
############################### Explorative Datenanalyse (EDA)  ####################
####################################################################################

# In diesem Schritt analysiert man die vorhandenen Daten, um ein besseres Verständnis für ihre Struktur, Verteilungen,
# Beziehungen und mögliche Muster zu gewinnen. Ziel ist es, Einblicke in die Daten zu gewinnen, bevor man Modelle
# erstellt.

##################################
# Allgemeines Bild
##################################


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df, head=5)


####################################################
# Erfassung numerischer und kategorischer Variablen
####################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Liefert die Bezeichnungen der kategorialen, numerischen sowie kardinalen, aber dennoch kategorialen Variablen im
    Datensatz.
    Notiz: Hierbei sind auch numerisch erscheinende kategoriale Variablen inbegriffen.

    Parameters
    ------
        dataframe: Dataframe
                DataFrame, aus dem die Variablennamen extrahiert werden sollen.
        cat_th: int, optional
                Grenzwert/ Schwellenwert für Variablen, die numerisch, jedoch kategorisch sind.
        car_th: int, optional
                Grenzwert für kategorische, jedoch kardinale Variablen.

    Returns
    ------
        cat_cols: list
                Liste der kategorialen Variablen
        num_cols: list
                Liste der numerischen Variablen
        cat_but_car: list
                Liste der kardinalen Variablen mit kategorischem Erscheinungsbild

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = Gesamtanzahl der Variablen
        num_but_cat ist in cat_cols enthalten.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car



#################################################
# Analyse kategorischer Variablen
################################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# Die Funktion kann bei Bedarf auf diese Weise aktualisiert werden.
def cat_summary_l(dataframe, cat_cols, plot=False):
    for col_name in cat_cols:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()

cat_summary_l(df, cat_cols)

for col in cat_cols:
    cat_summary(df, col)


##############################################
# Analyse numerischer Variablen
##############################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)


################################################################
# Analyse numerischer Variablen bezogen auf das Ziel (TARGET)
################################################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)



############################################################
# Analyse von Ausreißern
###########################################################


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Diese Funktion identifiziert die Obergrenze und Untergrenze für Ausreißer in der angegebenen Spalte.

    Parameters:
        dataframe: der Name des DataFrames
        col_name: Der Name der Spalte
        q1: Untergrenze
        q3: Obergrenze

    Returns:
        Untergrenze für Spaltenwerte, Obergrenze für Spaltenwerte
    """

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Diese Funktion überwacht, ob es in der entsprechenden Spalte Ausreißer gibt, basierend auf den vorgegebenen unteren
    und oberen Grenzwerten für Ausreißer.

    Parameters:
        dataframe: Der Name des DataFrames
        col_name: Der Name der Spalte

    Returns:
        True oder False
    """

    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in df.columns:
    print(col, check_outlier(df, col))




def grab_outliers(dataframe, col_name, index = False):
    """
    Diese Funktion gibt die ersten fünf Zeilen der Spalten mit Ausreißern zurück.

    Parameters:
        dataframe: der Name des DataFrames
        col_name: der Name der Spalte
        index : Parameter, um festzulegen, ob die Indizes der Ausreißer gespeichert werden sollen.

    Returns
    -------

    """
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


for col in num_cols:
    grab_outliers(df, col)






def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """
    Diese Funktion modifiziert die Ausreißer in den Spalten basierend auf den unteren und oberen Grenzwerten.

    Parameters:
        dataframe: der Name des DataFrames
        variable: Variable/ Spalte
        q1: Untergrenze
        q3: Obergrenze

    Returns:

    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Analyse von Ausreißern und Imputationsprozess
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))



########################################################
# Analyse von fehlenden Werten (missing values)
#######################################################

df.isnull().sum()  # OUT: False, ABER! wir wissen, dass die fehlende Werte (NaN) durch Null ersetzt worden sind, so dass
# wir von Missing Values ausgehen können, auch wenn False angezeigt wird. Denn:

# Bei einem Individuum können die Werte der Variablen außer 'Pregnancies' und 'Outcome' nicht 0 sein. Daher ist eine
# Entscheidung bezüglich dieser Werte erforderlich. Es könnte erwogen werden, den Wert 0 durch NaN (Not a Number) zu
# ersetzen.

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns


# In den Beobachtungseinheiten (Gözlem birim) haben wir alle Variablen mit einem Wert von 0 überprüft und die
# entsprechenden Beobachtungswerte durch NaN ersetzt.
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.NaN, df[col])


# Kontrolle:
df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    """
    Diese Funktion identifiziert Spalten mit fehlenden Werten (NA). Sie gibt die Anzahl der NA-Werte in diesen Spalten
    sowie deren Anteil an den Gesamtdaten an. Bei Bedarf gibt sie auch die Namen der Spalten mit NA-Werten aus.

    Parameters:
        dataframe: der Name des DataFrames
        na_name:  der Name der Spalte mit NA-Werten

    Returns:

    """

    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)



# Fehlende Werte auffüllen
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()

##################################
# Korrelation
#################################

# Korelation beschreibt in der Wahrscheinlichkeitstheorie und Statistik die Richtung und Stärke der linearen Beziehung
# zwischen zwei zufälligen Variablen.

df.corr()

# Korrelationsmatrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

####################################
# Visualisierung
###################################

diabetic = df[df.Outcome == 1]
healthy = df[df.Outcome == 0]

plt.scatter(healthy.Age, healthy.Insulin, color="green", label="Healthy", alpha = 0.4)
plt.scatter(diabetic.Age, diabetic.Insulin, color="red", label="Diabetic", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Insulin")
plt.legend()
plt.show()


##################################################################################
################################## Featuring Engineering #########################
##################################################################################


# In diesem Teil beabsichtige ich, die notwendigen Schritte der Datenanalyse und Merkmalsentwicklung vor der
# Modellentwicklung durchzuführen.

####################################
# Installation des Basismodells
####################################


y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.77
# Recall: 0.706           # Die Erfolgsquote bei der Vorhersage der positiven Klasse.
# Precision: 0.59         # Der Erfolg bei der Vorhersage der positiven Klasse.
# F1: 0.64
# Auc: 0.75


#######################################################
# Merkmalextraktion (Feature Extraction)
#######################################################

# Neue Variablen erstellen:

# 1:
df["Age"].unique()

# Altersgruppe
df.loc[(df["Age"] >= 20) & (df["Age"] < 30), "NEW_AGE_CAT"] = "young"
df.loc[(df['Age'] >= 30) & (df['Age'] < 46), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 46), 'NEW_AGE_CAT'] = 'senior'

# 2:
df["BMI"].unique()
# BMI-Stufe
df.loc[(df["BMI"] < 18.5), "NEW_BMI_CAT"] = "underweight"
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 24.9), 'NEW_BMI_CAT'] = 'healty_weight'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 29.9), 'NEW_BMI_CAT'] = 'overweight'
df.loc[(df['BMI'] >= 30), 'NEW_BMI_CAT'] = 'obese'

# äquivalent:
# df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])

df.groupby(["NEW_BMI_CAT", "NEW_AGE_CAT"]).agg({"Outcome": ["count", "mean"]})


# 3:
# Umwandlung des Glukosewerts in eine kategorische Variable
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# 4:
df["NEW_SKINTHICKNESS_LEVEL"] = pd.cut(df["SKINTHICKNESS"], bins=[0, 25, 30, 35, 40, df["SKINTHICKNESS"].max()], labels = ["E", "D", "C", "B", "A"])


# 5:
# Durch die Berücksichtigung von Alter und Body Mass Index (BMI) wurden drei Kategorien für die Erstellung einer
# kategorischen Variable identifiziert.
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"


# 6:
# Die Erstellung einer kategorischen Variable unter Berücksichtigung von Alter und Glukosewerten.
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


# 7:
# Erstellung einer kategorischen Variable basierend auf dem Insulinwert.
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)


# 8:
df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

# 9
df["NEW_GLUCO_PREGNANCIES"] = df["GLUCOSE"] * df["PREGNANCIES"]

# 10
df["NEW_GLUCO_AGE"] = df["GLUCOSE"] / df["AGE"]

# 11:
# Alter in Bezug auf Schwangerschaft
df["PREGNANCIES_AGE_RATE"] = df["PREGNANCIES"] / df["AGE"]


# 12:
# Achtung: Beachte die Werte, die Null sind!
df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]
#df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * (1+ df["Pregnancies"])

# Erklärung zur Erhöhung um eins:
# Sie dient dazu, den Umgang mit Null-Werten zu verbessern oder unerwünschte Ergebnisse zu verhindern. Wenn die Spalte
# "Pregnancies" Nullen enthält und direkt mit "Glucose" multipliziert wird, würde das Ergebnis ebenfalls Null sein,
# unabhängig von den Werten in der "Glucose"-Spalte.
# Durch das Hinzufügen von eins (1 + df["Pregnancies"]) wird sichergestellt, dass selbst wenn "Pregnancies" Null ist,
# das Ergebnis der Multiplikation mindestens eins ist. Dies trägt in diesem Kontext dazu bei, unerwünschte Effekte zu
# vermeiden, die durch die Multiplikation mit Null entstehen könnten.




# Wenn erwünscht: Schriftvergrößerung der Spalten
#df.columns = [col.upper() for col in df.columns]


############################
# ENCODING
###########################

# Die Klassifizierung nach Variablentypen
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# One-Hot Encoding durchführen
# die Aktualisierung der cat_cols Liste
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()


# NEW_GLUCOSE -> NEW_GLUCOSE_predi, NEW_GLUCOSE_normal, NEW_GLUCOSE_diab
# predi               1                      0              0
# normal              0                      1              0
# diab                0                      0              1




##################################
# Standardisierung
##################################

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape


df.head()

##################################
# Modellierung
##################################

y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.79
# Recall: 0.711
# Precision: 0.67
# F1: 0.69
# Auc: 0.77

# Base Model
# Accuracy: 0.77
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75



##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)


