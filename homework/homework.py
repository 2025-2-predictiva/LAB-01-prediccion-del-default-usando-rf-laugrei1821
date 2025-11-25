# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".


import pandas as pd
import os
import numpy

# Cargar los datos

Data_Folder = "files/input"

train_df_raw = pd.read_csv(os.path.join(Data_Folder, "train_data.csv.zip"))
test_df_raw = pd.read_csv(os.path.join(Data_Folder, "test_data.csv.zip"))

# Copias

train_df = train_df_raw.copy()
test_df = test_df_raw.copy()

# Visualización Inicial de los datos

# a. Información general
train_df.info()
print(train_df.head())

test_df.info()
print(test_df.head())

# b. Información estadística

train_df.describe()
print(train_df.head())

test_df.describe()
print(test_df.head())


# Renombre la columna "default payment next month" a "default".


train_df = train_df.rename(columns={"default payment next month": "default"})

test_df = test_df.rename(columns={"default payment next month": "default"})


# Remueva la columna "ID".


train_df = train_df.drop(columns=["ID"])

test_df = test_df.drop(columns=["ID"])



# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".



train_df["EDUCATION"] = train_df["EDUCATION"].apply(lambda x: 4 if x == 0 or x > 4 else x)
train_df= train_df[train_df["MARRIAGE"] != 0]
train_df=train_df.dropna()



test_df["EDUCATION"] = test_df["EDUCATION"].apply(lambda x: 4 if x == 0 or x > 4 else x)
test_df= test_df[test_df["MARRIAGE"] != 0]
test_df=test_df.dropna()


train_df["EDUCATION"].value_counts()
test_df["EDUCATION"].value_counts()


# Visualización de los datos limpios


print("Training Data after cleaning")
print(train_df.head())
train_df.info()


print("Test Data after cleaning")
print(test_df.head())
test_df.info()



# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

y_train = train_df["default"]
y_test = test_df["default"]


X_train = train_df.drop(columns=["default"])
X_test = test_df.drop(columns=["default"])


print(X_train.head)

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import gzip
import os
import pickle
import numpy as np
import pandas as pd

# Definir las características categóricas y numéricas


categorical_features = [
    "SEX", "EDUCATION", "MARRIAGE",
]


numerical_features = [
    "LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
    "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "PAY_0", "PAY_2",
    "PAY_3", "PAY_4", "PAY_5", "PAY_6",
]

# Detección de correlación X respecto de Y: Calcular MI (Mutual Information) = Posible Data Leakage

# MI para numericas

X_train_num = X_train[numerical_features]

print("\nColumnas numéricas utilizadas para MI:")
print(X_train_num.columns.tolist())

mi = mutual_info_regression(X_train_num, y_train, random_state=42)

mi_series = pd.Series(mi, index=numerical_features)
mi_series = mi_series.sort_values(ascending=False)

print("\nMutual Information entre variables numéricas y la Y:")
print(mi_series)

print("\nVariables con MI sospechosamente alta (posible leakage):")
print(mi_series[mi_series > mi_series.quantile(0.90)])

#R/: Definitivamente ninguna.


# Detección de correlación entre las X: Calcular Correlación Pearson

# Pearson para numericas

X_train_num = X_train[numerical_features]

corr_matrix = X_train_num.corr()

corr_abs = corr_matrix.abs()

high_corr_pairs = [
    (corr_abs.index[i], corr_abs.columns[j], corr_abs.iloc[i, j])
    for i in range(len(corr_abs))
    for j in range(i + 1, len(corr_abs))
    if corr_abs.iloc[i, j] > 0.80
]

print("\n===== PARES CON CORRELACIÓN ALTA (> 0.80) =====")

if high_corr_pairs:
    for var1, var2, corr in high_corr_pairs:
        print(f"{var1}  —  {var2}:  {corr:.3f}")
else:
    print("No se encontraron pares altamente correlacionados.")

# R/: Alta correlación estre variables.
# A los RandomForest les da igual si hay multicolinealidad.
# No necesitas eliminar variables muy correlacionadas.

# Pearson para categoricas

X_train_cat = X_train[categorical_features]

corr_matrix = X_train_cat.corr()

corr_abs = corr_matrix.abs()

high_corr_pairs = [
    (corr_abs.index[i], corr_abs.columns[j], corr_abs.iloc[i, j])
    for i in range(len(corr_abs))
    for j in range(i + 1, len(corr_abs))
    if corr_abs.iloc[i, j] > 0.80
]

print("\n===== PARES CON CORRELACIÓN ALTA (> 0.80) =====")

if high_corr_pairs:
    for var1, var2, corr in high_corr_pairs:
        print(f"{var1}  —  {var2}:  {corr:.3f}")
else:
    print("No se encontraron pares altamente correlacionados.")

# R/: No hay correlación entre variables.

# Detección de outliers usando el método IQR

df = X_train.copy()

outlier_info = {}

for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    outliers_col = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
    
    outlier_info[col] = {
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "n_outliers": outliers_col.shape[0],
        "outliers": outliers_col[col].values
    }

outlier_info

# Ahora que sé que hay aoutliers, quiero ver cuáles son:

outlier_rows = {}

for col, info in outlier_info.items():
    lower = info["lower_limit"]
    upper = info["upper_limit"]

    mask = (df[col] < lower) | (df[col] > upper)
    
    outlier_rows[col] = df[mask]


for col, rows in outlier_rows.items():
    print(f"\n=== OUTLIERS EN {col} ===")
    print(f"Total filas: {len(rows)}")
    print(rows)

# R:/ RandomForest no se ve afectado por distancias o escalas. 

# Transforma las variables categoricas usando el método one-hot-encoding.

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough")

# Crear el pipeline completo


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("rf", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )



# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

param_grid = {
        "rf__n_estimators": [300],
        "rf__max_depth": [5,10,20, None],
        "rf__min_samples_split": [10],
        "rf__min_samples_leaf": [4],
        "rf__max_features": [25],  
    }


grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
    verbose=1  
)


grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

import gzip
import os
import pickle

model_dir = "files/models"
os.makedirs(model_dir, exist_ok=True)


model_filename = os.path.join(model_dir, "model.pkl.gz")


with gzip.open(model_filename, "wb") as f:
    pickle.dump(grid_search, f)

# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}

import json
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
)


Output_dir = "files/output"
Metrics_path = os.path.join(Output_dir, "metrics.json")


os.makedirs(Output_dir, exist_ok=True)


y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)




train_metrics = {
    "type": "metrics",
    "dataset": "train",
    "precision": precision_score(y_train, y_train_pred, average="binary", pos_label=1, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
    "recall": recall_score(y_train, y_train_pred, average="binary", pos_label=1, zero_division=0),
    "f1_score": f1_score(y_train, y_train_pred, average="binary", pos_label=1, zero_division=0),
}


test_metrics = {
    "type": "metrics",
    "dataset": "test",
    "precision": precision_score(y_test, y_test_pred, average="binary", pos_label=1, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "recall": recall_score(y_test, y_test_pred, average="binary", pos_label=1, zero_division=0),
    "f1_score": f1_score(y_test, y_test_pred, average="binary", pos_label=1, zero_division=0),
}


with open(Metrics_path, "w") as f:
    f.write(json.dumps(train_metrics) + "\n")
    f.write(json.dumps(test_metrics) + "\n")




# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}


from sklearn.metrics import confusion_matrix



cm_train = confusion_matrix(y_train, y_train_pred)


tn_train, fp_train, fn_train, tp_train = cm_train.ravel()



cm_train_dict = {
    "type": "cm_matrix",
    "dataset": "train",
    "true_0": {"predicted_0": int(tn_train), "predicted_1": int(fp_train)},
    "true_1": {"predicted_0": int(fn_train), "predicted_1": int(tp_train)}
}



cm_test = confusion_matrix(y_test, y_test_pred)
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()



cm_test_dict = {
    "type": "cm_matrix",
    "dataset": "test",
    "true_0": {"predicted_0": int(tn_test), "predicted_1": int(fp_test)},
    "true_1": {"predicted_0": int(fn_test), "predicted_1": int(tp_test)}
}


with open(Metrics_path, "a") as f:
    f.write(json.dumps(cm_train_dict) + "\n")
    f.write(json.dumps(cm_test_dict) + "\n")


print("\n--- Lab finished successfully ")

