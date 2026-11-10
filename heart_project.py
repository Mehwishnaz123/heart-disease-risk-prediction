# =====================================================================
#  heart_project.py
#  The SAME steps as my notebook (HEART_DISEASE.ipynb), written as functions.
#  The dashboard (app.py) imports this file and runs these functions,
#  so the app follows exactly the same process as the notebook:
#
#   1. Load data  ->  2. EDA charts  ->  3. Missing values  ->  4. Outliers
#   5. X / y + preprocessing pipeline  ->  6. Train 6 models (GridSearchCV)
#   7. Evaluate best model  ->  8. Predict a new patient
# =====================================================================

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:                                   # XGBoost is optional: if it is not installed,
    from xgboost import XGBClassifier  # the other 5 models still run
except ImportError:
    XGBClassifier = None
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

import os
FOLDER = os.path.dirname(os.path.abspath(__file__))           # folder of this file
DATA_PATH = os.path.join(FOLDER, "heart_disease_uci.csv")     # csv in the same folder
MODEL_PATH = os.path.join(FOLDER, "heart_model.pkl")          # model saved by notebook
PINK, YELLOW = "#E8458B", "#F5B700"                    # main colours of the project
PALETTE = ["#E8458B", "#F5B700", "#FF8FB1", "#FFD166", "#C2185B", "#FFB347"]  # chart colours


# ---------------------------------------------------------------------
# STEP 1: LOAD DATA  (notebook: "Load the dataset")
# ---------------------------------------------------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------
# STEP 2: EDA CHARTS  (notebook: "Exploratory Data Analysis")
# ---------------------------------------------------------------------
def age_histogram(df):
    # notebook: px.histogram(data_frame=df, x="age", color="sex")
    return px.histogram(df, x="age", color="sex", title="Age distribution by sex",
                        color_discrete_sequence=[PINK, YELLOW])


def sex_counts(df):
    # notebook: df['sex'].value_counts() and male / female percentage
    counts = df["sex"].value_counts()
    percent = (counts / counts.sum() * 100).round(2)
    return counts, percent


def dataset_by_sex(df):
    # notebook: px.histogram(df, x="dataset", color="sex", barmode="group")
    return px.histogram(df, x="dataset", color="sex", barmode="group",
                        title="Patients per hospital (dataset) by sex",
                        color_discrete_sequence=[PINK, YELLOW])


def age_stats_by_dataset(df):
    # notebook: mean, median and mode of age grouped by dataset
    return df.groupby("dataset")["age"].agg(
        mean="mean", median="median",
        mode=lambda s: ", ".join(str(m) for m in s.mode())).round(1)


def chest_pain_by(df, colour_col):
    # notebook: sns.countplot(x="cp", hue="sex") and hue="dataset"
    return px.histogram(df, x="cp", color=colour_col, barmode="group",
                        title=f"Chest pain type (cp) by {colour_col}",
                        color_discrete_sequence=PALETTE)


def trestbps_histogram(df):
    # notebook: sns.histplot(df['trestbps'], kde=True)
    return px.histogram(df, x="trestbps", nbins=40, title="Resting blood pressure (trestbps)",
                        color_discrete_sequence=[PINK])


# ---------------------------------------------------------------------
# STEP 3: MISSING VALUES  (notebook: "Dealing with missing values")
# ---------------------------------------------------------------------
def missing_percent(df):
    # notebook: (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
    return (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)


def fill_missing_values(df):
    df = df.copy()

    # trestbps -> IterativeImputer
    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputer.fit(df[["trestbps"]])
    df["trestbps"] = imputer.transform(df[["trestbps"]])

    # ca, oldpeak, thalch, chol, fbs, exang -> IterativeImputer (one column at a time)
    imputer_2 = IterativeImputer(max_iter=10, random_state=42)
    for col in ["ca", "oldpeak", "thalch", "chol", "fbs", "exang"]:
        df[col] = imputer_2.fit_transform(df[[col]]).ravel()

    # thal, slope, restecg -> mode (most frequent value)
    for col in ["thal", "slope", "restecg"]:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


# ---------------------------------------------------------------------
# STEP 4: OUTLIERS  (notebook: "Outliers")
# ---------------------------------------------------------------------
NUMERIC_COLS = ["oldpeak", "thalch", "chol", "trestbps", "age"]


def numeric_boxplots(df):
    # notebook: df[numeric_cols].melt(...) then px.box(...)
    df_box = df[NUMERIC_COLS].melt(var_name="column", value_name="value")
    fig = px.box(df_box, x="column", y="value", color="column",
                 title="Box plots of numeric columns", color_discrete_sequence=PALETTE)
    fig.update_layout(showlegend=False)
    return fig


def remove_impossible_rows(df):
    # notebook: remove the row where trestbps (blood pressure) is 0
    return df[df["trestbps"] != 0]


def clean_data(df):
    """Steps 3 + 4 together, in the same order as the notebook."""
    df = fill_missing_values(df)
    df = remove_impossible_rows(df)
    return df


# ---------------------------------------------------------------------
# STEP 5: X, y AND PREPROCESSING PIPELINE  (notebook: "Machine Learning")
# ---------------------------------------------------------------------
def make_x_y(df):
    data = df.copy()
    y = (data["num"] > 0).astype(int)                       # 0 = healthy, 1 = disease
    X = data.drop(columns=["num", "id", "dataset"], errors="ignore")
    for c in ["chol", "trestbps"]:                           # fake zeros -> missing
        X[c] = X[c].replace(0, np.nan)
    return X, y


def make_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    X[cat_cols] = X[cat_cols].astype(str)
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return prep, X


def split_data(X, y):
    # 80% train, 20% test, same class ratio in both (stratify)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------------------------------------------------------
# STEP 6: TRAIN AND TUNE 6 MODELS  (notebook: GridSearchCV loop)
# ---------------------------------------------------------------------
SEARCH = {
    "Logistic Regression": (LogisticRegression(max_iter=2000),
        {"m__C": [0.01, 0.1, 1, 10]}),
    "SVC": (SVC(probability=True, random_state=42),
        {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.01, 0.1]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
        {"m__n_estimators": [200, 400], "m__max_depth": [4, 6, 10, None],
         "m__min_samples_leaf": [1, 3, 5]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
        {"m__n_estimators": [100, 200], "m__learning_rate": [0.03, 0.05, 0.1],
         "m__max_depth": [2, 3]}),
    "KNN": (KNeighborsClassifier(), {"m__n_neighbors": [5, 7, 9, 11, 15]}),
}
if XGBClassifier is not None:
    SEARCH["XGBoost"] = (XGBClassifier(eval_metric="logloss", random_state=42),
        {"m__n_estimators": [100, 200], "m__max_depth": [2, 3, 4],
         "m__learning_rate": [0.03, 0.05, 0.1]})


def train_models(prep, X_train, y_train):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results, fitted = [], {}
    for name, (est, grid) in SEARCH.items():
        pipe = Pipeline([("prep", prep), ("m", est)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1).fit(X_train, y_train)
        fitted[name] = gs
        results.append([name, round(gs.best_score_, 3), gs.best_params_])
    results = pd.DataFrame(results, columns=["Model", "CV F1", "Best settings"])
    best_name = max(fitted, key=lambda n: fitted[n].best_score_)
    best_model = fitted[best_name].best_estimator_
    return results, best_name, best_model


# ---------------------------------------------------------------------
# STEP 7: EVALUATE THE BEST MODEL ON THE TEST SET
# ---------------------------------------------------------------------
def evaluate(best_model, X_test, y_test):
    pred = best_model.predict(X_test)
    prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred),
        "report": pd.DataFrame(classification_report(y_test, pred, output_dict=True)).T,
        "roc_auc": roc_auc_score(y_test, prob),
        "roc": pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr}),
    }


def feature_importance(best_model, top=12):
    # notebook: rf.feature_importances_ with names from the preprocessor
    rf = best_model.named_steps["m"]
    names = best_model.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    return pd.Series(rf.feature_importances_, index=names).sort_values().tail(top)


# ---------------------------------------------------------------------
# STEP 8: PREDICT A NEW PATIENT  (notebook: new_patient + predict_proba)
# ---------------------------------------------------------------------
def predict_patient(model, patient: dict):
    new_patient = pd.DataFrame([patient])
    pred = model.predict(new_patient)[0]
    prob = model.predict_proba(new_patient)[0][1] * 100
    return ("Heart Disease Risk" if pred == 1 else "Healthy"), prob


# ---------------------------------------------------------------------
# WHOLE PROJECT IN ONE CALL (used by the app, runs steps 1 -> 7)
# ---------------------------------------------------------------------
def run_full_project(path=DATA_PATH):
    raw = load_data(path)
    df = clean_data(raw)
    X, y = make_x_y(df)
    prep, X = make_preprocessor(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    results, best_name, best_model = train_models(prep, X_train, y_train)
    scores = evaluate(best_model, X_test, y_test)
    return {"raw": raw, "clean": df, "X": X, "y": y,
            "sizes": (len(X_train), len(X_test)),
            "results": results, "best_name": best_name, "best_model": best_model,
            "scores": scores}
