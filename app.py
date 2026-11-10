# =====================================================================
#  HEART DISEASE RISK PREDICTION - Interactive Dashboard (Streamlit)
#  Mehwish Naz | BS Computer Science | University of Larkano
#
#  app.py only SHOWS the project. All project code (loading, cleaning,
#  training, evaluation, prediction) is in heart_project.py, which follows
#  the same steps as my notebook HEART_DISEASE.ipynb.
#  Every page has a "📜 Code" box that shows the exact code being run.
#
#  RUN:   pip install -r requirements.txt
#         streamlit run app.py
# =====================================================================

import os
import inspect                    # lets the app display the code of a function
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import heart_project as hp        # <- my project code (same steps as notebook)

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="🫀", layout="wide")
PINK, YELLOW, PALETTE = hp.PINK, hp.YELLOW, hp.PALETTE

# Every plotly chart in the app gets a clean white style and my pink/yellow colours
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

# ---------- CSS: only decides how things LOOK (colours, round corners, shadows) ----------
st.markdown("""
<style>
/* page background: very light pink to light yellow */
.stApp {background: linear-gradient(135deg, #FFF5FA 0%, #FFFDF3 100%);}

/* sidebar: soft pink to soft yellow */
[data-testid="stSidebar"] {background: linear-gradient(180deg, #FFE1EE 0%, #FFF4C7 100%);}
[data-testid="stSidebar"] * {color: #5A1E3E;}

/* big colourful banner at the top of every page */
.hero {background: linear-gradient(120deg, #E8458B 0%, #FF8FB1 45%, #F5B700 100%);
       border-radius: 22px; padding: 26px 30px; margin-bottom: 18px;
       box-shadow: 0 8px 24px rgba(232, 69, 139, .25);}
.hero h1 {color: white; margin: 0; font-size: 38px;}
.hero p {color: #FFF8E1; margin: 6px 0 0 0; font-size: 17px;}
.beat {display: inline-block; animation: beat 1.2s infinite;}
@keyframes beat {0%,100% {transform: scale(1);} 25% {transform: scale(1.25);} 40% {transform: scale(1);}}

/* number cards: pink and yellow take turns */
.card {border-radius: 18px; padding: 18px 20px; text-align: center;
       box-shadow: 0 4px 14px rgba(232, 69, 139, .15); transition: transform .2s;}
.card:hover {transform: translateY(-4px);}
.card h2 {margin: 0; font-size: 32px;}
.card p {margin: 4px 0 0 0; font-size: 14px; color: #6B4A5C;}
.pink   {background: #FFE3EF;} .pink h2   {color: #C2185B;}
.yellow {background: #FFF3C4;} .yellow h2 {color: #B07D00;}

/* workflow steps */
.step {border-radius: 16px; padding: 14px 8px; text-align: center; height: 100%;
       background: white; border: 2px solid #FFC2DA; font-size: 14px;}
.step b {color: #E8458B; font-size: 15px;}

/* result box on the prediction page */
.result {border-radius: 20px; padding: 24px; color: white; text-align: center;
         box-shadow: 0 8px 24px rgba(0,0,0,.12);}

/* tabs and buttons */
.stTabs [aria-selected="true"] {color: #E8458B !important;}
.stButton button, .stFormSubmitButton button, .stDownloadButton button {
       background: linear-gradient(90deg, #E8458B, #F5B700); color: white; border: 0;
       border-radius: 30px; padding: 8px 26px; font-weight: 700;}
</style>""", unsafe_allow_html=True)


# ---------- small helpers (only for display) ----------
def hero(title, subtitle=""):
    """Colourful banner at the top of a page."""
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


CARD_COUNT = [0]
def card(value, label):
    """Number card; colours switch pink / yellow automatically."""
    tone = "pink" if CARD_COUNT[0] % 2 == 0 else "yellow"
    CARD_COUNT[0] += 1
    st.markdown(f'<div class="card {tone}"><h2>{value}</h2><p>{label}</p></div>',
                unsafe_allow_html=True)


def show_code(*functions):
    """Shows the real source code of the functions this page runs."""
    with st.expander("📜 Code used on this page (from heart_project.py)"):
        st.code("\n\n".join(inspect.getsource(f) for f in functions), language="python")


# ---------- load data once / train once (cached) ----------
@st.cache_data
def get_raw():
    return hp.load_data()


@st.cache_data
def get_clean():
    return hp.clean_data(get_raw())


@st.cache_resource(show_spinner="Training 6 models with GridSearchCV (same as notebook)… "
                                 "this takes about 1–2 minutes the first time.")
def get_project():
    return hp.run_full_project()


raw = get_raw()
clean = get_clean()

# ---------- sidebar menu: one page per notebook section ----------
st.sidebar.markdown("## 🫀 Heart Disease\n### Risk Prediction")
st.sidebar.caption("Click a step to see it 👇")
page = st.sidebar.radio("Project steps", [
    "🏠 Overview",
    "1️⃣ Dataset",
    "2️⃣ EDA",
    "3️⃣ Missing Values",
    "4️⃣ Outliers",
    "5️⃣ Model Training",
    "6️⃣ Evaluation",
    "7️⃣ Predict Risk",
    "✅ Conclusion",
])
st.sidebar.markdown("---")
st.sidebar.caption("Mehwish Naz · University of Larkano")


# =====================================================================
if page == "🏠 Overview":
    hero('<span class="beat">🫀</span> Heart Disease Risk Prediction',
         "An interactive view of my machine learning project. Each page is one step "
         "of the notebook and runs the same code.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: card(len(raw), "patient records")
    with c2: card(raw.shape[1], "columns")
    with c3: card(raw["dataset"].nunique(), "hospitals")
    with c4: card("6", "ML models compared")

    st.subheader("Project workflow")
    steps = [("1. Dataset", "load csv"), ("2. EDA", "explore columns"),
             ("3. Missing values", "impute"), ("4. Outliers", "remove BP = 0"),
             ("5. Training", "6 models, GridSearchCV"), ("6. Evaluation", "test set"),
             ("7. Prediction", "new patient")]
    for col, (t, d) in zip(st.columns(7), steps):
        col.markdown(f'<div class="step"><b>{t}</b><br>{d}</div>', unsafe_allow_html=True)

    st.subheader("How the files connect")
    st.markdown("""
| File | What it is |
|---|---|
| `HEART_DISEASE.ipynb` | my notebook: where I explored and built the project step by step |
| `heart_project.py` | the same notebook steps written as functions |
| `app.py` | this dashboard: runs `heart_project.py` and shows the results visually |
| `heart_disease_uci.csv` | dataset (UCI Heart Disease) |
| `heart_model.pkl` | best model saved by the notebook (`joblib.dump`) |
""")

# =====================================================================
elif page == "1️⃣ Dataset":
    hero("1️⃣ Dataset", "920 patients · 16 columns · 4 hospitals")
    c1, c2, c3 = st.columns(3)
    with c1: card(raw.shape[0], "rows (patients)")
    with c2: card(raw.shape[1], "columns")
    with c3: card(int(raw.isnull().sum().sum()), "missing cells")
    st.subheader("First rows  —  df.head()")
    st.dataframe(raw.head(10), use_container_width=True)

    st.subheader("Column information  —  df.info()")
    info = pd.DataFrame({"dtype": raw.dtypes.astype(str),
                         "non-null": raw.notnull().sum(),
                         "missing": raw.isnull().sum()})
    st.dataframe(info, use_container_width=True)

    st.subheader("Records per hospital")
    st.bar_chart(raw["dataset"].value_counts(), color=PINK)
    show_code(hp.load_data)

# =====================================================================
elif page == "2️⃣ EDA":
    hero("2️⃣ Exploratory Data Analysis", "Understanding the data with interactive charts")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Age", "Sex", "Hospitals (dataset)", "Chest pain (cp)", "Blood pressure"])

    with tab1:
        st.plotly_chart(hp.age_histogram(raw), use_container_width=True)
        st.write(f"Age range: **{raw['age'].min()} – {raw['age'].max()}** years.")
        show_code(hp.age_histogram)
    with tab2:
        counts, percent = hp.sex_counts(raw)
        c1, c2 = st.columns(2)
        with c1: card(f"{percent['Male']}%", f"male ({counts['Male']})")
        with c2: card(f"{percent['Female']}%", f"female ({counts['Female']})")
        st.plotly_chart(px.pie(names=counts.index, values=counts.values, hole=0.5,
                               color_discrete_sequence=[PINK, YELLOW]),
                        use_container_width=True)
        show_code(hp.sex_counts)
    with tab3:
        st.plotly_chart(hp.dataset_by_sex(raw), use_container_width=True)
        st.write("Mean / median / mode of age per hospital:")
        st.dataframe(hp.age_stats_by_dataset(raw), use_container_width=True)
        show_code(hp.dataset_by_sex, hp.age_stats_by_dataset)
    with tab4:
        by = st.radio("Colour by", ["sex", "dataset"], horizontal=True)
        st.plotly_chart(hp.chest_pain_by(raw, by), use_container_width=True)
        st.dataframe(raw["cp"].value_counts(), use_container_width=True)
        show_code(hp.chest_pain_by)
    with tab5:
        st.plotly_chart(hp.trestbps_histogram(raw), use_container_width=True)
        st.dataframe(raw["trestbps"].describe().to_frame().T, use_container_width=True)
        st.caption("Minimum is 0, which is impossible for blood pressure → handled in step 4.")
        show_code(hp.trestbps_histogram)

# =====================================================================
elif page == "3️⃣ Missing Values":
    hero("3️⃣ Dealing with Missing Values", "Filling the empty cells before training")
    before = hp.missing_percent(raw)
    after = hp.missing_percent(clean)
    c1, c2 = st.columns(2)
    with c1: card(int(raw.isnull().sum().sum()), "missing cells BEFORE")
    with c2: card(int(clean.isnull().sum().sum()), "missing cells AFTER")
    st.write("")
    left, right = st.columns(2)
    b = before[before > 0].round(1).sort_values()
    left.plotly_chart(px.bar(x=b.values, y=b.index, orientation="h", text=b.values,
                             labels={"x": "missing %", "y": "column"},
                             title="Before cleaning", color_discrete_sequence=[YELLOW]),
                      use_container_width=True)
    with right:
        st.markdown("""
**Method used**
- `trestbps`, `ca`, `oldpeak`, `thalch`, `chol`, `fbs`, `exang` → **IterativeImputer**
  (one column at a time, so it fills the column average)
- `thal`, `slope`, `restecg` → **mode** (most frequent category)
        """)
        if after.sum() == 0:
            st.success("After cleaning: 0 missing values left in every column ✅")
    show_code(hp.missing_percent, hp.fill_missing_values)

# =====================================================================
elif page == "4️⃣ Outliers":
    hero("4️⃣ Outliers", "Finding impossible values with box plots")
    filled = hp.fill_missing_values(raw)
    st.plotly_chart(hp.numeric_boxplots(filled), use_container_width=True)
    zero_bp = filled[filled["trestbps"] == 0]
    c1, c2, c3 = st.columns(3)
    with c1: card(len(zero_bp), "row with blood pressure = 0 (impossible)")
    with c2: card(f"{len(filled)} → {len(clean)}", "rows after removing it")
    with c3: card(int((clean["chol"] == 0).sum()), "cholesterol = 0 (made missing in step 5)")
    st.subheader("The removed row")
    st.dataframe(zero_bp, use_container_width=True)
    show_code(hp.numeric_boxplots, hp.remove_impossible_rows, hp.clean_data)

# =====================================================================
elif page == "5️⃣ Model Training":
    hero("5️⃣ Model Training & Tuning", "6 models · GridSearchCV · 5-fold cross-validation")
    X, y = hp.make_x_y(clean)
    c1, c2, c3 = st.columns(3)
    with c1: card(X.shape[1], "input features")
    with c2: card(int((y == 0).sum()), "healthy (y = 0)")
    with c3: card(int((y == 1).sum()), "heart disease (y = 1)")
    st.caption("Target: num = 0 → healthy, num 1–4 → disease.")

    st.subheader("Models and settings tried (GridSearchCV, 5-fold, F1)")
    grids = pd.DataFrame([[n, str(g)] for n, (_, g) in hp.SEARCH.items()],
                         columns=["Model", "Settings tried"])
    st.dataframe(grids, use_container_width=True, hide_index=True)

    project = get_project()                         # trains once, then cached
    res = project["results"].sort_values("CV F1")
    st.subheader(f"Result: best model = {project['best_name']}")
    c1, c2 = st.columns(2)
    with c1: card(project["sizes"][0], "training patients (80%)")
    with c2: card(project["sizes"][1], "test patients (20%)")
    fig = px.bar(res, x="CV F1", y="Model", orientation="h", text="CV F1",
                 color=np.where(res["Model"] == project["best_name"], "Best", "Other"),
                 color_discrete_map={"Best": YELLOW, "Other": PINK},
                 range_x=[0.8, 0.88], title="Cross-validated F1 of each model")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(project["results"].astype(str), use_container_width=True, hide_index=True)
    show_code(hp.make_x_y, hp.make_preprocessor, hp.split_data, hp.train_models)

# =====================================================================
elif page == "6️⃣ Evaluation":
    hero("6️⃣ Evaluation on the Test Set", "How well the best model does on 184 unseen patients")
    project = get_project()
    s = project["scores"]
    rep = s["report"]
    st.write(f"Best model: **{project['best_name']}**, tested once on "
             f"**{project['sizes'][1]} unseen patients**.")
    cols = st.columns(5)
    values = [(f"{s['accuracy'] * 100:.1f}%", "accuracy"),
              (f"{rep.loc['1', 'recall'] * 100:.1f}%", "recall (disease found)"),
              (f"{rep.loc['1', 'precision'] * 100:.1f}%", "precision"),
              (f"{rep.loc['1', 'f1-score']:.3f}", "F1-score"),
              (f"{s['roc_auc']:.3f}", "ROC-AUC")]
    for col, (v, l) in zip(cols, values):
        with col: card(v, l)
    st.write("")
    left, right = st.columns(2)
    cm = s["confusion_matrix"]
    left.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="RdPu",
                                x=["Predicted Healthy", "Predicted Disease"],
                                y=["Actual Healthy", "Actual Disease"],
                                title="Confusion matrix"), use_container_width=True)
    left.caption(f"{cm[1, 1]} sick patients found, {cm[1, 0]} missed (false negatives), "
                 f"{cm[0, 1]} false alarms.")
    roc = px.area(s["roc"], x="False positive rate", y="True positive rate",
                  title=f"ROC curve (AUC = {s['roc_auc']:.3f})",
                  color_discrete_sequence=[PINK])
    roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    right.plotly_chart(roc, use_container_width=True)

    st.subheader("Classification report")
    st.dataframe(rep.round(3), use_container_width=True)

    if hasattr(project["best_model"].named_steps["m"], "feature_importances_"):
        imp = hp.feature_importance(project["best_model"])
        st.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h",
                               labels={"x": "importance", "y": "feature"},
                               title="Top 12 important features",
                               color_discrete_sequence=[PINK]), use_container_width=True)
        st.caption("Importance = what the model uses, not what causes disease.")
    show_code(hp.evaluate, hp.feature_importance)

# =====================================================================
elif page == "7️⃣ Predict Risk":
    hero("7️⃣ Predict a New Patient", "Enter values and let the trained model decide")
    # Use the model saved by the notebook. If it is not there, train it here.
    model = None
    if os.path.exists(hp.MODEL_PATH):
        try:
            model = joblib.load(hp.MODEL_PATH)
            st.caption("Using **heart_model.pkl** — the model saved in my notebook.")
        except Exception:
            model = None      # e.g. saved with a different scikit-learn version
    if model is None:
        model = get_project()["best_model"]
        st.caption("Using the Random Forest trained in step 5 (same code as the notebook).")

    def opts(col):
        return sorted(raw[col].dropna().unique().tolist())

    # ---- ready-made example patients (fill the form with one click) ----
    EXAMPLES = {
        "healthy": dict(age=42, sex="Female", cp="atypical angina", trestbps=120, chol=210,
                        fbs="No", restecg="normal", thalch=172, exang="No", oldpeak=0.0,
                        slope="upsloping", ca=0, thal="normal"),
        "borderline": dict(age=55, sex="Male", cp="typical angina", trestbps=140, chol=230,
                           fbs="No", restecg="normal", thalch=150, exang="No", oldpeak=1.5,
                           slope="flat", ca=1, thal="normal"),          # notebook demo patient
        "high_risk": dict(age=63, sex="Male", cp="asymptomatic", trestbps=150, chol=280,
                          fbs="Yes", restecg="lv hypertrophy", thalch=112, exang="Yes",
                          oldpeak=2.5, slope="flat", ca=2, thal="reversable defect"),
    }
    for key, value in EXAMPLES["borderline"].items():       # values shown the first time
        st.session_state.setdefault(key, value)

    st.markdown("**Try an example patient:**")
    b1, b2, b3 = st.columns(3)
    for col, (name, label) in zip([b1, b2, b3], [("healthy", "🟢 Healthy example"),
                                                ("borderline", "🟡 Borderline example"),
                                                ("high_risk", "🔴 High-risk example")]):
        if col.button(label, use_container_width=True):
            st.session_state.update(EXAMPLES[name])         # fill the form
            st.session_state["auto_predict"] = True          # and predict right away

    with st.form("patient"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 20, 100, key="age")
        sex = c1.selectbox("Sex", opts("sex"), key="sex")
        cp = c1.selectbox("Chest pain type (cp)", opts("cp"), key="cp")
        trestbps = c1.number_input("Resting blood pressure", 80, 220, key="trestbps")
        chol = c2.number_input("Cholesterol", 100, 600, key="chol")
        fbs = c2.radio("Fasting blood sugar > 120?", ["No", "Yes"], horizontal=True, key="fbs")
        restecg = c2.selectbox("Resting ECG", opts("restecg"), key="restecg")
        thalch = c2.number_input("Maximum heart rate (thalch)", 60, 220, key="thalch")
        exang = c3.radio("Exercise-induced angina?", ["No", "Yes"], horizontal=True, key="exang")
        oldpeak = c3.number_input("ST depression (oldpeak)", -3.0, 7.0, step=0.1, key="oldpeak")
        slope = c3.selectbox("ST slope", opts("slope"), key="slope")
        ca = c3.selectbox("Major vessels (ca)", [0, 1, 2, 3], key="ca")
        thal = c3.selectbox("Thal", opts("thal"), key="thal")
        go_btn = st.form_submit_button("🔍 Predict")

    go_btn = go_btn or st.session_state.pop("auto_predict", False)

    if go_btn:
        patient = {"age": age, "sex": sex, "cp": cp, "trestbps": float(trestbps),
                   "chol": float(chol), "fbs": 1.0 if fbs == "Yes" else 0.0,
                   "restecg": restecg, "thalch": float(thalch),
                   "exang": 1.0 if exang == "Yes" else 0.0, "oldpeak": float(oldpeak),
                   "slope": slope, "ca": float(ca), "thal": thal}
        label, risk = hp.predict_patient(model, patient)
        left, right = st.columns(2)
        colour = PINK if label != "Healthy" else "#3B9C6E"
        left.markdown(f'<div class="result" style="background:{colour}">'
                      f'<h2 style="color:white;margin:0">{label}</h2>'
                      f'<h1 style="color:white;margin:0">{risk:.1f}%</h1>'
                      f'<p>probability of heart disease (model score)</p></div>',
                      unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": PINK},
                   "steps": [{"range": [0, 50], "color": "#E9F6EF"},
                             {"range": [50, 100], "color": "#FFE3EF"}],
                   "threshold": {"line": {"color": YELLOW, "width": 4}, "value": 50}}))
        gauge.update_layout(height=250, margin=dict(t=20, b=10))
        right.plotly_chart(gauge, use_container_width=True)
        st.caption("50% or more → Heart Disease Risk. ⚠️ Educational prototype, not a diagnosis.")
    show_code(hp.predict_patient)

# =====================================================================
elif page == "✅ Conclusion":
    hero("✅ Conclusion & What I Learned", "What the project found and what I learned")
    left, right = st.columns(2)
    with left:
        st.subheader("What I found")
        st.markdown("""
- 6 models were compared fairly with 5-fold cross-validation
- **Random Forest** was the best model (highest CV F1)
- It finds most of the sick patients in the test set (high recall)
- Most important features: asymptomatic chest pain, exercise angina,
  ST depression, max heart rate, age
        """)
        st.subheader("Limitations")
        st.markdown("""
- Small, old dataset; 79% male
- `ca` and `thal` were mostly missing
- One train/test split only
- A risk score, not a medical diagnosis
        """)
    with right:
        st.subheader("What I learned")
        st.markdown("""
- **Clean data first:** missing values and impossible zeros must be fixed
- **Pipelines:** new data is processed exactly like training data
- **Recall matters:** missing a sick patient is the worst error
- **Fair comparison:** cross-validation, test set used only once
- **Honest results:** prediction ≠ diagnosis
        """)
        st.subheader("Tools")
        st.markdown("Python · Pandas · NumPy · Plotly · Scikit-learn · XGBoost · Joblib · Streamlit")
