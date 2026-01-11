import streamlit as st
import pickle
import pandas as pd

# Load pipeline
with open("model/rainfall_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.set_page_config(page_title="Rainfall Prediction", layout="centered")

# -------------------- HEADER --------------------
st.title("🌧️ Rainfall Prediction Dashboard")
st.caption("Simple ML-powered weather prediction")

# -------------------- LOAD FEATURES --------------------
preprocessor = pipeline.named_steps["preprocess"]

num_features, cat_features = [], []

for name, transformer, cols in preprocessor.transformers_:
    if name == "num":
        num_features = cols
    elif name == "cat":
        cat_features = cols

# -------------------- INPUT SECTION --------------------
st.subheader("🧾 Enter Weather Details")

input_data = {}

with st.container():
    
    for feature in num_features:
        input_data[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=0.0
        )

    st.markdown("### 🏷️ Categorical Features")
    for feature in cat_features:
        input_data[feature] = st.text_input(
            feature.replace("_", " ").title()
        )

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Rainfall"):
    input_df = pd.DataFrame([input_data])

    with st.spinner("Running prediction..."):
        prediction = pipeline.predict(input_df)[0]

        if hasattr(pipeline, "predict_proba"):
            probability = pipeline.predict_proba(input_df)[0][1]
        else:
            probability = None

    st.divider()

    # -------------------- DASHBOARD --------------------
    st.subheader("📊 Prediction Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Rainfall Prediction",
            value="YES ☔" if prediction == 1 else "NO ☀️"
        )

    with col2:
        if probability is not None:
            st.metric(
                label="Rain Probability",
                value=f"{probability:.1%}"
            )
        else:
            st.metric("Rain Probability", "N/A")

    with col3:
        st.metric(
            label="Model Output",
            value=int(prediction)
        )

    # Progress bar for probability
    if probability is not None:
        st.markdown("### 🌧️ Rain Probability Level")
        st.progress(probability)

    # Result message
    if prediction == 1:
        st.success("☔ High chance of rainfall. Carry an umbrella!")
    else:
        st.info("☀️ Rain unlikely. Weather looks clear.")

    # -------------------- INPUT PREVIEW --------------------
    st.markdown("### 📋 Input Summary")
    st.dataframe(input_df, use_container_width=True)
