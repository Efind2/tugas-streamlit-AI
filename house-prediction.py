import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("https://raw.githubusercontent.com/Efind2/tugas-streamlit-AI/main/housing.csv")

        return data
    except FileNotFoundError:
        st.error("File housing.csv tidak ditemukan.")
        st.stop()

# Preprocessing dan training
@st.cache_data
def preprocess_and_train(data):
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_prep, y_train)
    y_pred = model.predict(X_test_prep)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, preprocessor, mse, r2, X, y

# Main app
def main():
    st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")
    st.title("🏠 Prediksi Harga Rumah California")
    st.markdown("Masukkan fitur-fitur rumah di bawah ini untuk memprediksi nilai rumah (median_house_value).")

    data = load_data()
    model, preprocessor, mse, r2, X, y = preprocess_and_train(data)

    input_data = {}

    st.subheader("🔧 Input Data Rumah")
    for col in X.columns:
        if X[col].dtype == 'object':
            input_data[col] = st.selectbox(
                f"{col}",
                options=sorted(X[col].dropna().unique()),
                index=0
            )
        else:
            input_data[col] = st.number_input(
                f"{col}",
                min_value=float(X[col].min()),
                max_value=float(X[col].max()),
                value=float(X[col].mean())
            )

    if st.button("Prediksi Harga"):
        input_df = pd.DataFrame([input_data])
        input_preprocessed = preprocessor.transform(input_df)
        prediction = model.predict(input_preprocessed)[0]
        st.success(f"💰 Prediksi Median House Value: **${prediction:,.2f}**")
        st.info(f"Evaluasi model:\n- MSE: {mse:,.2f}\n- R² Score: {r2:.2f}")

    # Opsi tambahan
    if st.checkbox("📊 Tampilkan Dataset"):
        st.dataframe(data)

    if st.checkbox("📈 Tampilkan Visualisasi Distribusi Target"):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data["median_house_value"], bins=50, kde=True, ax=ax)
        ax.set_title("Distribusi Median House Value")
        st.pyplot(fig)

    if st.checkbox("📉 Tampilkan Korelasi"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
