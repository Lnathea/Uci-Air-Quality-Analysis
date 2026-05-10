import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Analysis",
    page_icon="🌫️",
    layout="wide"
)

# ── Load / Train Model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model dari file .pkl, atau latih ulang dari CSV jika tidak ada."""
    feature_cols = [
        'CO_sensor', 'NMHC_sensor', 'NOx_sensor', 'NO2_sensor', 'O3_sensor',
        'Temperature', 'Humidity_rel', 'Humidity_abs',
        'hour', 'dayofweek', 'month', 'is_weekend', 'is_rush_hour'
    ]
    if os.path.exists('model_rf.pkl'):
        model = joblib.load('model_rf.pkl')
    else:
        df = load_data()
        df_ml = df[feature_cols + ['CO_ref']].dropna()
        X = df_ml[feature_cols]
        y = df_ml['CO_ref']
        model = RandomForestRegressor(
            n_estimators=100, max_depth=12,
            min_samples_leaf=5, n_jobs=-1, random_state=42
        )
        model.fit(X, y)
    return model, feature_cols

@st.cache_data
def load_data():
    """Load cleaned dataset."""
    df = pd.read_csv('air_quality_cleaned.csv', index_col='DateTime', parse_dates=True)
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/UCI-Air%20Quality-blue", use_column_width=True)
st.sidebar.title("🌫️ Air Quality App")
st.sidebar.markdown("Analisis & prediksi kualitas udara berbasis Machine Learning")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "🔮 Prediksi CO", "📊 Eksplorasi Data"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🌫️ UCI Air Quality Analysis")
    st.markdown("**End-to-end Data Science project — EDA & Machine Learning pada data sensor kualitas udara**")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    df = load_data()
    col1.metric("Total Data", f"{len(df):,} jam")
    col2.metric("Periode", "Mar 2004 – Feb 2005")
    col3.metric("Rata-rata CO", f"{df['CO_ref'].mean():.2f} mg/m³")
    col4.metric("CO Tertinggi", f"{df['CO_ref'].max():.2f} mg/m³")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📌 Tentang Project")
        st.markdown("""
        Dataset ini berisi data sensor kualitas udara per jam dari Italia.
        Fitur utama yang dianalisis:
        - **CO, Benzene, NOx, NO2** — polutan dari kendaraan
        - **Suhu & Kelembapan** — kondisi cuaca
        - **Sensor vs Referensi** — akurasi sensor murah

        **Model terbaik: Random Forest (R² = 0.74)**
        """)

    with col_b:
        st.subheader("🏆 Hasil Model")
        results = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest ⭐', 'XGBoost'],
            'MAE':  [0.4415, 0.4388, 0.4473],
            'RMSE': [0.6623, 0.6557, 0.6744],
            'R²':   [0.7390, 0.7442, 0.7294]
        })
        st.dataframe(results.set_index('Model'), use_container_width=True)

    st.divider()
    st.subheader("🗺️ Alur Analisis")
    c1, c2, c3 = st.columns(3)
    c1.info("**Part 1**\n\nData Cleaning & Preprocessing\n\nHandling -200, DateTime parsing, outlier capping")
    c2.success("**Part 2**\n\nExploratory Data Analysis\n\nRush hour, korelasi, tren musiman")
    c3.warning("**Part 3**\n\nMachine Learning\n\nRegresi (RF, XGB) + K-Means Clustering")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDIKSI CO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi CO":
    st.title("🔮 Prediksi Kadar CO")
    st.markdown("Masukkan data sensor dan kondisi lingkungan untuk memprediksi kadar CO (mg/m³)")
    st.divider()

    model, feature_cols = load_model()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔧 Data Sensor")
        co_sensor   = st.slider("CO Sensor",    500,  2500, 1000)
        nmhc_sensor = st.slider("NMHC Sensor",  500,  2500, 900)
        nox_sensor  = st.slider("NOx Sensor",   100,  2000, 600)
        no2_sensor  = st.slider("NO2 Sensor",   500,  2500, 1000)
        o3_sensor   = st.slider("O3 Sensor",    500,  2500, 1000)

    with col2:
        st.subheader("🌤️ Kondisi Cuaca")
        temperature  = st.slider("Suhu (°C)",           -5,  45, 20)
        humidity_rel = st.slider("Kelembapan Relatif (%)", 10, 100, 55)
        humidity_abs = st.slider("Kelembapan Absolut",  0.0, 2.5, 1.0, step=0.1)

    with col3:
        st.subheader("⏰ Waktu")
        hour      = st.slider("Jam", 0, 23, 8)
        dayofweek = st.selectbox("Hari", 
            options=[0,1,2,3,4,5,6],
            format_func=lambda x: ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][x]
        )
        month     = st.slider("Bulan", 1, 12, 6)
        is_weekend   = 1 if dayofweek >= 5 else 0
        is_rush_hour = 1 if hour in [7,8,9,17,18,19] else 0

        st.markdown(f"🚗 Rush hour: **{'Ya' if is_rush_hour else 'Tidak'}**")
        st.markdown(f"📅 Weekend: **{'Ya' if is_weekend else 'Tidak'}**")

    st.divider()

    # Prediksi
    input_data = pd.DataFrame([[
        co_sensor, nmhc_sensor, nox_sensor, no2_sensor, o3_sensor,
        temperature, humidity_rel, humidity_abs,
        hour, dayofweek, month, is_weekend, is_rush_hour
    ]], columns=feature_cols)

    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)  # tidak boleh negatif

    # Tampilkan hasil
    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.metric("🎯 Prediksi CO", f"{prediction:.3f} mg/m³")

    with col_res2:
        # Klasifikasi level polusi
        if prediction < 1.0:
            level, color = "✅ Baik", "green"
        elif prediction < 2.0:
            level, color = "🟡 Sedang", "orange"
        elif prediction < 4.0:
            level, color = "🔴 Tidak Sehat", "red"
        else:
            level, color = "⛔ Berbahaya", "darkred"
        st.metric("📊 Level Polusi", level)

    with col_res3:
        avg_co = 1.65  # rata-rata dataset
        diff   = prediction - avg_co
        st.metric("📈 vs Rata-rata Dataset", f"{diff:+.3f} mg/m³",
                  delta_color="inverse")

    # Gauge sederhana
    st.divider()
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh(['CO'], [min(prediction, 6)], color='#0052D9', height=0.4)
    ax.barh(['CO'], [6], color='#E0E0E0', height=0.4)
    ax.barh(['CO'], [min(prediction, 6)], color=(
        '#1DB954' if prediction < 1 else
        '#FFA500' if prediction < 2 else
        '#FF3B6B' if prediction < 4 else '#8B0000'
    ), height=0.4)
    ax.axvline(avg_co, color='gray', linestyle='--', linewidth=1.5, label=f'Rata-rata ({avg_co})')
    ax.set_xlim(0, 6)
    ax.set_xlabel('CO (mg/m³)')
    ax.set_title('Level CO Prediksi')
    ax.legend(fontsize=8)
    ax.spines[['top','right','left']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EKSPLORASI DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data")
    df = load_data()
    st.divider()

    tab1, tab2, tab3 = st.tabs(["⏱️ Pola Waktu", "🔗 Korelasi", "📈 Tren"])

    # Tab 1: Pola Waktu
    with tab1:
        st.subheader("Rata-rata CO per Jam dalam Sehari")
        hourly = df.groupby('hour')['CO_ref'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        rush = hourly['hour'].isin([7,8,9,17,18,19])
        ax.bar(hourly['hour'], hourly['CO_ref'],
               color=['#FF6B35' if r else '#0052D9' for r in rush], alpha=0.8)
        ax.set_xlabel('Jam')
        ax.set_ylabel('Rata-rata CO (mg/m³)')
        ax.set_xticks(range(0, 24))
        ax.spines[['top','right']].set_visible(False)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#FF6B35', label='Rush hour'),
            Patch(color='#0052D9', label='Non-rush hour')
        ])
        st.pyplot(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Weekday vs Weekend")
            wk = df.groupby('is_weekend')['CO_ref'].mean()
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.bar(['Hari Kerja', 'Weekend'], wk.values,
                    color=['#0052D9', '#1DB954'], alpha=0.8)
            ax2.set_ylabel('Rata-rata CO (mg/m³)')
            ax2.spines[['top','right']].set_visible(False)
            st.pyplot(fig2, use_container_width=True)

        with col2:
            st.subheader("CO per Musim")
            season = df.groupby('season')['CO_ref'].mean().sort_values(ascending=False)
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            ax3.bar(season.index, season.values,
                    color=['#0052D9','#FF6B35','#1DB954','#00C4E8'], alpha=0.8)
            ax3.set_ylabel('Rata-rata CO (mg/m³)')
            ax3.spines[['top','right']].set_visible(False)
            st.pyplot(fig3, use_container_width=True)

    # Tab 2: Korelasi
    with tab2:
        st.subheader("Heatmap Korelasi Polutan")
        cols = ['CO_ref','Benzene_ref','NOx_ref','NO2_ref','Temperature','Humidity_rel']
        corr = df[cols].corr()
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=ax4, linewidths=0.5)
        ax4.set_title('Korelasi Antar Variabel')
        st.pyplot(fig4, use_container_width=True)

    # Tab 3: Tren
    with tab3:
        st.subheader("Tren CO Sepanjang Waktu")
        polutan = st.selectbox("Pilih polutan", ['CO_ref','Benzene_ref','NOx_ref','NO2_ref'])
        fig5, ax5 = plt.subplots(figsize=(12, 4))
        ax5.plot(df.index, df[polutan], alpha=0.15, color='#0052D9', linewidth=0.5)
        rolling = df[polutan].rolling(window=168, center=True).mean()
        ax5.plot(df.index, rolling, color='#FF6B35', linewidth=2, label='Rolling mean 7 hari')
        ax5.set_ylabel(polutan)
        ax5.legend()
        ax5.spines[['top','right']].set_visible(False)
        st.pyplot(fig5, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("Made with ❤️ by **[Nama Kamu]**")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/Lnathea/Uci-Air-Quality-Analysis)")
