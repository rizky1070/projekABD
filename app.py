import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Energy-Economy Nexus AI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        font-weight: 600;
    }
    .warning-box {
        background-color: #fffae5;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        color: #664d03;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 1. FUNGSI LOAD DATA (TERPUSAT)
# -------------------------------------------------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "model.h5")
scaler_path = os.path.join(base_path, "scalers.pkl")

@st.cache_data
def load_all_data():
    """Memuat semua dataset sekaligus agar sinkron"""
    data = {}
    
    # 1. Data LSTM (Data Bersih)
    try:
        df_lstm = pd.read_csv("data_bersih.csv")
        df_lstm["log_Energy"] = np.log10(df_lstm["Energy_Consumption_kWh"] + 1)
        data['lstm'] = df_lstm
    except:
        data['lstm'] = None

    # 2. Data DEC (Clustering)
    try:
        df_dec = pd.read_csv('clustered_data_dec.csv')
        # Logic Labeling
        avg_gdp = df_dec.groupby('Cluster')['GDP_per_Capita'].mean()
        if 0 in avg_gdp.index and 1 in avg_gdp.index:
            if avg_gdp[0] < avg_gdp[1]:
                label_map = {0: 'Low Economy - Low Energy', 1: 'High Economy - High Energy'}
            else:
                label_map = {1: 'Low Economy - Low Energy', 0: 'High Economy - High Energy'}
        else:
            label_map = {0: 'Cluster 0', 1: 'Cluster 1'}
        df_dec['Cluster Label'] = df_dec['Cluster'].map(label_map)
        data['dec'] = df_dec
    except:
        data['dec'] = None

    # 3. Data Granger
    try:
        data['granger'] = pd.read_csv('granger_result_final.csv')
    except:
        data['granger'] = None
        
    return data

# Load Data Awal
ALL_DATA = load_all_data()

# -------------------------------------------------------------------------
# 2. SIDEBAR (GLOBAL CONTROLLER)
# -------------------------------------------------------------------------
st.sidebar.title("🌍 Navigasi Utama")

# A. PILIH MODE
mode_analisis = st.sidebar.radio(
    "Mode Tampilan:",
    ["📊 Executive Dashboard", "📈 Detail: Forecasting", "🧩 Detail: Clustering", "🔗 Detail: Kausalitas"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Filter Global")

# B. PILIH NEGARA (Global State)
if ALL_DATA['lstm'] is not None:
    country_list = sorted(ALL_DATA['lstm']['Country Name'].unique())
else:
    country_list = ["Indonesia"]

if 'selected_country' not in st.session_state:
    st.session_state.selected_country = "Indonesia"

selected_country = st.sidebar.selectbox(
    "Pilih Negara Fokus:",
    country_list,
    index=country_list.index(st.session_state.selected_country) if st.session_state.selected_country in country_list else 0,
    key='country_selector'
)
st.session_state.selected_country = selected_country

# C. PILIH TAHUN (Global State)
current_year_max = 2023
if ALL_DATA['dec'] is not None:
    current_year_max = int(ALL_DATA['dec']['Year'].max())

selected_year = st.sidebar.slider("Tahun Analisis (Peta):", 2000, current_year_max, current_year_max)

st.sidebar.info(f"Fokus Analisis: **{selected_country}**")

# -------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------------------

def get_country_metrics(country, year=None):
    if ALL_DATA['dec'] is None: return None
    df = ALL_DATA['dec']
    
    # Filter Negara
    df_c = df[df['Country Name'] == country]
    if df_c.empty: return None
    
    # Filter Tahun (Strict)
    if year is not None:
        df_year = df_c[df_c['Year'] == year]
        if df_year.empty:
            return None 
        return df_year.iloc[0]
        
    # Default
    return df_c.sort_values('Year').iloc[-1]

def render_lstm_forecast(country, n_years=10):
    df_lstm = ALL_DATA['lstm']
    if df_lstm is None: return st.error("Data LSTM tidak ada.")
    
    try:
        from tensorflow.keras.layers import LSTM
        class FixedLSTM(LSTM):
            def __init__(self, **kwargs):
                if 'time_major' in kwargs: kwargs.pop('time_major')
                super().__init__(**kwargs)
                
        model = load_model(model_path, custom_objects={'LSTM': FixedLSTM}, compile=False)
        
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
            
        df_c = df_lstm[df_lstm["Country Name"] == country].sort_values("Year")
        values = df_c["log_Energy"].values
        look_back = int(model.input_shape[1])
        
        if len(values) < look_back + 1:
            st.warning("Data historis kurang untuk prediksi.")
            return

        last_seq = values[-look_back:]
        preds = []
        for _ in range(n_years):
            seq_scaled = scalers["scaler_X"].transform(last_seq.reshape(1, -1)).reshape(1, look_back, 1)
            p_scaled = model.predict(seq_scaled, verbose=0)
            p = scalers["scaler_y"].inverse_transform(p_scaled)[0][0]
            preds.append(p)
            last_seq = np.append(last_seq[1:], p)
            
        future_years = list(range(int(df_c["Year"].max()) + 1, int(df_c["Year"].max()) + 1 + n_years))
        y_pred_real = [10**p - 1 for p in preds]
        
        df_hist = df_c.tail(15)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist['Year'], y=df_hist['Energy_Consumption_kWh'], name='Data Historis', line=dict(color='#1f77b4', width=3)))
        
        x_connect = [df_hist['Year'].iloc[-1]] + future_years
        y_connect = [df_hist['Energy_Consumption_kWh'].iloc[-1]] + y_pred_real
        
        fig.add_trace(go.Scatter(x=x_connect, y=y_connect, name='Prediksi AI', line=dict(color='#ff7f0e', width=3, dash='dot')))
        
        fig.update_layout(title=f"Forecast Energi: {country} (+{n_years} Thn)", xaxis_title="Tahun", yaxis_title="kWh", height=350, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Gagal memuat model forecasting: {e}")

# -------------------------------------------------------------------------
# 4. HALAMAN UTAMA (EXECUTIVE DASHBOARD)
# -------------------------------------------------------------------------
if mode_analisis == "📊 Executive Dashboard":
    st.title(f"📊 Dashboard Analisis: {selected_country}")
    
    # --- BARIS 1: KEY METRICS ---
    metrics = get_country_metrics(selected_country, year=selected_year)
    
    if metrics is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GDP per Kapita", f"${metrics['GDP_per_Capita']:,.0f}", delta_color="normal")
        c2.metric("Konsumsi Energi", f"{metrics['Energy_Consumption_kWh']:,.0f} kWh", delta_color="normal")
        
        status_label = metrics['Cluster Label']
        status_color = "#4ECDC4" if "High" in status_label else "#FF6B6B"
        c3.markdown(f"**Status Ekonomi-Energi**\n\n<span style='color:{status_color}; font-weight:bold; font-size:1.2em'>{status_label}</span>", unsafe_allow_html=True)
        
        g_res = "Data Tidak Tersedia"
        if ALL_DATA['granger'] is not None:
            dg = ALL_DATA['granger']
            row_g = dg[dg['Country'] == selected_country]
            if not row_g.empty:
                g_res = row_g.iloc[0]['Hypothesis']
        c4.markdown(f"**Hubungan Kausalitas**\n\n{g_res}")
    else:
        st.warning(f"⚠️ **Data Tidak Tersedia**: Data {selected_country} untuk tahun {selected_year} kosong. Metrik tidak dapat ditampilkan.")
        st.caption("💡 Tips: Coba geser 'Slider Tahun' di sidebar ke tahun-tahun sebelumnya.")

    st.markdown("---")

    # --- BARIS 2: FORECASTING & GEOSPATIAL ---
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        df_lstm = ALL_DATA['lstm']
        df_c = df_lstm[df_lstm["Country Name"] == selected_country]
        if df_c.empty:
            st.warning(f"⚠️ Data historis untuk Forecasting {selected_country} tidak ditemukan.")
        else:
            render_lstm_forecast(selected_country, n_years=10)
        
    with col_right:
        st.subheader("📍 Posisi & Peta Global")
        
        tab_map, tab_scatter = st.tabs(["🗺️ Peta Dunia", "🔍 Scatter Plot"])
        
        df_dec = ALL_DATA['dec']
        if df_dec is not None:
            df_curr = df_dec[df_dec['Year'] == selected_year]
            hl = df_curr[df_curr['Country Name'] == selected_country]
            is_missing = hl.empty
            
            with tab_map:
                if is_missing:
                    st.warning(f"⚠️ Peta tahun {selected_year} tidak mencakup data {selected_country}.")
                fig_map = px.choropleth(
                    df_curr, locations="Country Name", locationmode="country names", color="Cluster Label",
                    color_discrete_map={'Low Economy - Low Energy': '#FF6B6B', 'High Economy - High Energy': '#4ECDC4'},
                    hover_name="Country Name", title=f"Peta Sebaran ({selected_year})"
                )
                fig_map.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), showlegend=False, geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth'))
                st.plotly_chart(fig_map, use_container_width=True)

            with tab_scatter:
                if is_missing:
                    st.warning(f"⚠️ Posisi statistik {selected_country} thn {selected_year} tidak diketahui.")
                fig_pos = px.scatter(
                    df_curr, x="log_GDP_per_Capita", y="log_Energy", color="Cluster Label",
                    color_discrete_map={'Low Economy - Low Energy': '#FF6B6B', 'High Economy - High Energy': '#4ECDC4'},
                    hover_name="Country Name", title=f"Posisi Statistik ({selected_year})"
                )
                if not is_missing:
                    fig_pos.add_trace(go.Scatter(
                        x=hl['log_GDP_per_Capita'], y=hl['log_Energy'], mode='markers',
                        marker=dict(size=25, color='yellow', symbol='star', line=dict(width=2, color='black')), 
                        name=selected_country, showlegend=False
                    ))
                fig_pos.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
                st.plotly_chart(fig_pos, use_container_width=True)

# -------------------------------------------------------------------------
# 5. HALAMAN DETAIL: FORECASTING (LSTM)
# -------------------------------------------------------------------------
elif mode_analisis == "📈 Detail: Forecasting":
    st.header(f"📈 Analisis Forecasting Mendalam: {selected_country}")
    
    df_lstm = ALL_DATA['lstm']
    df_c = df_lstm[df_lstm["Country Name"] == selected_country].sort_values("Year")
    
    if df_c.empty:
        st.error(f"❌ Data historis (LSTM) untuk negara **{selected_country}** sama sekali tidak ditemukan.")
        st.stop()
        
    last_year_data = int(df_c['Year'].max())
    if last_year_data < selected_year:
        st.warning(f"⚠️ Data historis terakhir **{selected_country}** adalah tahun **{last_year_data}**.")
    
    n_input = st.slider("Jumlah Tahun Prediksi:", 1, 30, 10)
    render_lstm_forecast(selected_country, n_years=n_input)
    st.subheader("📄 Data Historis")
    st.dataframe(df_c[['Year', 'Energy_Consumption_kWh', 'log_Energy']].sort_values('Year', ascending=False), use_container_width=True)

# -------------------------------------------------------------------------
# 6. HALAMAN DETAIL: CLUSTERING (DEC)
# -------------------------------------------------------------------------
elif mode_analisis == "🧩 Detail: Clustering":
    st.header(f"🧩 Peta Segmentasi Global ({selected_year})")
    
    df_dec = ALL_DATA['dec']
    df_year = df_dec[df_dec['Year'] == selected_year]
    hl = df_year[df_year['Country Name'] == selected_country]
    
    if hl.empty:
        st.warning(f"⚠️ **Data Kosong:** Negara **{selected_country}** tidak memiliki data clustering pada tahun **{selected_year}**.")
    else:
        st.success(f"✅ Menampilkan posisi **{selected_country}** pada tahun **{selected_year}**.")
    
    fig_map = px.choropleth(
        df_year, locations="Country Name", locationmode="country names", color="Cluster Label",
        color_discrete_map={'Low Economy - Low Energy': '#FF6B6B', 'High Economy - High Energy': '#4ECDC4'},
        title="Peta Distribusi Cluster", projection="natural earth"
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    st.subheader("🔍 Analisis Posisi")
    fig_sc = px.scatter(
        df_year, x="log_GDP_per_Capita", y="log_Energy", color="Cluster Label",
        hover_name="Country Name", color_discrete_map={'Low Economy - Low Energy': '#FF6B6B', 'High Economy - High Energy': '#4ECDC4'}
    )
    if not hl.empty:
        fig_sc.add_trace(go.Scatter(
            x=hl['log_GDP_per_Capita'], y=hl['log_Energy'], mode='markers',
            marker=dict(size=25, color='yellow', symbol='star', line=dict(width=2, color='black')),
            text=[selected_country], textposition="top center", name=selected_country
        ))
    st.plotly_chart(fig_sc, use_container_width=True)

# -------------------------------------------------------------------------
# 7. HALAMAN DETAIL: KAUSALITAS (GRANGER) - DENGAN LAYOUT BARU
# -------------------------------------------------------------------------
elif mode_analisis == "🔗 Detail: Kausalitas":
    st.header("🔗 Analisis Kausalitas Energi & Ekonomi")
    st.subheader("Metode: Granger Causality Test")
    st.markdown("Menentukan arah hubungan: **Apakah Energi mendorong Ekonomi, atau sebaliknya?**")
    
    # Penjelasan Metodologi
    st.info("""
    ℹ️ **Catatan:** Hasil Granger menggunakan seluruh data historis (Time Series) jangka panjang. 
    Filter 'Tahun' di sidebar tidak mempengaruhi hasil analisis ini.
    """)
    
    # 1. Load Data Granger
    df_granger = ALL_DATA['granger']
    
    if df_granger is None:
        st.error("⚠️ File 'granger_result_final.csv' tidak ditemukan.")
        st.warning("Harap jalankan script analisis Granger terlebih dahulu.")
    else:
        # Filter Negara (Menggunakan Global State 'selected_country')
        country_data_df = df_granger[df_granger['Country'] == selected_country]
        
        # Layout 1:2 (Kiri: Info, Kanan: Peta)
        col_kiri, col_kanan = st.columns([1, 2])
        
        with col_kiri:
            st.markdown(f"### 🔍 Hasil: {selected_country}")
            st.divider()
            
            if not country_data_df.empty:
                country_data = country_data_df.iloc[0]
                hasil = country_data['Hypothesis']
                
                # Tampilan Kartu Hasil (Menggunakan Code Anda)
                if hasil == 'Neutrality':
                    st.info(f"🟦 **{hasil}**")
                    st.caption("Tidak ada hubungan sebab-akibat langsung dalam jangka pendek.")
                elif hasil == 'Growth Hypothesis':
                    st.success(f"🟩 **{hasil}**")
                    st.caption("Energi mendorong Pertumbuhan Ekonomi.")
                elif hasil == 'Conservation Hypothesis':
                    st.warning(f"🟨 **{hasil}**")
                    st.caption("Pertumbuhan Ekonomi mendorong Konsumsi Energi.")
                else:
                    st.error(f"🟪 **{hasil}**")
                    st.caption("Saling mempengaruhi (Feedback).")
                    
                st.markdown("---")
                st.write("**Statistik (P-Value):**")
                st.write(f"Energi → GDP: `{country_data['P_Val_Energy_to_GDP']:.4f}`")
                st.write(f"GDP → Energi: `{country_data['P_Val_GDP_to_Energy']:.4f}`")
            else:
                 st.warning(f"""
                ⚠️ **Data Tidak Ditemukan:** Tidak ada hasil uji Granger untuk negara **{selected_country}**.
                Mungkin data historis terlalu pendek.
                """)
        
        with col_kanan:
            st.markdown("### 🗺️ Peta Persebaran Global")
            # Membuat Peta Choropleth (Code Anda)
            fig = px.choropleth(
                df_granger,
                locations="Country",
                locationmode='country names',
                color="Hypothesis",
                color_discrete_map={
                    'Neutrality': 'lightgrey',
                    'Growth Hypothesis': 'green',
                    'Conservation Hypothesis': 'orange',
                    'Feedback Hypothesis': 'purple'
                },
                hover_name="Country",
                hover_data=['P_Val_Energy_to_GDP', 'P_Val_GDP_to_Energy'],
                height=500
            )
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)