import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Analisis Big Data Energi & Ekonomi",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI LOAD DATA GRANGER (CACHE) ---
@st.cache_data
def load_granger_data():
    try:
        return pd.read_csv('granger_result_final.csv')
    except FileNotFoundError:
        return None

# --- SIDEBAR MENU (RADIO BUTTON) ---
st.sidebar.title("Navigasi Sistem")
st.sidebar.markdown("---")

# Menggunakan Radio Button agar menu terlihat semua (bukan dropdown)
pilihan_menu = st.sidebar.radio(
    "Pilih Metode Analisis:",
    [
        "🏠 Beranda",
        "📈 Forecasting (LSTM)",
        "🧩 Clustering (DEC)",
        "🔗 Kausalitas (Granger)",  # <--- GANTI NAMA MENU DI SINI
        "🌲 Klasifikasi (Deep Forest)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Sistem Analisis Big Data\nKelompok 4")

# --- HALAMAN BERANDA ---
if pilihan_menu == "🏠 Beranda":
    st.title("Analisis Big Data: Nexus Energi & Ekonomi")
    st.markdown("""
    Selamat datang di Dashboard Analisis Big Data. Sistem ini menggunakan 4 metode utama 
    untuk menganalisis hubungan antara **PDB (Ekonomi)**, **Konsumsi Energi**, dan **Populasi**:

    1.  **LSTM (Long Short-Term Memory)**
        * *Fungsi:* Memprediksi tren konsumsi energi global di masa depan berdasarkan data historis.
    2.  **DEC (Deep Embedded Clustering)**
        * *Fungsi:* Melakukan segmentasi negara menjadi klaster (misal: "Low-Low" dan "High-High").
    3.  **Granger Causality Test** * *Fungsi:* Menentukan **arah hubungan sebab-akibat**. Apakah Ekonomi mendorong Energi, atau Energi mendorong Ekonomi?
    4.  **Deep Forest (gcForest)**
        * *Fungsi:* Mengklasifikasikan kategori energi negara menggunakan pendekatan Deep Learning berbasis pohon keputusan.
    """)
    
    col1, col2, col3 = st.columns(3) 
    
    col1.metric("Total Negara", "266", "Data World Bank")
    col2.metric("Rentang Data", "1960 - 2024", "Tahun")
    col3.metric("Metode AI", "4 Model", "Integrated")

# --- HALAMAN 1: LSTM (Forecasting) ---
elif pilihan_menu == "📈 Forecasting (LSTM)":
    st.header("📈 Peramalan Deret Waktu (Time Series Forecasting)")
    st.subheader("Metode: Long Short-Term Memory (LSTM)")
    st.write("Model ini memprediksi konsumsi energi masa depan berdasarkan pola historis.")
    
    # Input User
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            n_years = st.number_input("Prediksi berapa tahun ke depan?", min_value=1, max_value=10, value=5)
            run_btn = st.button("Jalankan Prediksi LSTM")
        
        with col2:
            if run_btn:
                # TODO: Masukkan logika load model LSTM dan prediksi di sini
                st.success(f"Memproses prediksi untuk {n_years} tahun ke depan...")
                
                # Placeholder Grafik Dummy
                chart_data = pd.DataFrame(
                    np.random.randn(20, 3),
                    columns=['Aktual', 'Prediksi', 'Baseline']
                )
                st.line_chart(chart_data)
                st.caption("Grafik: Perbandingan Data Aktual vs Prediksi Model LSTM")

# --- HALAMAN 2: DEC (Clustering) ---
elif pilihan_menu == "🧩 Clustering (DEC)":
    st.header("🧩 Segmentasi Negara (Clustering)")
    st.subheader("Metode: Deep Embedded Clustering (DEC)")
    
    st.write("Mengelompokkan negara berdasarkan profil GDP dan Konsumsi Energi.")
    st.info("Berdasarkan analisis, jumlah cluster optimal adalah **K=2** (Low-Low & High-High).")
    
    tab1, tab2 = st.tabs(["Visualisasi Cluster", "Detail Data"])
    
    with tab1:
        if st.button("Tampilkan Scatter Plot"):
            st.success("Menampilkan hasil clustering...")
            
            # Placeholder Grafik Scatter Dummy
            df_dummy = pd.DataFrame({
                'Log GDP': np.random.rand(100) * 10,
                'Log Energi': np.random.rand(100) * 10,
                'Cluster': np.random.choice(['Low-Low', 'High-High'], 100)
            })
            
            st.scatter_chart(
                df_dummy,
                x='Log GDP',
                y='Log Energi',
                color='Cluster',
                size=20
            )
            
    with tab2:
        st.write("Data hasil clustering akan muncul di sini.")

# --- HALAMAN 3: GRANGER CAUSALITY (METODE ANDA) ---
elif pilihan_menu == "🔗 Kausalitas (Granger)":
    st.header("🔗 Analisis Kausalitas Energi & Ekonomi")
    st.subheader("Metode: Granger Causality Test")
    st.markdown("Menentukan arah hubungan: **Apakah Energi mendorong Ekonomi, atau sebaliknya?**")
    
    # 1. Load Data Granger
    df_granger = load_granger_data()
    
    if df_granger is None:
        st.error("⚠️ File 'granger_result_final.csv' tidak ditemukan.")
        st.warning("Harap jalankan script analisis Granger terlebih dahulu untuk menghasilkan data.")
    else:
        # Layout: Kiri (Pilih Negara & Info), Kanan (Peta)
        col_kiri, col_kanan = st.columns([1, 2])
        
        with col_kiri:
            st.markdown("### 🔍 Cek Negara")
            daftar_negara = sorted(df_granger['Country'].unique())
            selected_country = st.selectbox("Pilih Negara:", daftar_negara)
            
            # Ambil Data Negara
            country_data = df_granger[df_granger['Country'] == selected_country].iloc[0]
            hasil = country_data['Hypothesis']
            
            st.divider()
            st.markdown(f"**Hasil Analisis: {selected_country}**")
            
            # Tampilan Kartu Hasil
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
            st.write(f"Energi → GDP: `{country_data['P_Val_Energy_to_GDP']}`")
            st.write(f"GDP → Energi: `{country_data['P_Val_GDP_to_Energy']}`")
        
        with col_kanan:
            st.markdown("### 🗺️ Peta Persebaran Global")
            # Membuat Peta Choropleth
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

# --- HALAMAN 4: Deep Forest (Klasifikasi) ---
elif pilihan_menu == "🌲 Klasifikasi (Deep Forest)":
    st.header("🌲 Klasifikasi Kategori Energi")
    st.subheader("Metode: Deep Forest / gcForest")
    st.write("Memprediksi kategori klaster negara menggunakan input GDP dan Energi.")
    
    with st.form("form_klasifikasi"):
        c1, c2 = st.columns(2)
        input_gdp = c1.number_input("GDP per Capita (USD)", value=5000)
        input_energy = c2.number_input("Konsumsi Energi (kWh)", value=2000)
        
        submitted = st.form_submit_button("Prediksi Kategori")
        
        if submitted:
            # TODO: Masukkan logika prediksi model Deep Forest
            st.info("Memproses klasifikasi dengan Cascade Forest...")
            
            # Simulasi hasil
            hasil_prediksi = "Cluster 0 (Low-Low)"
            st.metric("Hasil Prediksi:", hasil_prediksi)
            st.success("Model berhasil mengklasifikasikan data input.")