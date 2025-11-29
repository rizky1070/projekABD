import streamlit as st
import pandas as pd
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Analisis Big Data Energi & Ekonomi",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        "🔗 Asosiasi (Apriori)",
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
        * [cite_start]*Fungsi:* Memprediksi tren konsumsi energi global di masa depan berdasarkan data historis[cite: 50, 264].
    2.  **DEC (Deep Embedded Clustering)**
        * [cite_start]*Fungsi:* Melakukan segmentasi negara menjadi klaster (misal: "Low-Low" dan "High-High")[cite: 36, 1181].
    3.  **Apriori (Association Rule Mining)**
        * [cite_start]*Fungsi:* Menemukan pola aturan tersembunyi, contoh: *"Jika GDP Rendah, maka Energi Rendah"*[cite: 37, 279].
    4.  **Deep Forest (gcForest)**
        * [cite_start]*Fungsi:* Mengklasifikasikan kategori energi negara menggunakan pendekatan Deep Learning berbasis pohon keputusan[cite: 38, 281].
    """)
    
    # Perhatikan baris di bawah ini, pastikan ada kurung tutup ')'
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
    
    # Pastikan baris ini punya kurung tutup ')' di akhir!
    st.write("Mengelompokkan negara berdasarkan profil GDP dan Konsumsi Energi.")
    
    # Baris ini yang tadi error (sekarang harusnya aman)
    st.info("Berdasarkan analisis, jumlah cluster optimal adalah **K=2** (Low-Low & High-High).")
    
    tab1, tab2 = st.tabs(["Visualisasi Cluster", "Detail Data"])
    
    with tab1:
        if st.button("Tampilkan Scatter Plot"):
            # TODO: Masukkan logika load hasil clustering
            st.success("Menampilkan hasil clustering...")
            
            # Placeholder Grafik Scatter Dummy
            # Membuat data dummy seolah-olah GDP vs Energi
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

# --- HALAMAN 3: Apriori (Asosiasi) ---
elif pilihan_menu == "🔗 Asosiasi (Apriori)":
    st.header("🔗 Analisis Pola Asosiasi")
    st.subheader("Metode: Algoritma Apriori")
    st.write("Menemukan aturan 'Sebab-Akibat' (If-Then rules) antara kategori ekonomi dan energi.")
    
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1)
    with col2:
        min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.6)
        
    if st.button("Cari Aturan (Generate Rules)"):
        # TODO: Masukkan logika Apriori di sini
        st.success("Analisis selesai. Menampilkan aturan yang ditemukan:")
        
        # [cite_start]Contoh hasil hardcoded sesuai dokumen Anda [cite: 1293]
        st.markdown("""
        **Aturan Signifikan yang Ditemukan:**
        1. `{GDP_Low} -> {Energy_Low}` (Lift: 2.40)
        2. `{GDP_High} -> {Energy_High}` (Lift: 2.35)
        """)

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