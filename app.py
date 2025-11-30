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

# --- FUNGSI LOAD DATA DEC (INTERNAL) ---
@st.cache_data
def load_dec_data():
    file_path = 'clustered_data_dec.csv'
    
    try:
        # Membaca file CSV lokal
        df = pd.read_csv(file_path)
        
        # --- LOGIKA PENENTUAN LABEL KLASTER (OTOMATIS) ---
        # Karena label 0 dan 1 bisa tertukar saat training ulang,
        # kita tentukan label berdasarkan rata-rata GDP_per_Capita.
        # Klaster dengan rata-rata GDP lebih rendah = "Developing/Low"
        
        avg_gdp = df.groupby('Cluster')['GDP_per_Capita'].mean()
        
        # Asumsi: Jika hanya ada 2 cluster (0 dan 1)
        if 0 in avg_gdp.index and 1 in avg_gdp.index:
            if avg_gdp[0] < avg_gdp[1]:
                label_map = {0: 'Low Economy - Low Energy', 1: 'High Economy - High Energy'}
            else:
                label_map = {1: 'Low Economy - Low Energy', 0: 'High Economy - High Energy'}
        else:
            # Fallback jika data hanya memiliki 1 jenis cluster
            label_map = {0: 'Cluster 0', 1: 'Cluster 1'}
            
        df['Cluster Label'] = df['Cluster'].map(label_map)
        return df
        
    except FileNotFoundError:
        # --- DATA DUMMY (JIKA FILE TIDAK DITEMUKAN) ---
        # Ini hanya agar aplikasi tidak error saat Anda belum menaruh file csv
        st.warning(f"File '{file_path}' tidak ditemukan. Menampilkan Data Dummy.")
        
        countries = ['Afghanistan', 'Indonesia', 'United States', 'China', 'India', 'Japan']
        years = range(2000, 2024)
        data = []
        
        for country in countries:
            # Simulasi nilai dasar
            base_gdp = 60000 if country in ['United States', 'Japan'] else 2000
            base_energy = 8000 if country in ['United States', 'Japan'] else 1000
            
            # Tentukan cluster dummy
            cluster = 1 if base_gdp > 20000 else 0 
            
            for year in years:
                growth = 1 + (np.random.rand() * 0.1)
                gdp_cap = base_gdp * growth * (1 + (year-2000)*0.05)
                energy = base_energy * growth * (1 + (year-2000)*0.03)
                
                data.append({
                    'Country Name': country,
                    'Year': year,
                    'Energy_Consumption_kWh': energy,
                    'GDP': gdp_cap * 1000000, # Asumsi populasi
                    'Population': 1000000,
                    'GDP_per_Capita': gdp_cap,
                    'log_Energy': np.log10(energy),
                    'log_GDP_per_Capita': np.log10(gdp_cap),
                    'Cluster': cluster,
                    # Label dummy manual
                    'Cluster Label': 'High Economy - High Energy' if cluster == 1 else 'Low Economy - Low Energy'
                })
        return pd.DataFrame(data)

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

# --- HALAMAN 2: DEC (CLUSTERING) ---
elif pilihan_menu == "🧩 Clustering (DEC)":
    st.header("🧩 Segmentasi Negara (Deep Embedded Clustering)")
    st.markdown("Analisis pengelompokan negara berdasarkan **GDP per Kapita** dan **Konsumsi Energi**.")
    
    # 1. Load Data
    df_dec = load_dec_data()
    
    # 2. Filter Layout (DI TENGAH HALAMAN, BUKAN SIDEBAR)
    # Membuat container dengan background tipis atau separator agar rapi
    with st.container():
        st.subheader("⚙️ Filter Data")
        
        col_f1, col_f2 = st.columns([2, 1]) # Kolom 1 lebih lebar untuk Slider
        
        with col_f1:
            # Slider Tahun
            min_year = int(df_dec['Year'].min())
            max_year = int(df_dec['Year'].max())
            selected_year = st.slider("Pilih Tahun Analisis:", min_year, max_year, max_year)
            
        with col_f2:
            # Dropdown Negara
            country_list = sorted(df_dec['Country Name'].unique())
            default_ix = country_list.index('Indonesia') if 'Indonesia' in country_list else 0
            selected_country = st.selectbox("Pilih Negara untuk Detail:", country_list, index=default_ix)

    # Filter Dataframe
    df_year = df_dec[df_dec['Year'] == selected_year]
    
    # --- LAYOUT VISUALISASI ---
    st.markdown("---")
    
    # A. PETA DUNIA
    st.subheader(f"🗺️ Peta Persebaran Kluster ({selected_year})")
    
    color_map = {
        'Low Economy - Low Energy': '#FF6B6B',  # Merah
        'High Economy - High Energy': '#4ECDC4' # Cyan
    }
    
    fig_map = px.choropleth(
        df_year,
        locations="Country Name",
        locationmode="country names",
        color="Cluster Label",
        hover_name="Country Name",
        hover_data={
            "GDP_per_Capita": ":.2f", 
            "Energy_Consumption_kWh": ":.2f",
            "Cluster": False,
            "Country Name": False
        },
        color_discrete_map=color_map,
        projection="natural earth",
        title=f"Global Clustering: Ekonomi vs Energi ({selected_year})"
    )
    fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, height=450)
    st.plotly_chart(fig_map, use_container_width=True)
    
    # B. DETAIL NEGARA & SCATTER PLOT
    col_kiri, col_kanan = st.columns([1, 2])
    
    with col_kiri:
        st.subheader(f"📊 Detail: {selected_country}")
        
        country_data = df_year[df_year['Country Name'] == selected_country]
        
        if not country_data.empty:
            row = country_data.iloc[0]
            label = row['Cluster Label']
            
            if "High" in label:
                st.success(f"**Status:** {label}")
                desc = "Negara ini memiliki tingkat ekonomi dan konsumsi energi yang **Tinggi**."
            else:
                st.warning(f"**Status:** {label}")
                desc = "Negara ini memiliki tingkat ekonomi dan konsumsi energi yang **Rendah/Berkembang**."
            
            st.markdown(desc)
            
            m1, m2 = st.columns(2)
            m1.metric("GDP/Kapita", f"${row['GDP_per_Capita']:,.0f}")
            m2.metric("Energi/Kapita", f"{row['Energy_Consumption_kWh']:,.0f} kWh")
            
            st.caption(f"Log-Scale: GDP={row['log_GDP_per_Capita']:.2f}, Energi={row['log_Energy']:.2f}")
        else:
            st.error(f"Data untuk {selected_country} pada tahun {selected_year} tidak tersedia.")

    with col_kanan:
        st.subheader("📈 Posisi dalam Cluster (Scatter Plot)")
        
        fig_scatter = px.scatter(
            df_year,
            x="log_GDP_per_Capita",
            y="log_Energy",
            color="Cluster Label",
            hover_name="Country Name",
            color_discrete_map=color_map,
            title=f"Sebaran Negara (Log Scale) - {selected_year}",
            labels={
                "log_GDP_per_Capita": "Log GDP per Capita",
                "log_Energy": "Log Energy Consumption"
            }
        )
        
        # Highlight Negara Terpilih
        highlight = df_year[df_year['Country Name'] == selected_country]
        if not highlight.empty:
            fig_scatter.add_scatter(
                x=highlight['log_GDP_per_Capita'],
                y=highlight['log_Energy'],
                mode='markers+text',
                marker=dict(size=15, color='black', symbol='circle-open', line=dict(width=3)),
                text=[selected_country],
                textposition="top center",
                name="Pilihan"
            )

        st.plotly_chart(fig_scatter, use_container_width=True)

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