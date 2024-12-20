# Import Library
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Data
@st.cache_data
def load_data():
    day_df = pd.read_csv("../data/day.csv")
    hour_df = pd.read_csv("../data/hour.csv")
    return day_df, hour_df

day_df, hour_df = load_data()

# Set Style
sns.set(style="whitegrid")

# --- Header ---
st.title("📊 Bike Sharing Analysis Dashboard")
st.markdown("""
Selamat datang di **Dashboard Analisis Penyewaan Sepeda**. 
Dashboard ini menampilkan hasil analisis data dengan visualisasi interaktif, termasuk tren penggunaan sepeda berdasarkan waktu, cuaca, dan musim.
""")

# --- Sidebar ---
st.sidebar.header("🔧 Filter Data")
selected_data = st.sidebar.radio("Pilih Dataset", ["Daily Data", "Hourly Data"])
show_raw_data = st.sidebar.checkbox("Tampilkan Data Mentah", False)

if selected_data == "Daily Data":
    data = day_df
    st.sidebar.write("Data harian dipilih.")
else:
    data = hour_df
    st.sidebar.write("Data per jam dipilih.")

if show_raw_data:
    st.write("### Data Mentah")
    st.dataframe(data)

# --- 1. Tren Penyewaan Sepeda Berdasarkan Waktu ---
st.header("🚴 Tren Penyewaan Sepeda Berdasarkan Waktu")
time_option = st.selectbox("Pilih Analisis Waktu", ["Bulan", "Hari", "Jam"])

if time_option == "Bulan":
    monthly_trend = data.groupby('mnth')['cnt'].sum().reset_index()
    max_month = monthly_trend.loc[monthly_trend['cnt'].idxmax()]

    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x='mnth', y='cnt', data=monthly_trend)

    for bar, month, count in zip(bars.patches, monthly_trend['mnth'], monthly_trend['cnt']):
        if month == max_month['mnth']:
            bar.set_facecolor('orange')
            bar.set_edgecolor('red')
            bar.set_linewidth(2)

    plt.text(max_month['mnth']-1, max_month['cnt'] + 10000,
         f"Max: {max_month['cnt']:,}", color='red', fontsize=10, ha='center')
    
    plt.title('Tren Jumlah Penyewa Sepeda Berdasarkan Bulan', fontsize=14, fontweight='bold')
    plt.xlabel('Bulan')
    plt.ylabel('Jumlah Penyewa')
    plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(plt)

    st.markdown("""
    **Insight:**
    Penyewa lebih banyak pada bulan Juni - September. 
    Dimana puncak sewa terbanyak terdapat pada bulan Agustus yakni 351.194 penyewaan. 
    Sedangkan terendah pada bulan Januari - Februari.
    """)

elif time_option == "Hari":
    day_trend = data.groupby('weekday')['cnt'].mean()
    day_trend.index = ['Minggu', 'Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu']
    plt.figure(figsize=(10, 6))
    day_trend.plot(kind='bar', color='skyblue', edgecolor='k')
    plt.title('Tren Penyewaan Sepeda Berdasarkan Hari', fontweight='bold')
    plt.xlabel('Hari')
    plt.ylabel('Rata-rata Jumlah Penyewa')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

    st.markdown("""
    **Insight:**
    Tidak ada perbedaan signifikan penyewa sepeda berdasarkan hari.
    """)

else:
    hourly_trend = hour_df.groupby('hr')['cnt'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_trend.index, hourly_trend.values,
         marker='o', linestyle='-', color='royalblue', linewidth=2)
    plt.title('Tren Penyewaan Sepeda Harian Berdasarkan Jam', fontsize=14, fontweight='bold')
    plt.xlabel('Jam', fontsize=12)
    plt.ylabel('Rata-rata Jumlah Penyewa', fontsize=12)
    plt.xticks(range(0, 24), labels=[f"{i}:00" for i in range(24)], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)

    st.markdown("""
    **Insight:**
    Mulai jam 05.00 - 08.00 dan 15.00 - 17.00 terdapat peningkatan permintaan penyewaan sepeda. 
    Terdapat kemungkinan bahwasanya pada jam tersebut seseorang berangkat dan pulang kerja. 
    Pada jam tersebut merupakan waktu yang baik untuk olahraga maupun menikmati suasana.
    """)

# --- 2. Perbandingan Penyewaan Sepeda Casual dan Registered ---
st.header("✨Perbandingan Penyewaan Sepeda Casual dan Registered")

df_grouped = data.groupby('yr')[['registered', 'casual']].sum().reset_index()
df_grouped['yr'] = df_grouped['yr'].map({0: '2011', 1: '2012'})
st.dataframe(df_grouped)

tot_registered = day_df['registered'].sum()
tot_casual = day_df['casual'].sum()

total = [tot_registered, tot_casual]
labels = ['Registered', 'Casual']

plt.figure(figsize=(7, 5))
plt.pie(total, labels=labels, autopct='%1.1f%%')
plt.title('Persentase Penyewaan Sepeda Registered vs Casual', fontweight='bold', fontsize=16)
plt.tight_layout()
st.pyplot(plt)

st.markdown("""
**Insight:**
Penyewaan sepeda Registered mencapai 81.2% lebih banyak dari Casual. 
Pada tahun 2011 dan 2012 juga lebih banyak penyewa sepeda Registered.
""")

# --- 3. Pengaruh Cuaca terhadap Penyewaan ---
st.header("🌦️ Pengaruh Cuaca terhadap Penyewaan Sepeda")

plt.figure(figsize=(10, 6))
sns.boxplot(x='weathersit', y='cnt', data=day_df)

plt.title('Distribusi Jumlah Penyewa Berdasarkan Kondisi Cuaca', fontweight='bold')
plt.xlabel('Kondisi Cuaca')
plt.ylabel('Jumlah Penyewa')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Clear', 'Cloudy', 'Light Snow', 'Heavy Rain'], rotation=20)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(plt)

st.markdown("""
**weathersit:**
1. Clear, Few clouds, Partly cloudy, Partly cloudy
2. Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3. Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4. Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

**Insight:**
- Cuaca cerah memiliki penyewaan sepeda tertinggi.
- Penyewaan berkurang saat salju ringan atau hujan ringan.
- Tidak ada penyewaan saat hujan deras atau badai.
""")

# --- 4. Segmentasi Cluster Penyewa Sepeda ---
st.header("📈 Segmentasi Penyewa Sepeda Menggunakan K-Means")

# Preprocessing
features = ['temp', 'atemp', 'season']
X = day_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
day_df['cluster'] = kmeans.fit_predict(X_scaled)

season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
day_df['season_label'] = day_df['season'].map(season_map)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='season_label', hue='cluster', palette='Set2', data=day_df, s=100)
plt.title('Segmentasi Penyewaan Sepeda Berdasarkan Cluster')
plt.xlabel('Temperatur (temp)')
plt.ylabel('Musim')
plt.legend(title='Cluster', loc='upper right')
st.pyplot(plt)

st.markdown("""
**Interpretasi Cluster:**
- Cluster 0: Pengguna dengan temperatur rendah, feeling temperature yang juga rendah, dan cenderung ke musim dingin.
- Cluster 1: Pengguna dengan temperatur dan feeling temperature yang lebih tinggi, dengan sedikit kecenderungan ke musim panas dan gugur.
- Cluster 2: Pengguna dengan temperatur dan feeling temperature yang sangat rendah, serta cenderung ke musim semi.
""")

# --- 4. Rekomendasi Strategi Promosi ---
st.header("🎯 Rekomendasi Strategi Promosi")
st.info("""
1. **Periode Promosi**: Fokus pada bulan **Juni - September** karena jumlah penyewaan sepeda tertinggi.
2. **Cuaca Cerah**: Dorong promosi ketika cuaca cerah untuk meningkatkan penyewaan.
3. **Pagi dan Sore Hari**: Tawarkan diskon pada jam sibuk (05:00 - 08:00 dan 15:00 - 17:00).
4. **Segmentasi Pelanggan**: Gunakan hasil **clustering** untuk memberikan program loyalitas.
""")

# --- Footer ---
st.write("---")
st.write("🚀 Dashboard ini dibuat menggunakan **Streamlit** | Moch. Andyka Saputra | Desember 2024")

