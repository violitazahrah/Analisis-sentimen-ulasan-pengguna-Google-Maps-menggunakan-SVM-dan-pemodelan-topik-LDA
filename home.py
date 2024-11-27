import streamlit as st
import joblib

# Judul Pada Tab
st.set_page_config(page_title="Analisis Sentimen Google Maps", page_icon="📊")

def main():
  # Judul Website
  st.title("Analisis Sentimen Ulasan Pengguna Google Maps")
  st.image("img/gmaps.png", width=100)

  # st.image("hero.jpg")
  # logo
  st.logo("img/gmaps.png")

  # Perkenalan website
  st.write("""
  Hi! Selamat Datang di Website **ANALISIS SENTIMEN APLIKASI GOOGLE MAPS**. Website ini merupakan sebuah sistem yang dibangun dengan tujuan memahami serta mengetahui sentimen pengguna terhadap aplikasi Google Maps. Pada sistem akan dihasilkan kelompok ulasan dengan Sentimen Positif dan Sentimen Negatif, tentunya dilengkapi dengan konsep Pemodelan Topik guna mengetahui topik utama bahkan topik tersembunyi pada ulasan pengguna.
  """)

  # Information tentang aplikasi
  st.header("Informasi Google Maps")
  st.write("""
  - **Goggle Maps**: Google Maps merupakan sebuah aplikasi peta online yang dapat diakses secara fleksibel menggunakan handphone dengan sambungan internet yang disediakan oleh layanan Google. Dengan tujuan mempermudah perjalanan dari suatu titik lokasi ke lokasi lainnya.
  """)

  #  Spesifikasi fitur yang akan ditampilkan
  st.header("Spesifikasi Sistem")
  st.write(""" Sistem ini memuat beberapa informasi yang akan memberi tahu pengguna mengenai layanan Google Maps, hal tersebut berupa:
  - **Klasifikasi Sentimen**: Sistem ini akan menampilkan ulasan dengan kelas **Positif** atau **Negatif**.
  - **Visualisasi Sentimen**: Dengan menampilkan visualisasi perbandingan sentimen positif dan negatif dengan WordCloud akan mempermudah dalam menyimpulkan performa aplikasi Google Maps yang masih terus harus ditingkatkan.
  - **Pemodelan Topik**: Pemodelan topik akan menampilkan kata tersembunyi dan kata yang menjadi topik utama pembahasan pengguna terhadap aplikasi Google Maps   
  """)
if __name__ == "__main__":
    main()