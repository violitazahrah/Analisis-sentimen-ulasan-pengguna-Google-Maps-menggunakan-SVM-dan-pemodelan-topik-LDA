import streamlit as st
import joblib

# Judul Pada Tab
st.set_page_config(page_title="Analisis Sentimen Google Maps", page_icon="📊") 

def main():
  st.title("Uji Ulasan Baru")

  # logo
  st.logo("img/gmaps.png")

  # TESTING
  # input data testing
  st.subheader("Input Ulasan Baru")

  # Input ulasan baru dari pengguna
  input_text = st.text_input("Masukkan Ulasan Baru: ")

  # Memuat model yang sudah disimpan
  loaded_model = joblib.load('model/model_gmaps.pkl')
        
  if input_text:
    # Prediksi hasil analisis menggunakan model grid_search
    result_test = loaded_model.predict([input_text])
    # Tampilkan hasil analisis
    st.write(f"Hasil Analisis: {result_test[0]}")
  # END TESTING


if __name__ == "__main__":
    main()

# Footer
st.markdown("""
---
Developed by Violita A. Zahrah. All rights reserved.
""")