import csv
import os
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model
import pyLDAvis.gensim_models as gensimvis
import gensim
from gensim.models import CoherenceModel
from gensim import corpora
import streamlit.components.v1 as components
from html2image import Html2Image

nltk.download('punkt')
nltk.download('stopwords')
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud, STOPWORDS
import re
import joblib

# Fungsi untuk memuat data
def load_data():
    try:
        df = pd.read_csv('preprocessed_data.csv')
        return df
    except FileNotFoundError:
        st.error("File 'preprocessed_data.csv' tidak ditemukan.")
        return None
    
# Fungsi untuk membersihkan teks
def clean_maps_data(text):
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)
    return text

# Fungsi untuk normalisasi kata
def normalisasi(str_text, norm):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
    return str_text

#  Fungsi untuk menghapus stopwords
def stopword_removal(text, stopwords):
    tokens = word_tokenize(text)
    return ' '.join([token for token in tokens if token.lower() not in stopwords])

# Fungsi stemming
def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

# Fungsi tokenizing
def tokenize(text):
    return text.split()

# Fungsi untuk membersihkan data tokenized
def clean_data(tokenized_list):
    # Mengubah list kembali menjadi string
    clean_str = ' '.join(tokenized_list)
    return clean_str

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaModel(corpus=corpus,
                                        id2word=dictionary,
                                        num_topics=num_topics, 
                                        random_state=42,
                                        chunksize=100,
                                        passes=10,
                                        per_word_topics=True)
        
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        model_list.append(model)
    return model_list, coherence_values
# Fungsi untuk membangun model LDA
def build_lda_model(df_review, review_colname):
    docs_raw = df_review[review_colname].tolist()

    # Menggunakan TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                        lowercase=True,
                                        token_pattern=r'\b[a-zA-Z]{3,}\b',
                                        max_df=0.9,
                                        min_df=10)

    # Konversi ke document-term matrix
    dtm_tfidf = tfidf_vectorizer.fit_transform(docs_raw)
    return dtm_tfidf, tfidf_vectorizer

# Fungsi untuk mendapatkan topik dan probabilitas
def get_topic_probabilities(model, dtm):
    topic_probabilities = model.transform(dtm)
    return topic_probabilities

# Judul Pada Tab
st.set_page_config(page_title="Analisis Sentimen Google Maps", page_icon="📊")

# Sidebar untuk navigasi
def main():
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Pilih Menu", ("Unggah Data", "Cleansing Data", "Preprocessing Data", "Visualisasi Data", "Splitting Data", "Pemodelan Topik LDA"))

    if menu == "Unggah Data":
        show_upload_data()
    else:
        if 'uploaded_df' not in st.session_state:
            st.warning("Harap unggah file CSV terlebih dahulu di menu 'Unggah Data'.")
        else:
            if menu == "Cleansing Data":
                show_cleansing_data()
            elif menu == "Preprocessing Data":
                show_preprocessing_data()
            elif menu == "Visualisasi Data":
                show_visualization()
            elif menu == "Splitting Data":
                show_splitting_data()
            elif menu == "Pemodelan Topik LDA":
                show_lda_modeling()

# Fungsi untuk mengunggah data (hanya dilakukan sekali)
def show_upload_data():
    st.title("Unggah Data")
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])

    if uploaded_file is not None:
        st.session_state['uploaded_df'] = pd.read_csv(uploaded_file)  # Simpan ke session_state
        st.write("Data yang diunggah:")
        st.dataframe(st.session_state['uploaded_df'].tail())
        st.success("File berhasil diunggah. Anda sekarang dapat mengakses proses lainnya.")

# Fungsi untuk cleansing data
def show_cleansing_data():
    st.title("Cleansing Data")

    # Menghapus duplikat
    st.subheader("Menghapus Duplikat")
    df_clean = st.session_state['uploaded_df'].drop_duplicates(subset='content')
    st.dataframe(df_clean.tail())
    st.write(f"Data setelah menghapus duplikat: {df_clean.shape[0]} baris")

    # Menghapus nilai kosong
    st.subheader("Menghapus Nilai Kosong")
    df_clean = df_clean.dropna()
    st.dataframe(df_clean.tail())
    st.write(f"Data setelah menghapus nilai kosong: {df_clean.shape[0]} baris")

    # Fungsi untuk membersihkan teks
    def clean_maps_data(text):
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Menghapus mention
        text = re.sub(r'#\w+', '', text)  # Menghapus hashtag
        text = re.sub(r'RT[\s]+', '', text)  # Menghapus 'RT'
        text = re.sub(r'https?://\S+', '', text)  # Menghapus URL
        text = re.sub(r'[^A-Za-z0-9 ]', '', text)  # Menghapus karakter yang tidak diinginkan
        text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
        return text

    # Menerapkan fungsi pembersihan pada kolom 'content'
    df_clean['content'] = df_clean['content'].apply(clean_maps_data)

    st.subheader("Menghapus simbol dan angka")
    st.dataframe(df_clean.tail())  # Menampilkan data setelah pembersihan
    st.write(f"Data setelah menghapus simbol dan angka: {df_clean.shape[0]} baris")


    # Fungsi yang digunakan untuk menghapus nomor pada teks
    def remove_numbers(text):
        return re.sub(r'\d+', '', text)

    # Gunakan fungsi pada kolom 'content'
    df_clean['content'] = df_clean['content'].apply(remove_numbers)
    st.subheader("Data Setelah Menghapus Nomor")
    st.dataframe(df_clean.tail())

    # Mengubah menjadi lower case
    st.subheader("Mengubah menjadi Lowercase")
    df_clean['content'] = df_clean['content'].str.lower()
    st.dataframe(df_clean.tail())

    # Langkah 3: Filter baris di mana jumlah kata dalam kolom 'content' kurang dari 4
    df_clean = df_clean[df_clean['content'].apply(lambda x: len(x.split()) >= 4)]
    st.subheader("Filterisasi data kurang dari 4 kata")
    st.dataframe(df_clean.tail())
    st.write(f"Data memiliki: {df_clean.shape[0]} baris dan {df_clean.shape[1]} kolom")

        # Simpan hasil preprocess ke CSV
    st.session_state['uploaded_df'] = df_clean
    csv_file = 'hasil_cleaned.csv'
    df_clean.to_csv(csv_file, index=False)  # Simpan file CSV


# Fungsi untuk preprocessing data
def show_preprocessing_data():
    st.title("Preprocessing Data")

    # Load CSV hasil cleansing
    try:
        df_preprocessed = pd.read_csv('hasil_cleaned.csv')  # Memuat data hasil cleansing dari CSV
        st.success("Data hasil cleansing berhasil dimuat!")
    except FileNotFoundError:
        st.error("File hasil cleansing tidak ditemukan. Silakan lakukan proses cleansing terlebih dahulu.")
        return

    # Normalisasi
    st.subheader("Normalisasi Kata")
    norm = {' gk ': ' tidak ', ' mmg ': ' memang ', ' krna ': ' karena ', ' krn ': ' karena ',
            ' no ': ' nomor ', ' ktp ': ' kartu tanda penduduk ', ' trus ': ' terus ',
            ' yobain ': ' cobain ', ' sya ': ' saya ', ' doang ': ' saja ', ' kaga ': ' tidak ',
            ' mf ': ' maaf ', ' sayah ': ' saya ', ' ilang ': ' hilang ', ' apk ': ' aplikasi ',
            ' gua ': ' saya ', ' duwit ': ' uang ', ' yg ': ' yang ', ' gx ': ' tidak ',
            ' knpa ': ' kenapa ', ' mulu ': ' terus ', ' skrng ': ' sekarang ', ' kyk ': ' seperti ',
            ' bgt ': ' sangat ', ' naroh ': ' meletakkan ', ' mw ': ' mau ', ' sdh ': ' sudah ',
            ' dapt ': ' dapat ', ' bukak ': ' buka ', ' tdk ': ' tidak ', ' jg ': ' juga ',
            ' makasih ': ' terimakasih ', ' makin ': ' semakin ', ' ga ': ' tidak ', ' ngirim ': ' mengirim ',
            ' knp ': ' kenapa ', ' muter ': ' putar ', ' ni ': ' ini ', ' skarang ': ' sekarang ',
            ' kalo ': ' kalau ', ' jgn ': ' jangan ', ' bgtu ': ' begitu ', ' mulu ': ' terus ',
            ' dibales ': ' dibalas ', ' blm ': ' belum ', ' bgs ': ' bagus ', ' cmn ': ' cuma ',
            ' dah ': ' sudah ', ' mnding ': ' mending ', ' pdhal ': ' padahal ', ' smua ': ' semua ',
            ' lg ': ' lagi ', ' dri ': ' dari ', ' tida ': ' tidak ', ' nmr ': ' nomor ', ' br ': ' baru ',
            ' tmn ': ' teman ', ' gw ': ' saya ', ' aja ': ' saja ', ' gausah ': ' tidak perlu ',
            ' tlp ': ' telepon ', ' sistim ': ' sistem ', ' udh ': ' sudah ', ' goblooook ': ' bodoh ',
            ' vermuk ': ' verifikasi muka ', ' seyelah ': ' setelah ', ' opresinal ': ' operasional ',
            ' aktivitasi ': ' aktivasi ', ' gabisa ': ' tidak bisa ', ' garagara ': ' dikarenakan ',
            ' trs ': ' terus ', ' verivikasi ': ' verifikasi ', ' nyesel ': ' menyesal ', ' tlg ': ' tolong ',
            ' moga ': ' semoga ', ' ngga ': ' tidak ', ' diem ': ' diam ', ' klo ': ' kalau ',
            ' kayak ': ' seperti ', ' tololll ': ' bodoh ', ' ngak ': ' nggak ', ' tpi ': ' tetapi ',
            ' bengking ': ' mobile banking ', ' jd ': ' jadi ', ' bs ': ' dapat ', ' g ': ' Tidak ',
            ' ekspetasi ': ' harapan ', ' ko ': ' mengapa ', ' ajg ': ' anjing ', ' kok ': ' mengapa ',
            ' trasaksi ': ' transaksi ', ' utk ': ' untuk ', ' berkalikali ': ' berulang ', ' mlh ': ' malah ',
            ' pdhl ': ' padahal', ' mkanan ': ' makanan ', ' ad ': ' ada ', ' yng ': ' yang ', ' tidk ': ' tidak ',
            ' sbg ': ' sebagai ', ' teruss ': ' terus ', ' sekarng  ': ' sekarang ', ' kpd ': ' kepada ',
            ' gimna ': ' bagaimana ', ' mngoprssikam ': ' mengoperasikan ', ' mlkukan ': ' melakukan ',
            ' prjlanan ': ' perjalanan ', ' puter ': ' putar ', ' muter ': ' putar ', ' tp ': ' tetapi ',
            ' dlm ': ' dalam ', ' byak ': ' banyak ', ' hrus ': ' harus ', ' msih ': ' masih ', ' tdak ': ' tidak ',
            ' sring ': ' sering ', ' tsb ': ' tersebut ', ' smngkin ': ' semoga ', ' buat ': ' untuk ',
            ' dlu ': ' dulu ', ' blkg ': ' belakang ', ' rjl ': ' perjalanan ', ' knjng ': ' keinginan ',
            ' dg ': ' dengan ', ' smsn ': ' semesta ', ' bln ': ' bulan ', ' lgsg ': ' langsung ',
            ' mls ': ' malas ', ' jg ': ' juga ', ' mw ': ' mau ', ' nyok ': ' ayo ', ' bjkr ': ' bicara ',
            ' pokok ': ' pokoknya ', ' ayok ': ' ayo ', ' grik ': ' berlari ', ' lsg ': ' langsung ',
            ' ngasih ': ' memberi ', ' ngantor ': ' ke kantor ', ' gb ': ' tidak perlu ',
            ' rekaman ': ' merekam ', ' dr ': ' dari ', ' bsk ': ' besok ', ' skrg ': ' sekarang ', 
            ' udah ': ' sudah', ' tmpt ': ' tempat ', ' maksud ': ' tujuan', ' dan ': ' dan', 
            ' ada ': ' ada', ' untuk ': ' untuk', 'sy' : 'saya', 'sprt' : 'seperti'}

    df_preprocessed['content'] = df_preprocessed['content'].apply(lambda x: normalisasi(x, norm))
    st.write("Data setelah normalisasi:")
    st.dataframe(df_preprocessed[['content']].tail())

    # LABELLING
    st.subheader("Labelling Data")
    lexicon_positive = dict()
    with open("kamus_indonesia_sentiment_lexicon/new_positive.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) >= 2:  # pastikan setiap baris memiliki minimal dua kolom
                lexicon_positive[row[0]] = int(row[1])

    lexicon_negative = dict()
    with open("kamus_indonesia_sentiment_lexicon/new_negative.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) >= 2:  # pastikan setiap baris memiliki minimal dua kolom
                lexicon_negative[row[0]] = int(row[1])

    # Fungsi untuk menentukan polaritas sentimen
    def sentiment_analysis_lexicon_indonesia(text):
        score = 0
        words = text.split()

        for word in words:
            if word in lexicon_positive:
                score += lexicon_positive[word]
        for word in words:
            if word in lexicon_negative:
                score += lexicon_negative[word]

        polarity = ''
        if score > 0:
            polarity = 'POSITIVE'
        elif score < 0:
            polarity = 'NEGATIVE'
        else:
            polarity = 'NEUTRAL'
        return score, polarity

    # Terapkan analisis sentimen pada kolom 'content'
    results = df_preprocessed['content'].apply(sentiment_analysis_lexicon_indonesia)
    results = list(zip(*results))
    df_preprocessed['polarity_score'] = results[0]
    df_preprocessed['sentiment'] = results[1]

    st.write("Data setelah pelabelan:")
    st.write(df_preprocessed['sentiment'].value_counts())
    st.dataframe(df_preprocessed[['content', 'polarity_score', 'sentiment']].tail())

    # Hapus sentimen netral
    st.subheader("Hapus Sentimen Netral")
    df_preprocessed = df_preprocessed[df_preprocessed['sentiment'] != 'NEUTRAL']
    st.write("Data setelah menghapus sentimen netral:")
    st.dataframe(df_preprocessed[['content', 'sentiment']].tail())

    # Stopword removal
    st.subheader("Hapus Stopword")
    stop_factory = StopWordRemoverFactory()
    stop_words = stop_factory.get_stop_words()
    df_preprocessed['content_stopwords'] = df_preprocessed['content'].apply(lambda x: stopword_removal(x, stop_words))
    st.write("Data setelah menghapus stopword:")
    st.dataframe(df_preprocessed[['content_stopwords', 'sentiment']].tail())

   # Stemming
    st.subheader("Stemming")
    df_preprocessed['content_stemmed'] = df_preprocessed['content_stopwords'].apply(lambda x: stemming(x))
    st.write("Data setelah stemming:")
    st.dataframe(df_preprocessed[['content_stemmed', 'sentiment']].tail())

    # Tokenizing dan menyimpan hasil preprocessing akhir
    st.subheader("Tokenizing dan Stemming")
    df_preprocessed['cleaned_tokenized_stemmed'] = df_preprocessed['content_stemmed'].apply(lambda x: tokenize(x))
    st.write("Data setelah tokenizing dan stemming:")
    st.dataframe(df_preprocessed[['cleaned_tokenized_stemmed', 'sentiment']].tail())

    # Menambahkan kolom baru yang menggabungkan token menjadi kalimat
    st.subheader("Hasil Tokenisasi dan Stemming dalam Bentuk Kalimat")
    df_preprocessed['cleaned_sentence'] = df_preprocessed['cleaned_tokenized_stemmed'].apply(lambda tokens: ' '.join(tokens))
    st.write("Data setelah tokenizing dan stemming dalam bentuk kalimat:")
    st.dataframe(df_preprocessed[['cleaned_sentence', 'sentiment']].tail())

    # Simpan hasil ke session state
    st.session_state['uploaded_df'] = df_preprocessed
    # Simpan hasil preprocess ke CSV
    csv_file = 'hasil_preprocessing.csv'
    df_preprocessed.to_csv(csv_file, index=False)  # Simpan file CSV

# Fungsi untuk menampilkan visualisasi
import io  # untuk menyimpan gambar dalam memory

# Fungsi untuk visualisasi data
def show_visualization():
    st.title("Visualisasi Data")

    # Cek apakah data sudah dimuat ke dalam session state
    if 'uploaded_df' not in st.session_state:
        st.error("Data belum diupload atau belum diproses. Silakan lakukan preprocessing terlebih dahulu.")
        return

    # Ambil data yang telah diproses dari session state
    df = st.session_state['uploaded_df']

    if 'cleaned_tokenized_stemmed' not in df.columns:
        st.error("Kolom 'cleaned_tokenized_stemmed' tidak ditemukan. Pastikan telah dilakukan preprocessing.")
        return

    if df is not None and 'content' in df.columns:
        # Menampilkan informasi data
        st.write("Data yang digunakan untuk visualisasi:")
        st.dataframe(df.tail())

        # Kode untuk mengekspor ke CSV
        export_path = 'preprocessed/preprocessed_data_gmaps2.csv'
        os.makedirs(os.path.dirname(export_path), exist_ok=True)  # Membuat folder jika belum ada
        df.to_csv(export_path, index=False)

        # Tampilkan jumlah ulasan dengan sentimen positif dan negatif
        st.subheader("Jumlah ulasan dengan sentimen positif dan negatif")
        if 'sentiment' in df.columns:  # pastikan kolom 'sentiment' ada
            sentiment_counts = df['sentiment'].value_counts()
            st.write(sentiment_counts)

            # Visualisasi Distribusi Sentimen
            st.subheader("Sentiment Distribution")
            palette = {'positif': 'skyblue', 'NEGATIVE': 'coral', 'POSITIVE': 'lightgreen'}  # Menambahkan warna untuk 'POSITIVE'
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.countplot(x='sentiment', data=df, palette=palette, ax=ax)
            ax.set_title('Sentiment Distribution')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Sentiment Count')
            st.pyplot(fig)

            # Simpan plot distribusi sentimen sebagai gambar
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="Download Distribusi Sentimen sebagai PNG",
                data=buf,
                file_name="distribusi_sentimen.png",
                mime="image/png"
            )

            # WORDCLOUD
            st.subheader("Wordcloud")

            data_negatif = df[df['sentiment'] == 'NEGATIVE']
            data_positif = df[df['sentiment'] == 'POSITIVE']  # Ubah dari 'positif' ke 'POSITIVE'

            if not data_positif.empty:
                all_text_s1 = ' '.join(str(word) for word in data_positif['content'])  # Ganti dengan kolom yang benar
                wordcloud_positif = WordCloud(colormap='Blues', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s1)
                fig_positif, ax_positif = plt.subplots(figsize=(10, 10))
                ax_positif.imshow(wordcloud_positif, interpolation='bilinear')
                ax_positif.axis('off')
                ax_positif.set_title('Positive Sentiment Visualization', color='black')
                st.pyplot(fig_positif)

                # Simpan wordcloud positif sebagai gambar
                buf_positif = io.BytesIO()
                fig_positif.savefig(buf_positif, format="png", dpi=300)
                buf_positif.seek(0)
                st.download_button(
                    label="Download Wordcloud Positif sebagai PNG",
                    data=buf_positif,
                    file_name="wordcloud_positif.png",
                    mime="image/png"
                )

            if not data_negatif.empty:
                all_text_s0 = ' '.join(str(word) for word in data_negatif['content'])  # Ganti dengan kolom yang benar
                wordcloud_negatif = WordCloud(colormap='Reds', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s0)
                fig_negatif, ax_negatif = plt.subplots(figsize=(10, 10))
                ax_negatif.imshow(wordcloud_negatif, interpolation='bilinear')
                ax_negatif.axis('off')
                ax_negatif.set_title('Negative Sentiment Visualization', color='black')
                st.pyplot(fig_negatif)

                # Simpan wordcloud negatif sebagai gambar
                buf_negatif = io.BytesIO()
                fig_negatif.savefig(buf_negatif, format="png", dpi=300)
                buf_negatif.seek(0)
                st.download_button(
                    label="Download Wordcloud Negatif sebagai PNG",
                    data=buf_negatif,
                    file_name="wordcloud_negatif.png",
                    mime="image/png"
                )
        else:
            st.error("Kolom 'sentiment' tidak ditemukan dalam DataFrame.")
    else:
        st.error("Data tidak tersedia untuk visualisasi.")

# Fungsi untuk splitting data
def show_splitting_data():
    st.title("Splitting Data")

    if 'uploaded_df' not in st.session_state:
        st.error("Data belum diupload. Silakan upload data terlebih dahulu.")
        return

    df_split = st.session_state['uploaded_df'].copy()
    if 'cleaned_tokenized_stemmed' not in df_split.columns:
        st.error("Kolom 'cleaned_tokenized_stemmed' tidak ditemukan. Pastikan telah dilakukan preprocessing.")
        return

    X = df_split['cleaned_tokenized_stemmed'].apply(lambda x: ' '.join(x))  # Gabungkan kembali token
    y = df_split['sentiment']

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Data Train")
    st.dataframe(x_train)
    st.write(f"Data Train sebanyak: {x_train.count()}")

    st.subheader("Data Test")
    st.dataframe(x_test)
    st.write(f"Data Test sebanyak: {x_test.count()}")

    # Training data dengan SVM
    st.header("Training Data dengan SVM")
    tvec = TfidfVectorizer()
    clf = svm.SVC(kernel="rbf")
    model = Pipeline([('vectorizer', tvec), ('classifier', clf)])
    model.fit(x_train, y_train)
    st.write(f"Akurasi model: {model.score(x_test, y_test)}")

    hasil = model.predict(x_test)
    matrix = classification_report(y_test, hasil)
    st.write(f"Classification Report:\n {matrix}")

    # Hyperparameter Tuning
    st.header("Hyperparameter Tuning")
    parameters = {
        'vectorizer__max_df': [0.8, 0.9, 1.0],
        'vectorizer__min_df': [2, 5, 10],
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(model, parameters, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)
    st.write("Best Parameters:", grid_search.best_params_)
    st.write("Best Score:", grid_search.best_score_)

    # Simpan model
    joblib.dump(grid_search, 'model_gmaps.pkl')
    
    y_pred = grid_search.predict(x_test)
    st.write(f"Classtification Report : \n {classification_report(y_test, y_pred)}")

    #confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    st.write(f"Confusion Matrix")
    st.write(f"True Positive : {tp}")
    st.write(f"False Positive : {fp}")
    st.write(f"True Negative : {tn}")
    st.write(f"False Negative : {fn}")
    # END EVALUASI CONFUSION MATRIX

  
def show_lda_modeling():
    st.title("Pemodelan Topik LDA")

    # Cek apakah data sudah dimuat ke dalam session state
    if 'uploaded_df' not in st.session_state:
        st.error("Data belum diupload atau belum diproses. Silakan lakukan preprocessing terlebih dahulu.")
        return

    # Ambil data yang telah diproses dari session state
    df = st.session_state['uploaded_df']

    if 'cleaned_tokenized_stemmed' not in df.columns:
        st.error("Kolom 'cleaned_tokenized_stemmed' tidak ditemukan. Pastikan telah dilakukan preprocessing.")
        return

    # Memproses data ulasan
    texts = df['cleaned_tokenized_stemmed'].tolist()  # Pastikan ini adalah list of list (tokenized words)
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Transformasi teks ke dalam bentuk vektor menggunakan CountVectorizer
    tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                        lowercase=True,
                                        token_pattern=r'\b[a-zA-Z]{3,}\b',  # kata dengan lebih dari 3 karakter
                                        max_df=0.9,  # menghapus kata yang muncul di lebih dari 90% ulasan
                                        min_df=10)  # menghapus kata yang muncul di kurang dari 10 ulasan

    # Apply transformation ke dalam bentuk tf-idf
    dtm_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_tokenized_stemmed'].apply(lambda x: ' '.join(x)))

    st.write(f"Shape dari tfidf: {dtm_tfidf.shape}, artinya ada {dtm_tfidf.shape[0]} ulasan dan {dtm_tfidf.shape[1]} tokens yang terbentuk.")

    # Langkah 2: GridSearch & tuning parameter untuk menemukan model LDA yang optimal
    search_params = {'n_components': [5, 10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

    # Inisialisasi model LDA dan Grid Search
    lda = LatentDirichletAllocation()
    model_lda = GridSearchCV(lda, param_grid=search_params)
    model_lda.fit(dtm_tfidf)

    # Menampilkan hasil model terbaik
    best_lda_model = model_lda.best_estimator_
    st.write("Best Model's Params: ", model_lda.best_params_)
    st.write("Model Log Likelihood Score: ", model_lda.best_score_)
    st.write("Model Perplexity: ", best_lda_model.perplexity(dtm_tfidf))

    # Langkah 4: Membandingkan LDA Model Performance Scores
    gscore = model_lda.cv_results_
    n_topics = [5, 10, 15, 20, 25, 30]

    log_likelyhoods_5 = []
    log_likelyhoods_7 = []
    log_likelyhoods_9 = []

    for i, params in enumerate(gscore['params']):
        if params['learning_decay'] == 0.5:
            log_likelyhoods_5.append(gscore['mean_test_score'][i])
        elif params['learning_decay'] == 0.7:
            log_likelyhoods_7.append(gscore['mean_test_score'][i])
        elif params['learning_decay'] == 0.9:
            log_likelyhoods_9.append(gscore['mean_test_score'][i])

    # Mengambil nilai-nilai n_topics yang sesuai
    log_likelyhoods_5 = log_likelyhoods_5[:len(n_topics)]
    log_likelyhoods_7 = log_likelyhoods_7[:len(n_topics)]
    log_likelyhoods_9 = log_likelyhoods_9[:len(n_topics)]

    # Tampilkan grafis performa LDA model
    st.subheader("Grafik Performa LDA Model")
    plt.figure(figsize=(12, 8))

    # Pastikan panjang yang sesuai sebelum memplot
    if len(log_likelyhoods_5) == len(n_topics):
        plt.plot(n_topics, log_likelyhoods_5, label='Learning decay 0.5')
    if len(log_likelyhoods_7) == len(n_topics):
        plt.plot(n_topics, log_likelyhoods_7, label='Learning decay 0.7')
    if len(log_likelyhoods_9) == len(n_topics):
        plt.plot(n_topics, log_likelyhoods_9, label='Learning decay 0.9')

    plt.title("Memilih Model LDA yang Optimal")
    plt.xlabel("Jumlah Topik")
    plt.ylabel("Log Likelihood Scores")
    plt.legend(loc='best')
    st.pyplot(plt)

    # Perhitungan nilai koherensi
    st.subheader("Grafik Koherensi Topik")
    start, limit, step = 2, 30, 3
    coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=texts, start=start, limit=limit, step=step)

    # Plot koherensi
    x = range(start, limit, step)
    plt.figure(figsize=(10, 6))
    plt.plot(x, coherence_values)
    plt.xlabel("Jumlah Topik")
    plt.ylabel("Koherensi")
    plt.title("Grafik Nilai Koherensi untuk Berbagai Jumlah Topik")
    plt.xticks(x)
    st.pyplot(plt)

    # Menampilkan tabel nilai koherensi
    st.subheader("Tabel Nilai Koherensi")
    coherence_table = pd.DataFrame({'Jumlah Topik': x, 'Nilai Koherensi': coherence_values})
    st.dataframe(coherence_table)

    # Visualisasi topik menggunakan pyLDAvis
    lda_vis = pyLDAvis.lda_model.prepare(best_lda_model, dtm_tfidf, tfidf_vectorizer)
    html_string = pyLDAvis.prepared_data_to_html(lda_vis)

    # Tentukan nama file HTML dan pastikan direktori tujuan ada
    html_file_path = "visualisasi_topik.html"
    
    # Simpan visualisasi topik sebagai file HTML
    with open(html_file_path, "w") as f:
        f.write(html_string)

    # Tampilkan visualisasi di Streamlit
    st.components.v1.html(html_string, width=1300, height=800)

    # Berikan opsi untuk mengunduh file HTML
    st.subheader("Unduh Visualisasi Topik")
    html_bytes = io.BytesIO(html_string.encode('utf-8'))
    st.download_button(
        label="Download Visualisasi Topik sebagai HTML",
        data=html_bytes,
        file_name="visualisasi_topik.html",
        mime="text/html"
    )

    # Konversi file HTML ke gambar PNG menggunakan html2image
    hti = Html2Image()
    
    # Pastikan file HTML sudah ada sebelum mengonversi
    if os.path.exists(html_file_path):
        hti.screenshot(html_file=html_file_path, save_as='visualisasi_topik_hd.png', size=(1920, 1080))

        # Tampilkan gambar PNG yang sudah dihasilkan di Streamlit
        st.image("visualisasi_topik_hd.png", caption="Visualisasi Topik HD", use_column_width=True)

        # Berikan opsi untuk mengunduh gambar PNG
        with open("visualisasi_topik_hd.png", "rb") as img_file:
            st.download_button(
                label="Download Gambar Visualisasi Topik HD",
                data=img_file,
                file_name="visualisasi_topik_hd.png",
                mime="image/png"
            )
    else:
        st.error("File HTML visualisasi topik tidak ditemukan.")

    # Tambahkan kode baru di bawah ini untuk menampilkan probabilitas kata pada topik
    # Lakukan pemodelan LDA
    lda_model = LatentDirichletAllocation(n_components=5)  # Atur jumlah topik sesuai kebutuhan
    lda_model.fit(dtm_tfidf)

    # Mendapatkan topik dan probabilitas
    st.subheader("Probabilitas Kata pada Topik Ulasan")

    if dtm_tfidf is not None:
        topic_probabilities = get_topic_probabilities(lda_model, dtm_tfidf)

        results = []
        seen_words = set()  # Set untuk melacak kata-kata yang sudah ditampilkan

        # Dapatkan jumlah topik yang tersedia di model LDA
        n_topics = lda_model.components_.shape[0]  # Komponen di model LDA

        # Loop melalui setiap ulasan dan distribusi probabilitas topiknya
        for idx, prob_dist in enumerate(topic_probabilities):
            # Loop untuk setiap topik, dan simpan hanya yang punya probabilitas signifikan (>0.1)
            for topic_num, prob in enumerate(prob_dist):
                if topic_num < n_topics:  # Pastikan topic_num tidak melebihi jumlah topik
                    if prob > 0.1:  # Hanya topik dengan probabilitas lebih dari 0.1
                        # Mendapatkan kata-kata dan nilai probabilitas dari topik tersebut
                        topic = lda_model.components_[topic_num]
                        top_words_with_prob = [f"{round(topic[i], 4)}*{tfidf_vectorizer.get_feature_names_out()[i]}" 
                                               for i in topic.argsort()[:-21:-1]]  # Ambil 20 kata dengan probabilitas tertinggi
                        
                        # Gabungkan hasil ke dalam format (probabilitas*kata)
                        formatted_top_words = ', '.join(top_words_with_prob)

                        # Cek apakah kata-kata ini sudah ditampilkan sebelumnya
                        if formatted_top_words not in seen_words:
                            # Tambahkan ke set jika belum ada
                            seen_words.add(formatted_top_words)

                            # Menyimpan nomor topik, kata dengan probabilitas, dan probabilitas topik
                            results.append({
                                'Nomor Topik': f'Topik {topic_num + 1}',  # Nomor topik
                                'Kata dengan Probabilitas': formatted_top_words  # Kata dan probabilitas
                            })

        # Mengonversi hasil ke dalam DataFrame
        topics_df = pd.DataFrame(results)

        # Menampilkan DataFrame di Streamlit
        st.dataframe(topics_df)  # Menampilkan tabel

    # Mendapatkan topik dan probabilitas
    st.subheader("Topik Ulasan")

    if dtm_tfidf is not None:
        topic_probabilities = get_topic_probabilities(lda_model, dtm_tfidf)

        # Membuat list untuk menyimpan hasil
        results = []
        for idx, prob_dist in enumerate(topic_probabilities):
            topic_num = prob_dist.argmax()  # Mendapatkan nomor topik dengan probabilitas tertinggi
            prob = prob_dist[topic_num]  # Mendapatkan probabilitas dari topik tersebut
            results.append({
                'Ulasan': df['cleaned_sentence'].iloc[idx],  # Ambil ulasan bersih
                'Nomor Topik': f'Topik {topic_num + 1}',  # Nomor topik
                'Probabilitas': prob  # Probabilitas
            })

        # Mengonversi hasil ke dalam DataFrame
        topics_df = pd.DataFrame(results)

        # Menampilkan DataFrame di Streamlit
        st.dataframe(topics_df)  # Menampilkan tabel 

def compute_coherence_values(dictionary, corpus, texts, start, limit, step):
    coherence_values = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return coherence_values

def get_topic_probabilities(model, dtm):
    # Mendapatkan probabilitas topik dari model LDA
    return model.transform(dtm)


if __name__ == "__main__":
    main()
