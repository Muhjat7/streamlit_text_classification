# =====================================================
# app.py — LexaClass (Lexicon-based + Training)
# Polarity:
# 0 = senang
# 1 = emosi
# 2 = netral (fallback)
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from preprocess import preprocess_texts

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="LexaClass — Gas Klasifikasi Teks"
)

st.title("🔥 LexaClass")
st.caption("Ngasih label emosi & latih model teks dalam sekali jalan 🚀")


# =====================================================
# SESSION HELPERS
# =====================================================
def get_df():
    return st.session_state.get("df")

def save_df(df):
    st.session_state["df"] = df

def set_uploaded_name(name):
    st.session_state["uploaded_name"] = name

def get_uploaded_name():
    return st.session_state.get("uploaded_name")

# =====================================================
# LEXICON (NETRAL TIDAK ADA)
# =====================================================
LEXICON_SENANG = [
    "senang","bahagia","gembira","suka","puas","bangga","salut",
    "keren","mantap","bagus","baik","hebat","luar biasa",
    "top","recommended","makasih","terima kasih","jos",
    "mantul","setuju","valid","worth it","inspiratif", "senang","bahagia","gembira","suka","puas","bangga","salut","keren","mantap",
    "bagus","baik","hebat","luar biasa","top","oke","ok","recommended",
    "terima kasih","makasih","makasi","thanks","thank you",
    "asik","asyik","enak","nikmat","memuaskan","kagum","terharu",
    "setuju","valid","mantul","jos","kece","cakep","epic","amazing",
    "brilian","cerdas","kreatif","keren banget","mantap jiwa",
    "suka banget","seneng","happy","love","lovely","perfect",
    "pas","cocok","berhasil","sukses","bermanfaat","berguna",
    "worth it","puas banget","senang sekali",
    "good","great","excellent","nice","cool","solid",
    "menarik","inspiratif","menginspirasi","respect","respek","thumbs up",
    "terbaik","the best","luar biasa sekali","healing","adem",
    "nyaman","tenang","lega","bangga banget","memikat","recommended banget"
]

LEXICON_EMOSI = [
     # marah / agresif
    "marah","kesal","emosi","muak","benci","parah","kacau","rusak",
    "jelek","buruk","busuk","menyebalkan","menjengkelkan",
    "goblok","tolol","bodoh","bangsat","brengsek","bajingan",
    "anjing","tai","sial","kampret","kejam","ngeselin","nyebelin",
    "emosi banget","kesel","payah","lemah","konyol","absurd","ngaco",
    "gagal","gagal total","memalukan","tidak becus","tidak kompeten",
    "tidak berguna","bohong","penipu","palsu","curang","ngawur",
    "asal-asalan","songong","angkuh","jahat","keji","tidak adil",

    # sedih / kehilangan
    "sedih","kecewa","kecewa berat","sangat kecewa","pilu","duka",
    "berduka","kehilangan","menyedihkan","tragis","miris",
    "kasihan","kasihan banget","prihatin","nangis","menangis",
    "air mata","terpuruk","down","hancur","remuk","patah hati",
    "tersakiti","sakit hati","nelangsa","putus asa","galau",
    "sunyi","sepi","sendu","haru","menyesal","penyesalan",
    "duka cita","ikut sedih","menderita","derita","perih","luka batin"
]

LEXICON_TAKUT = {
    "takut", "ketakutan", "menakutkan", "ngeri", "seram", "horor",
    "cemas", "khawatir", "waswas", "gelisah", "panik", "deg-degan",
    "ancam", "ancaman", "terancam", "bahaya", "berbahaya",
    "ngancem", "mengancam",
    "parno", "paranoid",
    "stress", "stres", "tertekan",
    "bingung", "resah",
    "merinding", "trauma",
    "tidak aman", "ga aman", "gak aman",
    "ngeri banget", "takut banget",
    "mati", "dibunuh", "dibantai"
}


# =====================================================
# RULE-BASED LABELING (FIXED)
# =====================================================
def rule_label(text: str) -> int:
    t = (text or "").lower()

    score_senang = sum(w in t for w in LEXICON_SENANG)
    score_emosi  = sum(w in t for w in LEXICON_EMOSI)
    score_takut  = sum(w in t for w in LEXICON_TAKUT)

    # PRIORITAS: EMOSI KUAT > TAKUT > SENANG
    if score_emosi > 0 and score_emosi >= score_takut and score_emosi >= score_senang:
        return 1   # emosi
    elif score_takut > 0 and score_takut >= score_senang:
        return 2   # takut
    elif score_senang > 0:
        return 0   # senang
    else:
        return 2   # fallback = takut (tidak ada netral)

# =====================================================
# 1) UPLOAD CSV
# =====================================================
st.header("1️⃣ Upload CSV  Lo!!! — Gas Masukin Data 🚀")

uploaded = st.file_uploader(
    "Upload CSV (wajib ada kolom `text`)",
    type=["csv"]
)

if uploaded is not None:
    if uploaded.name != get_uploaded_name():
        df = pd.read_csv(uploaded)

        if "text" not in df.columns:
            st.error("CSV harus memiliki kolom `text`.")
            st.stop()

        st.info("Preprocessing teks...")
        df["cleaned_text"] = preprocess_texts(
            df["text"].astype(str).tolist()
        )

        save_df(df)
        set_uploaded_name(uploaded.name)
        st.success("Dataset dimuat & diproses.")
else:
    if get_df() is None:
        st.stop()

df = get_df()

st.subheader("Preview Dataset")
st.dataframe(df.head())
st.markdown("---")

# =====================================================
# 2) AUTO-LABEL (LEXICON)
# =====================================================
st.header("2️⃣ Auto-Labeling (Berbasis Kamus Kata)")

st.caption("Klik tombol di bawah buat ngasih label otomatis ke data teks lo 👇")

if st.button("🔥 Gas Auto-Label"):
    df_local = df.copy()
    df_local["polarity"] = df_local["text"].astype(str).apply(rule_label)
    save_df(df_local)

    st.success("✅ Auto-label kelar! Data siap dipakai.")
    st.dataframe(df_local[["text", "polarity"]].head(10))

df = get_df()


# =====================================================
# DISTRIBUSI LABEL
# =====================================================
# =====================================================
# DISTRIBUSI LABEL (TAMPIL SETELAH AUTO-LABEL)
# =====================================================
st.subheader("📊 Sebaran Emosi Teks")

if "polarity" not in df.columns:
    st.info("Belum ada label. Jalankan auto-label dulu biar datanya kebaca 👆")
else:
    label_map = {
        0: "senang 😄",
        1: "emosi 😡",
        2: "takut 😨"
    }

    counts = df["polarity"].map(label_map).value_counts()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)

    st.markdown("### 📌 Ringkasan Jumlah Data")
    st.dataframe(counts.rename("jumlah data"))

# =====================================================
# 3) TRAIN MODEL
# =====================================================
st.header("3️⃣ Gas Training Model 🚀")

model_choice = st.radio(
    "Pilih model:",
    ("Naive Bayes", "Logistic Regression", "SVM (Linear)")
)

test_size_pct = st.slider("Test size (%)", 5, 40, 20)

if st.button("Train Model"):
    X = df["cleaned_text"].astype(str).tolist()
    y = df["polarity"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_pct / 100,
        random_state=42
    )

    st.write("Distribusi label TRAIN:")
    st.write(pd.Series(y_train).value_counts())

    if model_choice == "Naive Bayes":
        clf = MultinomialNB()
        model_name = "nb_model.joblib"
    elif model_choice == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000)
        model_name = "lr_model.joblib"
    else:
        clf = LinearSVC(max_iter=10000)
        model_name = "svm_model.joblib"

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", clf)
    ])

    with st.spinner("Training model..."):
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )

    joblib.dump(pipe, model_name)

    st.session_state["train_results"] = {
        "model": model_choice,
        "f1": f1,
        "report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["senang", "emosi", "netral"],
            output_dict=True
        ),
        "cm": cm
    }

    st.success(
        # f"Training selesai — F1 (macro): {f1:.4f} | "
        f"Model disimpan sebagai `{model_name}`"
    )

st.markdown("---")

# =====================================================
# 4) HASIL TRAINING
# =====================================================
st.header("4️⃣ Hasil Training — Gas Lihat Skornya 🚀")

if "train_results" in st.session_state:
    res = st.session_state["train_results"]

    st.write(f"Model: **{res['model']}**")
    # st.write(f"F1 (macro): **{res['f1']:.4f}**")

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(res["report"]).transpose().round(4))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(res["cm"], cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["senang", "emosi", "netral"])
    ax.set_yticklabels(["senang", "emosi", "netral"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, res["cm"][i, j],
                    ha="center", va="center", color="black")

    st.pyplot(fig)
