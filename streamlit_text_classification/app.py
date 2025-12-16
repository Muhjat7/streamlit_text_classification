# app.py (replace your existing file with this)
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

from preprocess import preprocess_texts  # your preprocessing function

# Page
st.set_page_config(page_title="LexaClass: Text Classification Toolkit")

# -------------------------
# Helpers: single source of truth = st.session_state["df"]
# -------------------------
def get_df():
    return st.session_state.get("df", None)

def save_df(df):
    st.session_state["df"] = df

def set_uploaded_name(name: str):
    st.session_state["uploaded_name"] = name

def get_uploaded_name():
    return st.session_state.get("uploaded_name", None)

def mark_labeled(flag: bool = True):
    st.session_state["is_labeled"] = bool(flag)

def is_labeled():
    return st.session_state.get("is_labeled", False)

# -------------------------
# 1) Upload CSV (only load if new file)
# -------------------------
st.header("1) Upload CSV")
uploaded = st.file_uploader("Upload CSV (wajib ada kolom `text`; `label` optional)", type=["csv"])

if uploaded is not None:
    # Only LOAD if this is a new upload (prevent overwriting session df on reruns)
    if uploaded.name != get_uploaded_name():
        try:
            df_uploaded = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            st.stop()

        if "text" not in df_uploaded.columns:
            st.error("CSV harus memiliki kolom `text`. Hentikan.")
            st.stop()

        st.info("Menjalankan preprocessing (cleaning)...")
        df_uploaded["cleaned_text"] = preprocess_texts(df_uploaded["text"].astype(str).tolist())

        # save and record uploaded filename
        save_df(df_uploaded)
        set_uploaded_name(uploaded.name)

        # set labeled flag if file actually has label values
        if "label" in df_uploaded.columns and df_uploaded["label"].astype(str).str.strip().ne("").any():
            mark_labeled(True)
        else:
            mark_labeled(False)

        st.success(f"File '{uploaded.name}' dimuat dan disimpan di session. Preprocessing selesai.")
    else:
        st.info(f"File '{uploaded.name}' sudah dimuat sebelumnya — tidak menimpa session.")
else:
    if get_df() is None:
        st.info("Belum ada file diupload.")
    else:
        st.info(f"Dataset dari `{get_uploaded_name()}` sudah tersedia di session. (Upload file baru untuk mengganti.)")

# ensure df is available from session
df = get_df()
if df is None:
    st.stop()

# show top rows
st.subheader("Preview dataset (5 rows)")
st.dataframe(df.head(5))
st.markdown("---")

# -------------------------
# 2) Auto-label (rule-based) & save (single action)
# -------------------------
st.header("2) Auto-label (rule-based) & Simpan label (1 tombol)")

def rule_label(text: str) -> str:
    t = (text or "").lower()
    pos = ["bagus","baik","suka","love","mantap","recommended","ok","oke","senang","favorit"]
    neg = ["jelek","buruk","benci","hate","tidak suka","ga suka","gak suka","sad","jelek banget"]
    score = 0
    for w in pos:
        if w in t:
            score += 1
    for w in neg:
        if w in t:
            score -= 1
    # rough negation handling
    if any(n in t for n in ["tidak ", "bukan ", "gak ", "ga "]):
        score -= 1
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

# show status
st.write(f"- Rows: {len(df)}")
st.write(f"- Has column 'label'? {'label' in df.columns and df['label'].astype(str).str.strip().ne('').any()}")

# Single-button action: auto-label and save directly to session df
if st.button("🚀 Auto-label dan Simpan (langsung ke kolom 'label')"):
    df_local = get_df().copy()
    df_local["label"] = df_local["text"].astype(str).apply(rule_label)
    save_df(df_local)
    mark_labeled(True)
    st.success("Auto-label selesai dan disimpan ke kolom `label`.")
    st.dataframe(df_local[["text","label"]].head(10))

# show distribution if label exists
df = get_df()  # refresh local ref
if "label" in df.columns and df["label"].astype(str).str.strip().ne("").any():
    st.subheader("Distribusi label (ground truth / auto-generated)")
    counts = df["label"].astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(5,4))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)
else:
    st.info("Kolom `label` belum tersedia. Jalankan Auto-label jika ingin membuat label otomatis.")
st.markdown("---")

# DEBUG expander: show session keys and df head to verify
with st.expander("DEBUG: session_state keys & df head"):
    st.write("session_state keys:", list(st.session_state.keys()))
    tmp = get_df()
    st.write("df.head():")
    st.dataframe(tmp.head(5) if tmp is not None else "No df in session")

# -------------------------
# 3) Train model (reads from session)
# -------------------------
st.header("3) Pilih model & Train")

# re-fetch latest df
df = get_df()

if "label" not in df.columns or df["label"].astype(str).str.strip().eq("").all():
    st.info("Training dinonaktifkan — dataset belum memiliki kolom `label`. Jalankan Auto-label terlebih dahulu atau upload dataset yang berlabel.")
else:
    model_choice = st.radio("Pilih model:", ("Naive Bayes", "Logistic Regression", "SVM (Linear)"))
    test_size_pct = st.slider("Test size (%)", 5, 50, 20)

    if st.button("Train model"):
        df_train = get_df()
        X = df_train["cleaned_text"].astype(str).tolist()
        y = df_train["label"].astype(str).tolist()

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_pct/100.0, stratify=y, random_state=42)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_pct/100.0, random_state=42)

        if model_choice == "Naive Bayes":
            pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])
        elif model_choice == "Logistic Regression":
            pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
        else:
            pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC(max_iter=10000))])

        with st.spinner("Training..."):
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        try:
            joblib.dump(pipe, "model.joblib")
        except Exception as e:
            st.warning(f"Gagal menyimpan model: {e}")

        st.success(f"Training selesai. Model disimpan sebagai model.joblib  — F1 (macro) = {f1:.4f}")

        st.session_state["train_results"] = {
            "model_name": model_choice,
            "f1": f1,
            "report": report,
            "confusion_matrix": cm,
            "y_test": list(y_test),
            "y_pred": list(y_pred)
        }

        st.write("**Ringkasan:**")
        st.write(f"- Model: **{model_choice}**")
        st.write(f"- F1 (macro): **{f1:.4f}**")
        st.dataframe(pd.DataFrame(cm))

st.markdown("---")

# -------------------------
# 4) Prediksi & Pie chart (pred distribution on test)
# -------------------------
# st.header("4) Prediksi & Pie chart (opsional)")
# if "train_results" in st.session_state:
#     res = st.session_state["train_results"]
#     pred_counts = pd.Series(res["y_pred"]).value_counts()
#     fig2, ax2 = plt.subplots(figsize=(5,4))
#     ax2.pie(pred_counts.values, labels=pred_counts.index, autopct="%1.1f%%", startangle=90)
#     ax2.axis("equal")
#     st.pyplot(fig2)
# else:
#     st.info("Belum ada hasil training => belum ada distribusi prediksi.")

# st.markdown("---")

# -------------------------
# 5) Show training results
# -------------------------
st.header("5) Tampilkan hasil training")
if st.button("Tampilkan hasil training (F1, classification report, confusion matrix)"):
    if "train_results" not in st.session_state:
        st.error("Belum ada training.")
    else:
        res = st.session_state["train_results"]
        st.subheader(f"Hasil untuk model: {res['model_name']}")
        st.write(f"F1 (macro): **{res['f1']:.4f}**")

        rpt_df = pd.DataFrame(res["report"]).transpose()
        for c in rpt_df.columns:
            try:
                rpt_df[c] = pd.to_numeric(rpt_df[c], errors="coerce").round(4)
            except Exception:
                pass
        st.markdown("**Classification report**")
        st.dataframe(rpt_df)

        cm = np.array(res["confusion_matrix"])
        fig3, ax3 = plt.subplots(figsize=(5,4))
        ax3.imshow(cm, cmap="Blues")
        ax3.set_title("Confusion Matrix")
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")
        labels = sorted(list(set(res["y_test"])))
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        st.pyplot(fig3)
