# classify.py
import joblib
import json
import os
from typing import List, Tuple, Dict, Any

from preprocess import preprocess_texts

# sklearn imports for training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def normalize_label_map(label_map):
    if not label_map:
        return None
    normalized = {}
    for k, v in label_map.items():
        normalized[str(k)] = v
        try:
            ik = int(k)
            normalized[ik] = v
        except Exception:
            pass
    return normalized

def load_model_artifacts(model_path="model.joblib", label_map_path="label_map.json"):
    model, label_map = None, None

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            model = None

    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                label_map = json.load(f)
            label_map = normalize_label_map(label_map)
        except Exception as e:
            print(f"Failed to load label_map: {e}")
            label_map = None

    return model, label_map

def fallback_predict(texts: List[str]) -> Tuple[List[int], List[str]]:
    fallback_map = {
        0: "negative",
        1: "neutral",
        2: "positive",
        "0": "negative",
        "1": "neutral",
        "2": "positive"
    }
    preds = []
    for t in texts:
        tx = str(t).lower()
        score = 0
        if any(w in tx for w in ["good", "bagus", "great", "love", "senang", "suka"]):
            score += 1
        if any(w in tx for w in ["bad", "hate", "buruk", "jelek", "sad", "sedih"]):
            score -= 1
        if any(neg in tx for neg in [" tidak ", " bukan ", " ga ", " gak ", " never ", "n't"]):
            score = score - 1
        if score > 0:
            preds.append(2)
        elif score < 0:
            preds.append(0)
        else:
            preds.append(1)
    mapped = [fallback_map.get(str(p), str(p)) for p in preds]
    return preds, mapped

def map_label(label, label_map):
    if not label_map:
        return None
    if label in label_map:
        return label_map[label]
    if str(label) in label_map:
        return label_map[str(label)]
    try:
        if int(label) in label_map:
            return label_map[int(label)]
    except Exception:
        pass
    return str(label)

def predict_texts(texts: List[str], model, label_map, already_preprocessed: bool = False):
    """
    Predict texts. If already_preprocessed=True, assumes texts are cleaned strings ready for model.
    Otherwise preprocess_texts will be applied.
    """
    if not already_preprocessed:
        texts_proc = preprocess_texts(texts)
    else:
        texts_proc = texts

    if model is not None:
        try:
            preds = model.predict(texts_proc)
            # if model returns strings, try mapping; else convert to int if possible
            try:
                preds_int = [int(p) for p in preds]
                mapped = [map_label(p, label_map) for p in preds_int]
                return preds_int, mapped
            except Exception:
                # preds are likely labels as strings -> map directly if label_map provided
                mapped = [map_label(p, label_map) if label_map else p for p in preds]
                return preds, mapped
        except Exception as e:
            print(f"Model predict failed: {e}")
    return fallback_predict(texts_proc)

# ------------------- Training utilities -------------------
def train_and_evaluate(df, text_col='cleaned_text', label_col='label', test_size=0.2, random_state=42):
    """
    Train using GridSearchCV untuk LR & SVM,
    NB tetap standar (karena parameternya sedikit).
    Simpan model terbaik berdasarkan f1_macro.
    """

    from sklearn.model_selection import GridSearchCV

    # --- Prep data ---
    d = df[[text_col, label_col]].dropna()
    X = d[text_col].astype(str).tolist()
    y = d[label_col].astype(str).tolist()

    # label encoding
    unique_labels = sorted(set(y))
    label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
    index_to_label = {i: lab for lab, i in label_to_index.items()}
    y_int = [label_to_index[v] for v in y]

    # split
    stratify = y_int if len(unique_labels) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # ----------------------
    # Model 1: Naive Bayes
    # ----------------------
    pipe_nb = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])
    pipe_nb.fit(X_train, y_train)
    preds_nb = pipe_nb.predict(X_test)
    acc_nb = accuracy_score(y_test, preds_nb)
    f1_nb = f1_score(y_test, preds_nb, average="macro")

    # ----------------------
    # Model 2: Logistic Regression + GridSearchCV
    # ----------------------
    pipe_lr = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    param_lr = {
        "clf__C": [0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }

    grid_lr = GridSearchCV(pipe_lr, param_lr, n_jobs=-1, cv=3, scoring="f1_macro")
    grid_lr.fit(X_train, y_train)
    preds_lr = grid_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, preds_lr)
    f1_lr = f1_score(y_test, preds_lr, average="macro")

    # ----------------------
    # Model 3: SVM Linear + GridSearchCV
    # ----------------------
    pipe_svm = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])

    param_svm = {
        "clf__C": [0.1, 1, 5, 10]
    }

    grid_svm = GridSearchCV(pipe_svm, param_svm, n_jobs=-1, cv=3, scoring="f1_macro")
    grid_svm.fit(X_train, y_train)
    preds_svm = grid_svm.predict(X_test)
    acc_svm = accuracy_score(y_test, preds_svm)
    f1_svm = f1_score(y_test, preds_svm, average="macro")

    # -------------------------------------
    # Pilih model terbaik berdasarkan f1
    # -------------------------------------
    scores = {
        "nb": f1_nb,
        "lr": f1_lr,
        "svm": f1_svm
    }

    best_model_name = max(scores, key=scores.get)
    best_model = {
        "nb": pipe_nb,
        "lr": grid_lr.best_estimator_,
        "svm": grid_svm.best_estimator_
    }[best_model_name]

    # ---------------------------------------
    joblib.dump(pipe_nb, "model_nb.joblib")
    joblib.dump(grid_lr.best_estimator_, "model_lr.joblib")
    joblib.dump(grid_svm.best_estimator_, "model_svm.joblib")

    # save artifacts
    joblib.dump(best_model, "model.joblib")
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in index_to_label.items()}, f, indent=2)

    # returned metrics
    return {
        "best_model": best_model_name,
        "index_to_label": index_to_label,
        "metrics": {
            "nb": {"accuracy": acc_nb, "f1": f1_nb, "report": classification_report(y_test, preds_nb, output_dict=True)},
            "lr": {"accuracy": acc_lr, "f1": f1_lr, "report": classification_report(y_test, preds_lr, output_dict=True)},
            "svm": {"accuracy": acc_svm, "f1": f1_svm, "report": classification_report(y_test, preds_svm, output_dict=True)},
        }
    }
