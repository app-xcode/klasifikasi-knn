import streamlit as st  
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    /* Hilangkan header Streamlit */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Hilangkan footer (dibawah kiri, tulisan Made with Streamlit) */
    footer {
        display: none !important;
    }

    /* Hilangkan toolbar menu kanan atas (≡ / About / Settings) */
    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Hilangkan profil / preview (kalau ada) */
    .stApp header, .stApp [data-testid="stDecoration"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------
# Label kelas
# -------------------------
LABEL_ORDER = ["kecil", "sedang", "besar"]

# -------------------------
# Mapping angka ke label
# -------------------------
def fix_labels(df):
    mapping = {0: "kecil", 1: "sedang", 2: "besar"}
    if df["Kelas"].dtype in [int, float]:
        df["Kelas"] = df["Kelas"].map(mapping)
    elif df["Kelas"].astype(str).str.isdigit().any():
        df["Kelas"] = df["Kelas"].replace({"0": "kecil", "1": "sedang", "2": "besar"})
    return df

# -------------------------
# Plot & Confusion Matrix
# -------------------------
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, interpolation='nearest', aspect='auto', cmap="Blues")
    ax.set_xticks(np.arange(len(LABEL_ORDER)))
    ax.set_yticks(np.arange(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    return fig

def show_confusion_matrix(cm, title="Confusion Matrix"):
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in LABEL_ORDER],
        columns=[f"Pred {c}" for c in LABEL_ORDER]
    )
    st.write(f"#### {title}")
    st.dataframe(cm_df)
    st.pyplot(plot_confusion_matrix(cm, title))

# -------------------------
# KNN Cross Validation
# -------------------------
def knn_crossval(df, k=3, n_splits=5):
    df = fix_labels(df.copy())
    X = df.drop(columns=["Kelas"]).astype(int)
    y = df["Kelas"].astype(str)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results, cms = [], []
    for i, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        acc = accuracy_score(y.iloc[test_idx], y_pred)
        cm = confusion_matrix(y.iloc[test_idx], y_pred, labels=LABEL_ORDER)
        fold_results.append((i, acc))
        cms.append((i, cm))
    avg_acc = np.mean([acc for _, acc in fold_results])
    return avg_acc, fold_results, cms

# -------------------------
# Dataset overlap (150 data: 50 kecil, 50 sedang, 50 besar)
# -------------------------
np.random.seed(42)

daya_list = np.concatenate([
    np.random.choice([450, 900], 50),         # kecil bisa 450/900
    np.random.choice([450, 900, 1300], 50),   # sedang overlap penuh
    np.random.choice([900, 1300], 50)         # besar bisa 900/1300
])
pulsa_list = np.concatenate([
    np.random.choice([25, 50, 100], 50),      # kecil 25–100
    np.random.choice([50, 100, 200], 50),     # sedang overlap 50–200
    np.random.choice([100, 200, 400], 50)     # besar overlap 100–400
])
alat_list = np.concatenate([
    np.random.randint(1, 7, 50),              # kecil 1–6 alat
    np.random.randint(3, 9, 50),              # sedang 3–8 alat
    np.random.randint(5, 11, 50)              # besar 5–10 alat
])
kelas_list = (["kecil"] * 50) + (["sedang"] * 50) + (["besar"] * 50)

df_expanded = pd.DataFrame({
    "Daya": daya_list,
    "Pulsa": pulsa_list,
    "Alat": alat_list,
    "Kelas": kelas_list
})

df_expanded = df_expanded.sample(frac=1, random_state=42).reset_index(drop=True)

df_expanded = fix_labels(df_expanded)

# -------------------------
# Streamlit UI
# -------------------------
st.title("Klasifikasi Pola Konsumsi Energi Listrik Rumah Tangga Menggunakan Metode K-NN")

st.write("### Dataset overlap (150 data: 50 kecil, 50 sedang, 50 besar):")
# st.dataframe(df_expanded.head())
# st.dataframe(df_expanded.sample(10))
st.dataframe(df_expanded)


# Pilih nilai K (hanya 3,4,5,7,9)
k_value = st.selectbox("Pilih Nilai K", [3, 4, 5, 7, 9], index=0)

# --- Jalankan 5-Fold CV ---
if st.button("Jalankan 5-Fold CV"):
    avg_acc, fold_results, all_conf_matrices = knn_crossval(df_expanded, k=k_value)
    st.subheader(f"Hasil Rata-rata Akurasi (k={k_value}): {avg_acc:.2f}")
    for fold, acc in fold_results:
        st.write(f"Fold {fold} - Akurasi: {acc:.2f}")
    for fold, cm in all_conf_matrices:
        show_confusion_matrix(cm, title=f"Fold {fold}")

# --- Jalankan Split 120:30 ---
if st.button("Jalankan Split Data (120:30)"):
    df_expanded = fix_labels(df_expanded)
    X = df_expanded.drop(columns=["Kelas"]).astype(int)
    y = df_expanded["Kelas"].astype(str)

    # bagi 120 latih dan 30 uji
    X_train, X_test = X.iloc[:120], X.iloc[120:]
    y_train, y_test = y.iloc[:120], y.iloc[120:]

    # tampilkan data latih & uji
    st.subheader("Data Latih (120 baris)")
    st.dataframe(pd.concat([X_train, y_train], axis=1))

    st.subheader("Data Uji (30 baris)")
    st.dataframe(pd.concat([X_test, y_test], axis=1))

    # training dan evaluasi
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)

    st.subheader(f"Akurasi Split 120:30 (k={k_value}): {acc:.2f}")
    show_confusion_matrix(cm, title="Confusion Matrix (Split 120:30)")

# --- Cari k terbaik (detail per fold dan akurasi tertinggi) ---
if st.button("Cari Model Terbaik (3,4,5,7,9)"):
    best_overall_k = None
    summary_results = []  # simpan ringkasan hasil tiap k
    best_overall_acc = -1

    for k in [3, 4, 5, 7, 9]:
        avg_acc, fold_results, _ = knn_crossval(df_expanded, k=k, n_splits=5)

        # tampilkan hasil per fold untuk k ini
        st.write(f"### Hasil 5-Fold untuk k={k}")
        for fold, acc in fold_results:
            st.write(f"Fold {fold} - Akurasi: {acc:.2f}")

        # cari akurasi tertinggi dari 5 fold
        max_acc = max([acc for _, acc in fold_results])
        summary_results.append((k, avg_acc, max_acc))

        # cek apakah ini akurasi terbaik secara keseluruhan
        if max_acc > best_overall_acc:
            best_overall_acc = max_acc
            best_overall_k = k

    # buat tabel ringkasan
    df_summary = pd.DataFrame(summary_results, columns=["k", "Akurasi Rata-rata", "Akurasi Tertinggi"])
    st.write("### Ringkasan Hasil Uji k")
    st.dataframe(df_summary)

    st.success(f"Model terbaik: k={best_overall_k} dengan akurasi tertinggi {best_overall_acc:.2f}")

    # --- Lanjutkan ke prediksi data baru otomatis dengan model terbaik ---
st.subheader("Prediksi Data Baru dengan Model Terbaik")
daya = st.selectbox("Pilih Daya", [450, 900, 1300], key="best_daya")
pulsa = st.selectbox("Pilih Pulsa", [25, 50, 100, 200, 400], key="best_pulsa")
alat = st.number_input("Jumlah Alat (1-10)", min_value=1, max_value=10, value=3, key="best_alat")

if st.button("Prediksi Kelas (Model Terbaik)"):
    best_overall_k = None
    summary_results = []  # simpan ringkasan hasil tiap k
    best_overall_acc = -1

    for k in [3, 4, 5, 7, 9]:
        avg_acc, fold_results, _ = knn_crossval(df_expanded, k=k, n_splits=5)

        # tampilkan hasil per fold untuk k ini
        st.write(f"### Hasil 5-Fold untuk k={k}")
        for fold, acc in fold_results:
            st.write(f"Fold {fold} - Akurasi: {acc:.2f}")

        # cari akurasi tertinggi dari 5 fold
        max_acc = max([acc for _, acc in fold_results])
        summary_results.append((k, avg_acc, max_acc))

        # cek apakah ini akurasi terbaik secara keseluruhan
        if max_acc > best_overall_acc:
            best_overall_acc = max_acc
            best_overall_k = k
    df_expanded = fix_labels(df_expanded)
    X = df_expanded.drop(columns=["Kelas"]).astype(int)
    y = df_expanded["Kelas"].astype(str)
    model = KNeighborsClassifier(n_neighbors=best_overall_k)
    model.fit(X, y)
    x_new = np.array([[daya, pulsa, alat]])
    pred = model.predict(x_new)[0]
    st.success(f"Hasil Prediksi dengan k={best_overall_k}: {pred}")
