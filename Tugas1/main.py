import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset dari CSV
df = pd.read_csv("dataset.csv")

# 2. Konversi teks ke vektor numerik menggunakan TF-IDF agar lebih akurat
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["berita"])
y = df["label"]

# 3. Split data menjadi train dan test dengan stratifikasi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Buat model Decision Tree dengan kedalaman lebih fleksibel
clf = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluasi Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# 6. Prediksi berita dari input user
def prediksi_berita():
    berita_baru = input("\nMasukkan judul berita: ")
    X_new = vectorizer.transform([berita_baru])  # Transform input user ke vektor
    prediksi = clf.predict(X_new)[0]  # Ambil nilai prediksi pertama
    probas = clf.predict_proba(X_new)  # Ambil probabilitas prediksi

    # Klasifikasi berdasarkan prediksi model
    kategori = {
        1: "Olahraga",
        2: "Kriminal",
        3: "Ekonomi",
        4: "Bencana",
        0: "Politik"
    }

    print(f"\nJenis berita: {kategori.get(prediksi, 'Tidak Diketahui')}")
    print(f"Probabilitas Prediksi: {probas}")

# Jalankan fungsi prediksi
prediksi_berita()
