import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset dari CSV
df = pd.read_csv("dataset.csv")

# 2. Konversi teks ke vektor numerik
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["berita"])
y = df["label"]

# 3. Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Buat model Decision Tree
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluasi Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Prediksi berita dari input user
def prediksi_berita():
    berita_baru = input("\nMasukkan judul berita: ")
    X_new = vectorizer.transform([berita_baru])
    prediksi = clf.predict(X_new)
    hasil = "Olahraga" if prediksi[0] == 1 else "Non-Olahraga"
    print(f"\nJenis berita: {hasil}")

# Jalankan fungsi prediksi
prediksi_berita()
