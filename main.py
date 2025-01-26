import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# Wczytaj dane z pliku (zamień 'dane.csv' na ścieżkę do Twojego pliku)
file_path = 'baza_danych_17k.xlsx'  # Możesz zmienić rozszerzenie na .xlsx jeśli dane są w formacie Excel
if file_path.endswith('.csv'):
    data = pd.read_csv(file_path)
elif file_path.endswith('.xlsx'):
    data = pd.read_excel(file_path)
else:
    raise ValueError("Plik musi być w formacie .csv lub .xlsx")

# Sprawdź dane
print("Podgląd danych:")
print(data.head())

# Zakładamy, że pole 'pm' (temperatura magnesów trwałych) jest celem (y), a pozostałe kolumny są cechami (X)
X = data.drop(columns=['pm'])
y = data['pm'] > 60  # Przykład binarnej klasyfikacji: true jeśli temperatura magnesów > 100 stopni

# Podziel dane na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standaryzuj cechy dla modeli wymagających skalowania
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Metoda 1: Regresja logistyczna
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)
logistic_preds = logistic_model.predict(X_test_scaled)
logistic_acc = accuracy_score(y_test, logistic_preds)
print("\nRegresja Logistyczna")
print("Accuracy:", accuracy_score(y_test, logistic_preds))
print(classification_report(y_test, logistic_preds))

# Metoda 2: Drzewo decyzyjne
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
decision_tree_preds = decision_tree_model.predict(X_test)
decision_tree_acc = accuracy_score(y_test, decision_tree_preds)
print("\nDrzewo Decyzyjne")
print("Accuracy:", accuracy_score(y_test, decision_tree_preds))
print(classification_report(y_test, decision_tree_preds))

# Metoda 3: Maszyna wektorów nośnych (SVM)
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_preds)
print("\nSVM")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))

# Wykresy porównawcze wyników
model_names = ["Logistyczna", "Drzewo Decyzyjne", "SVM"]
accuracies = [logistic_acc, decision_tree_acc, svm_acc]

plt.figure(figsize=(8, 6))
sns.barplot(x=model_names, y=accuracies)
plt.ylim(0.9, 1.0)
plt.title("Porównanie dokładności modeli", fontsize=16)
plt.ylabel("Dokładność", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.show()

# Macierze konfuzji
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models = [(logistic_model, X_test_scaled, "Regresja Logistyczna"),
          (decision_tree_model, X_test, "Drzewo Decyzyjne"),
          (svm_model, X_test_scaled, "SVM")]

for i, (model, X_data, title) in enumerate(models):
    ConfusionMatrixDisplay.from_estimator(model, X_data, y_test, ax=axes[i], cmap="viridis")
    axes[i].set_title(title)

plt.tight_layout()
plt.show()

# Podsumowanie
print("\nModelowanie zakończone. Porównaj wyniki modeli, aby wybrać najlepszy.")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
