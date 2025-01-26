import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# Wczytaj dane z pliku (musi znajdować się w tym samym folderze co plik main, lub zmień ścieżkę)
file_path = 'baza_danych_17k.xlsx'
if file_path.endswith('.csv'):
    data = pd.read_csv(file_path)
elif file_path.endswith('.xlsx'):
    data = pd.read_excel(file_path)
else:
    raise ValueError("Plik musi być w formacie .csv lub .xlsx")

# Sprawdzenie danych
print("Podgląd danych:")
print(data.head())

# Przygotowanie danych do testów
X = data.drop(columns=['pm', 'profile_id', 'coolant', 'stator_winding', 'stator_tooth', 'stator_yoke'])
#X = data.drop(columns=['pm', 'profile_id'])
y = data['pm'] > 60

# Podział danych na treningowe i do testów (jeżeli dane treningowe znajdują się w innym pliku trzeba je wczytać na nowo, w naszej sytuacji istnieje na tyle rekordów, że można z tej samej bazy danych zarówno trenować i testować)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standaryzacja
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalizacja
scaler1 = MinMaxScaler()  # Domyślnie zakres (0, 1), ale można ustawić np. feature_range=(0, 1)
X_train_normalized = scaler1.fit_transform(X_train)
X_test_normalized = scaler1.transform(X_test)

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

# Wyświetlenie całego drzewa
plt.figure(figsize=(15, 10))
plot_tree(decision_tree_model, feature_names=list(X.columns), class_names=["False", "True"], filled=True)
plt.title("Drzewo Decyzyjne")
plt.show()

# Metoda 3: Maszyna wektorów nośnych (SVM)
svm_model = SVC()
svm_model.fit(X_train_normalized, y_train)
svm_preds = svm_model.predict(X_test_normalized)
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
          (svm_model, X_test_normalized, "SVM")]
for i, (model, X_data, title) in enumerate(models):
    ConfusionMatrixDisplay.from_estimator(model, X_data, y_test, ax=axes[i], cmap="viridis")
    axes[i].set_title(title)
plt.tight_layout()
plt.show()

# Korelacja między danymi
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Macierz korelacji między zmiennymi", fontsize=16)
plt.show()

# Korelacja między danymi a stricte wartością pm
pm_correlation = correlation_matrix['pm']
sorted_abs_correlation = pm_correlation.abs().sort_values(ascending=True)
plt.figure(figsize=(10, 6))
sorted_abs_correlation.plot(kind='bar', color='skyblue')
plt.title("Korelacje względem 'pm' (posortowane rosnąco)", fontsize=16)
plt.ylabel("Wartość korelacji", fontsize=12)
plt.xlabel("Zmienna", fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Podsumowanie
print("\nModelowanie zakończone.")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
