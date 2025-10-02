import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -------------------------
# Load the Laptop 1 CSV (training data)
# -------------------------
csv_path = "model/keypoint_classifier/keypoint.csv"
data = pd.read_csv(csv_path, header=None)  # no headers in CSV

# First column = gesture label, rest = features (hand landmarks)
X = data.iloc[:, 1:]  # features
y = data.iloc[:, 0].astype(str)   # labels as strings

# -------------------------
# Map gestures to game directions
# -------------------------
gesture_to_direction = {
    "1": "UP",
    "2": "DOWN",
    "3": "LEFT",
    "4": "RIGHT"
}

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# KNN Classifier
# -------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# -------------------------
# SVM Classifier
# -------------------------
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# -------------------------
# Example: Map gestures to directions
# -------------------------
sample_gestures = ["1", "2", "3", "4"]  # test samples
print("\nSample Mapped Directions:")
for g in sample_gestures:
    print(f"Gesture {g} -> Direction {gesture_to_direction.get(g, 'UNKNOWN')}")
