from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression  # or use DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'iris_model.joblib')
print("✅ Model trained and saved as iris_model.joblib")