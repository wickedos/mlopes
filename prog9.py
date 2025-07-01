import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=cols, skipinitialspace=True)

# Drop rows with missing values
data = data.replace('?', pd.NA).dropna()

# Encode categorical features
categorical_cols = data.select_dtypes(include='object').columns
label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_cols}
for col in categorical_cols:
    data[col] = label_encoders[col].transform(data[col])

# Features and target
X = data.drop('income', axis=1)
y = data['income']  # 0: <=50K, 1: >50K

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)




# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Overall accuracy
print("Overall accuracy:", accuracy_score(y_test, y_pred))

# Check accuracy by gender
sex_col_index = X.columns.get_loc('sex')
men_idx = X_test['sex'] == label_encoders['sex'].transform(['Male'])[0]
women_idx = ~men_idx

acc_men = accuracy_score(y_test[men_idx], y_pred[men_idx])
acc_women = accuracy_score(y_test[women_idx], y_pred[women_idx])

print("Accuracy (Men):", acc_men)
print("Accuracy (Women):", acc_women)




from sklearn.utils import class_weight

# Reweighting based on 'sex'
sample_weights = class_weight.compute_sample_weight(class_weight='balanced', y=X_train['sex'])

model_balanced = LogisticRegression(max_iter=1000)
model_balanced.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_balanced = model_balanced.predict(X_test)

# Fairness comparison
acc_men_bal = accuracy_score(y_test[men_idx], y_pred_balanced[men_idx])
acc_women_bal = accuracy_score(y_test[women_idx], y_pred_balanced[women_idx])

print("\nAfter mitigation:")
print("Accuracy (Men):", acc_men_bal)
print("Accuracy (Women):", acc_women_bal)
