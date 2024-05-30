import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load and preprocess your data
df = pd.read_csv('D:/AML proj/inv/PurchasesFINAL12312016.csv.zip')
columns_to_drop = ['Brand', 'Description', 'VendorNumber', 'Classification']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Preprocessing steps
df['Size'] = df['Size'].astype(str)
df['PurchasePrice'] = pd.to_numeric(df['PurchasePrice'], errors='coerce')
df['Dollars'] = pd.to_numeric(df['Dollars'], errors='coerce')
numeric_features = ['PurchasePrice', 'Dollars']
categorical_features = ['Size']

imputer_num = SimpleImputer(strategy='median')
df[numeric_features] = imputer_num.fit_transform(df[numeric_features])

imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

le = LabelEncoder()
df = df.dropna(subset=['VendorName'])
y = le.fit_transform(df['VendorName'])

encoder = OneHotEncoder(handle_unknown='ignore')
encoded_cat = encoder.fit_transform(df[categorical_features])
encoded_cat_df = pd.DataFrame(encoded_cat.toarray(), columns=encoder.get_feature_names_out(categorical_features))

X = pd.concat([df[numeric_features], encoded_cat_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the model and preprocessing objects
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(encoder, 'one_hot_encoder.pkl')
joblib.dump(imputer_num, 'imputer_num.pkl')
joblib.dump(imputer_cat, 'imputer_cat.pkl')
