import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import lime
import lime.lime_tabular

# === 1. Carregar o dataset ===
colunas = [
    'status_checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment_since', 'installment_rate', 'personal_status_sex',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'number_existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
    'target'
]

df = pd.read_csv('data/german.data', sep=' ', header=None)
df.columns = colunas

# === 2. Pré-processamento ===
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# === 3. Separar features e target ===
X = df.drop('target', axis=1)
y = df['target']

# === 4. Dividir entre treino e teste ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Treinar modelo Random Forest ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 6. Avaliação básica ===
y_pred = model.predict(X_test)
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# === 7. Aplicar LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Good', 'Bad'],
    mode='classification'
)

# Selecionar uma amostra do teste
i = 0
instance = X_test.iloc[i]

# Gerar explicação
exp = explainer.explain_instance(instance.to_numpy(), model.predict_proba, num_features=10)

# Mostrar explicação no terminal
print("\nExplicação da instância:", i)
print(exp.as_list())

# === 8. Salvar explicações ===
fig = exp.as_pyplot_figure()
fig.savefig('images/lime_explanation.png', bbox_inches='tight')
exp.save_to_file('images/lime_explanation.html')

print("\nExplicações salvas na pasta 'images/'")
