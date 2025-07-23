import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lime
import lime.lime_tabular
import os

# === 1. Carregar o dataset ===
colunas = [
    'status_checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment_since', 'installment_rate', 'personal_status_sex',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'number_existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
    'target'
]

df = pd.read_csv('data/german.data', sep=r'\s+', header=None)
df.columns = colunas

# === 2. Pré-processamento ===
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Ajustar target: 1 = Bom pagador (0), 2 = Mau pagador (1)
df['target'] = df['target'].astype(str).str.strip().map({"1": 1, "2": 0})

# Verificação de integridade
if df['target'].isnull().any():
    raise ValueError("Erro no mapeamento do target. Verifique os valores na coluna 'target'.")

# === 3. Separar features e target ===
X = df.drop('target', axis=1)
y = df['target']

# === 4. Dividir entre treino e teste ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Treinar modelo ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 6. Avaliação ===
y_pred = model.predict(X_test)
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# === 7. Aplicar LIME ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=['Bom Pagador', 'Mau Pagador'],
    mode='classification'
)

# Escolher instância
i = 0
instance = X_test.iloc[i]
exp = explainer.explain_instance(instance.to_numpy(), model.predict_proba, num_features=10)

# === 8. Salvar gráfico aprimorado ===
fig, ax = plt.subplots(figsize=(10, 6))
exp_list = exp.as_list()
features = [f.replace(' <= ', ' ≤ ').replace(' > ', ' > ') for f, _ in exp_list]
weights = [w for _, w in exp_list]
colors = ['green' if w > 0 else 'red' for w in weights]

bars = ax.barh(features, weights, color=colors)
ax.set_title("Explicação Local: Classificação como Mau Pagador", fontsize=14, pad=12)
ax.set_xlabel("Contribuição para a Predição", fontsize=12)

# Adicionar valores numéricos padronizados
for bar, val in zip(bars, weights):
    ax.text(bar.get_width() + 0.01 * np.sign(bar.get_width()),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va='center', ha='left' if val > 0 else 'right',
            fontsize=9, fontweight='bold', color='black')

# Legenda explicativa
legend_text = (
    "🟩 Verde: Características que contribuíram para classificar como Mau Pagador.\n"
    "🟥 Vermelho: Características que apontam para Bom Pagador."
)
ax.text(1.01, -0.12, legend_text, transform=ax.transAxes,
        fontsize=9, bbox=dict(facecolor='white', edgecolor='gray'), va='top')

plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/lime_explanation_friendly.png", dpi=300)
plt.close()

# === 9. Salvar HTML explicativo ===
html_exp = """
<div style="font-family: sans-serif; padding: 20px; background-color: #f4f4f4; line-height: 1.6;">
  <h2 style="color: #333;">Explicação da Decisão de Crédito</h2>
  <p>O modelo classificou este cliente como <strong style="color: red;">Mau Pagador</strong>.</p>
  <ul>
    <li><span style="color: green; font-weight: bold;">🟩</span> Indica fatores que reforçaram a decisão de <strong>negar o crédito</strong>.</li>
    <li><span style="color: red; font-weight: bold;">🟥</span> Indica fatores que sugeririam <strong>aprovação</strong> do crédito.</li>
  </ul>
  <p>Essa explicação ajuda a entender como a IA toma decisões, promovendo <strong>transparência</strong> para clientes e reguladores.</p>
</div>
"""

with open("images/lime_explanation_friendly.html", "w", encoding="utf-8") as f:
    f.write(html_exp)
    f.write(exp.as_html())

print("✅ Explicações salvas em: images/lime_explanation_friendly.png e .html")
