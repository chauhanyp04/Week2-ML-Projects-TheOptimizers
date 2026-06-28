import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

# ── Dataset ───────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target          # 0 = Malignant, 1 = Benign

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Model ─────────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# ── Output ────────────────────────────────────────────────────────────────────
print("=" * 50)
print("   BREAST CANCER CLASSIFICATION – LOGISTIC REGRESSION")
print("=" * 50)
print(f"Accuracy : {acc*100:.2f}%")
print(f"AUC-ROC  : {auc:.4f}")
print()
print(classification_report(y_test, y_pred,
      target_names=['Malignant', 'Benign']))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im = axes[0].imshow(cm, cmap='Blues')
axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
axes[0].set_xticklabels(['Malignant','Benign'])
axes[0].set_yticklabels(['Malignant','Benign'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i,j], ha='center', va='center',
                     fontsize=20, color='white' if cm[i,j]>cm.max()/2 else 'black')

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve – Logistic Regression')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Breast Cancer Classification  |  Accuracy: {acc*100:.2f}%',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/classification_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plots saved!")
