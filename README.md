# 🩺 Diabetes Prediction using Ensemble Learning and LIME

This project predicts whether a patient has diabetes using the **Pima Indian Diabetes Dataset**. It involves preprocessing, model tuning, stacking ensemble learning with a meta-learner, and instance-level explanation using **LIME**.

---

## 📊 Dataset

- **Source**: [Kaggle - Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Attributes**: Glucose, Blood Pressure, BMI, Age, Insulin, etc.
- **Target**: `Outcome` (0 = Non-diabetic, 1 = Diabetic)

---

## ✅ Workflow Overview

| Step | Description |
|------|-------------|
| 📥 Data Loading | Loaded dataset using pandas |
| 🧹 Preprocessing | Handled missing values with `KNNImputer`, removed outliers using IQR |
| 📈 EDA | Correlation heatmap, class distribution, boxplots |
| 🧪 Train-Test Split | 80/20 split using `train_test_split` with stratification |
| 🔍 Model Tuning | Applied `GridSearchCV` on:<br>• RandomForestClassifier<br>• ExtraTreesClassifier<br>• XGBClassifier |
| 🧠 Meta Learner | Combined best models using `StackingClassifier` with `LogisticRegression` as final estimator |
| 📊 Evaluation | Evaluated with accuracy, confusion matrix, classification report |
| 💡 Explainability | Used `LIME` to explain individual predictions |

---

## 🔧 Libraries Used

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lime
