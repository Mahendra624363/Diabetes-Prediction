# 🩺 Diabetes Prediction using Ensemble Learning

This project predicts whether a patient has diabetes using the **Pima Indian Diabetes Dataset**. It includes preprocessing, exploratory data analysis, model tuning, and a stacked ensemble classifier to improve prediction accuracy.

---

## 📊 Dataset

- **Source**: [Kaggle - Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- **Target**: `Outcome` (0 = Non-diabetic, 1 = Diabetic)

---

## ✅ Workflow Overview

| Step              | Description |
|------------------|-------------|
| 📥 **Data Loading**        | Loaded data using `pandas` |
| 🧹 **Preprocessing**       | - Replaced zeroes in invalid columns<br>- Handled missing values using `KNNImputer`<br>- Removed outliers using IQR |
| 📊 **Exploratory Data Analysis** | - Heatmap of feature correlations<br>- Boxplots<br>- Class distribution plot |
| 🧪 **Train-Test Split**    | Performed 80/20 split using `train_test_split` with stratification |
| 🔍 **Model Tuning**        | Applied `GridSearchCV` on:<br> • RandomForestClassifier<br> • ExtraTreesClassifier<br> • LogisticRegression |
| 🧠 **Stacking Classifier** | Combined best models using `StackingClassifier` with `LogisticRegression` as final estimator |
| 📈 **Evaluation Metrics**  | Evaluated with:<br> - Accuracy<br> - Confusion Matrix<br> - Classification Report |

---

## 🧠 Models Used

- Random Forest Classifier
- Extra Trees Classifier
- Logistic Regression
- Stacking Classifier (Meta Learner)

---

## 🔧 Libraries Used

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
