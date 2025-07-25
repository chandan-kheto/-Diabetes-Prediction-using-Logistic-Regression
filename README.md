 # Diabetes Prediction using Logistic Regression
🔍 Project Overview
This project predicts whether a person is diabetic or not based on diagnostic health features, using Logistic Regression, a powerful supervised ML algorithm for binary classification.

📂 Dataset
Pima Indians Diabetes Dataset

📥 Download from Kaggle

Outcome column → Target variable (1 = Diabetic, 0 = Non-Diabetic)

🔧 Tech Stack
Tool	Purpose
Python	Programming language
Pandas	Data manipulation
NumPy	Numerical operations
Seaborn & Matplotlib	Data visualization
Scikit-learn	ML model, metrics, preprocessing

✅ Workflow
Import Libraries

Load Dataset

EDA (Explore Data)

Feature Scaling

Train-Test Split

Train Logistic Regression Model

Evaluate Model

Visualize Confusion Matrix

📊 Model Performance

Accuracy: ~76%

Confusion Matrix:
[[79 20]
 [18 37]]

 Classification Report:
               precision    recall  f1-score   support

           0       0.81      0.80      0.81        99
           1       0.65      0.67      0.66        55

    accuracy                           0.75       154
   macro avg       0.73      0.74      0.73       154
weighted avg       0.76      0.75      0.75       154


Precision, Recall, F1-Score for both classes
📈 Confusion Matrix Plot
<!-- Upload and link your plot image if needed -->

📁 Folder Structure
Copy code
📂 Diabetes-Prediction-LogisticRegression
│
├── diabetes.csv
├── model_code.py
├── confusion_matrix.png
└── README.md
🚀 Future Improvements
Try other models: SVM, Random Forest

Hyperparameter tuning

Add Flask app for live prediction

🙌 Credits
Dataset from UCI ML Repo / Kaggle

