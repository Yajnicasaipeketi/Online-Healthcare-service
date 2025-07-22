import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load and preprocess dataset
df = pd.read_csv("test_ds_2.csv")

# Filter out diseases with very few samples
y_counts = df['Disease'].value_counts()
classes_to_keep = y_counts[y_counts > 1].index
df_filtered = df[df['Disease'].isin(classes_to_keep)]

# Data augmentation: ensure a minimum number of samples per class
min_samples_per_class = 10  # Adjust this threshold as needed
for disease in df_filtered['Disease'].unique():
    count = df_filtered['Disease'].value_counts()[disease]
    if count < min_samples_per_class:
        additional_samples_needed = min_samples_per_class - count
        additional_samples = df_filtered[df_filtered['Disease'] == disease].sample(n=additional_samples_needed, replace=True)
        df_filtered = pd.concat([df_filtered, additional_samples], ignore_index=True)

# Prepare features and labels
X = df_filtered[['Symptoms', 'Severity', 'Duration', 'Frequency']]
y = df_filtered['Disease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing pipeline
text_transformer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'Symptoms'),
        ('categorical', categorical_transformer, ['Severity', 'Duration', 'Frequency'])
    ]
)

# Define the model pipeline with hyperparameter tuning
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', class_weight='balanced'))
])

# Hyperparameter grid for tuning
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']  # Only applies to non-linear kernels
}

# Grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after tuning
best_model = grid_search.best_estimator_

# Evaluate model performance
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Tuned Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print(classification_report(y_test, y_pred))

# Save the best model and the vectorizer
joblib.dump(best_model, 'best_disease_predictor_model_svm.pkl')
# Save the vectorizer from the pipeline
joblib.dump(grid_search.best_estimator_.named_steps['preprocessor'].named_transformers_['text'], 'tfidf_vectorizer.pkl')

print("Best model and vectorizer saved successfully!")

'''
Tuned Model Accuracy: 0.9076923076923077
Confusion Matrix:
 [[2 0 0 ... 0 0 0]
 [0 2 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 0 2]]
                 precision    recall  f1-score   support

    Alzheimer's       1.00      1.00      1.00         2
         Anemia       1.00      1.00      1.00         2
      Arthritis       1.00      1.00      1.00         2
         Asthma       1.00      1.00      1.00         2
     Bronchitis       1.00      1.00      1.00         2
       COVID-19       1.00      1.00      1.00         2
 Celiac Disease       1.00      0.50      0.67         2
     Chickenpox       1.00      1.00      1.00         2
        Cholera       1.00      1.00      1.00         2
           Cold       1.00      0.50      0.67         2
Common Headache       1.00      1.00      1.00         2
         Dengue       1.00      0.50      0.67         2
       Diabetes       1.00      1.00      1.00         2
       Epilepsy       1.00      0.50      0.67         2
            Flu       1.00      1.00      1.00         2
      Gastritis       0.67      1.00      0.80         2
      Hepatitis       1.00      1.00      1.00         2
   Hypertension       1.00      0.50      0.67         2
Hyperthyroidism       0.75      1.00      0.86         3
 Hypothyroidism       0.67      1.00      0.80         2
       Jaundice       1.00      1.00      1.00         2
          Lupus       1.00      1.00      1.00         2
        Malaria       1.00      1.00      1.00         2
        Measles       1.00      1.00      1.00         2
       Migraine       0.50      0.50      0.50         2
    Parkinson's       1.00      1.00      1.00         2
      Pneumonia       1.00      1.00      1.00         2
      Psoriasis       1.00      1.00      1.00         2
      Sinusitis       0.67      1.00      0.80         2
   Tuberculosis       1.00      1.00      1.00         2
        Typhoid       0.67      1.00      0.80         2
          Ulcer       1.00      1.00      1.00         2

       accuracy                           0.91        65
      macro avg       0.93      0.91      0.90        65
   weighted avg       0.93      0.91      0.90        65

Best model and vectorizer saved successfully!
'''

