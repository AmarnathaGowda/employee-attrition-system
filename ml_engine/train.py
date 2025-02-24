# ml_engine/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from .pipelines.preprocessing import preprocessor
from sklearn.pipeline import Pipeline

def train_models():
    df = pd.read_csv('ml_engine/data/emp_attrition.csv')
    X = df.drop(['Attrition', 'EmployeeNumber'], axis=1)
    y = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    models = {
        'logreg': LogisticRegression(class_weight='balanced'),
        'randomforest': RandomForestClassifier(n_estimators=200),
        'xgboost': GradientBoostingClassifier(n_estimators=300)
    }
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', None)
    ])
    
    results = {}
    for name, model in models.items():
        full_pipeline.set_params(classifier=model)
        full_pipeline.fit(X_train, y_train)
        
        preds = full_pipeline.predict(X_test)
        probas = full_pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'f1': f1_score(y_test, preds),
            'roc_auc': roc_auc_score(y_test, probas)
        }
        
        joblib.dump(full_pipeline, f'ml_engine/models/{name}_pipeline.joblib')
        results[name] = metrics
    
    return results
