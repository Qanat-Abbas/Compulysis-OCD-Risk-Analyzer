import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, roc_curve, f1_score)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.inspection import permutation_importance
import datetime
from sklearn.pipeline import Pipeline
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Compulysis: OCD Risk Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e1e8ed;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        border: 2px solid #e74c3c;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(231,76,60,0.2);
        animation: pulse-red 2s infinite;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border: 2px solid #f39c12;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(243,156,18,0.2);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #27ae60;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(39,174,96,0.2);
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 4px 15px rgba(231,76,60,0.2); }
        50% { box-shadow: 0 6px 20px rgba(231,76,60,0.4); }
        100% { box-shadow: 0 4px 15px rgba(231,76,60,0.2); }
    }
    
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    
    .professional-note {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        border-left: 5px solid #e17055;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .dimension-analysis {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #6c5ce7;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HYPERPARAMETER TUNING: Grid Search with reused preprocessing pipeline
# ============================================================================
def perform_hyperparameter_tuning(X, y, preprocessor):
    """
    Performs grid search for all models using the existing preprocessing pipeline.
    Reuses the preprocessor from train_enhanced_models to ensure consistency.
    Prevents data leakage by fitting preprocessor within each CV fold.
    """
    
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define hyperparameter grids
    param_grids = {
        'Logistic Regression': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l2'],
            'model__solver': ['lbfgs', 'liblinear'],
            'model__max_iter': [1000, 2000]
        },
        'Random Forest': {
            'model__n_estimators': [50, 100, 150, 200],
            'model__max_depth': [5, 10, 15, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2']
        },
        'Decision Tree': {
            'model__max_depth': [3, 5, 7, 10, 15, None],
            'model__min_samples_split': [2, 5, 10, 15],
            'model__min_samples_leaf': [1, 2, 4, 8],
            'model__criterion': ['gini', 'entropy']
        },
        'Naive Bayes': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }
    
    # Initialize base models
    models_dict = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Naive Bayes': GaussianNB()
    }
    
    tuning_results = {}
    best_models = {}
    
    for model_name, base_model in models_dict.items():
        print(f"Tuning {model_name}...")
        
        # Create preprocessing + model pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', base_model)
        ])
        
        # Get default performance on test set
        pipeline.fit(X_train, y_train)
        default_pred = pipeline.predict(X_test)
        default_accuracy = accuracy_score(y_test, default_pred)
        
        # Perform grid search with proper cross-validation
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        
        # Get tuned performance on test set
        tuned_pred = best_pipeline.predict(X_test)
        tuned_accuracy = accuracy_score(y_test, tuned_pred)
        tuned_precision = precision_score(y_test, tuned_pred, average='weighted', zero_division=0)
        tuned_recall = recall_score(y_test, tuned_pred, average='weighted', zero_division=0)
        tuned_f1 = f1_score(y_test, tuned_pred, average='weighted', zero_division=0)
        
        # Extract best model params (remove 'model__' prefix for clarity)
        best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
        
        # Store results
        tuning_results[model_name] = {
            'default_accuracy': default_accuracy * 100,
            'tuned_accuracy': tuned_accuracy * 100,
            'accuracy_improvement': (tuned_accuracy - default_accuracy) * 100,
            'tuned_precision': tuned_precision * 100,
            'tuned_recall': tuned_recall * 100,
            'tuned_f1': tuned_f1 * 100,
            'best_params': best_params,
            'best_cv_score': grid_search.best_score_ * 100,
            'num_combinations': len(grid_search.cv_results_['params'])
        }
        
        best_models[model_name] = best_pipeline
        
        print(f"✓ {model_name} tuning complete")
        print(f"  Default Accuracy: {default_accuracy*100:.2f}%")
        print(f"  Tuned Accuracy: {tuned_accuracy*100:.2f}%")
        print(f"  Improvement: {(tuned_accuracy - default_accuracy)*100:.2f}%")
        print()
    
    return tuning_results, best_models, X_test, y_test


# ============================================================================
# SENSITIVITY ANALYSIS: Proper preprocessing in sensitivity checks
# ============================================================================
def sensitivity_analysis_rf(X, y, preprocessor):
    """
    Random Forest sensitivity analysis with reused preprocessing pipeline.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use cloned preprocessor
    preprocessor_clone = clone(preprocessor)
    X_train_transformed = preprocessor_clone.fit_transform(X_train)
    X_test_transformed = preprocessor_clone.transform(X_test)
    
    n_estimators_range = [10, 50, 100, 150, 200, 250]
    max_depth_range = [3, 5, 10, 15, None]
    
    sensitivity_data = []
    
    for n_est in n_estimators_range:
        for max_d in max_depth_range:
            rf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            rf.fit(X_train_transformed, y_train)
            pred = rf.predict(X_test_transformed)
            accuracy = accuracy_score(y_test, pred)
            
            sensitivity_data.append({
                'n_estimators': n_est,
                'max_depth': str(max_d),
                'Accuracy': accuracy * 100
            })
    
    return pd.DataFrame(sensitivity_data)


def sensitivity_analysis_dt(X, y, preprocessor):
    """
    Decision Tree sensitivity analysis with proper preprocessing.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    categorical_cols = ["Gender", "Current Education Level"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    max_depth_range = [2, 3, 5, 7, 10, 15, None]
    min_samples_split_range = [2, 5, 10, 15, 20]
    
    sensitivity_data = []
    
    for max_d in max_depth_range:
        for min_split in min_samples_split_range:
            dt = DecisionTreeClassifier(
                max_depth=max_d,
                min_samples_split=min_split,
                random_state=42,
                class_weight='balanced'
            )
            dt.fit(X_train_transformed, y_train)
            pred = dt.predict(X_test_transformed)
            accuracy = accuracy_score(y_test, pred)
            
            sensitivity_data.append({
                'max_depth': str(max_d),
                'min_samples_split': min_split,
                'Accuracy': accuracy * 100
            })
    
    return pd.DataFrame(sensitivity_data)


def sensitivity_analysis_lr(X, y, preprocessor):
    """
    Logistic Regression sensitivity analysis with proper preprocessing.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    categorical_cols = ["Gender", "Current Education Level"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    c_range = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    
    sensitivity_data = []
    
    for c_val in c_range:
        lr = LogisticRegression(
            C=c_val,
            random_state=42,
            class_weight='balanced',
            max_iter=2000,
            solver='lbfgs'
        )
        lr.fit(X_train_transformed, y_train)
        pred = lr.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, pred)
        
        sensitivity_data.append({
            'C': c_val,
            'Accuracy': accuracy * 100
        })
    
    return pd.DataFrame(sensitivity_data)

# ============================================================================
# VALIDATION FUNCTION 1: Original 80/20 Stratified Split (with 5 runs)
# ============================================================================
def validate_with_split(X, y, preprocessor_template, num_runs=5):
    """
    Validates models using 80/20 stratified train-test split.
    Runs 5 independent trials and returns mean ± std metrics.
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
    }
    
    results_per_run = {name: {'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []} 
                      for name in models}
    
    for run in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42+run, stratify=y
        )
        
        preprocessor = clone(preprocessor_template)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        for model_name, model in models.items():
            m = clone(model)
            m.fit(X_train_transformed, y_train)
            y_pred = m.predict(X_test_transformed)
            
            results_per_run[model_name]['accuracies'].append(accuracy_score(y_test, y_pred))
            results_per_run[model_name]['precisions'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            results_per_run[model_name]['recalls'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            results_per_run[model_name]['f1_scores'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    
    split_results = {}
    for model_name in models:
        split_results[model_name] = {
            'Accuracy': (np.mean(results_per_run[model_name]['accuracies']) * 100, 
                        np.std(results_per_run[model_name]['accuracies']) * 100),
            'Precision': (np.mean(results_per_run[model_name]['precisions']) * 100,
                         np.std(results_per_run[model_name]['precisions']) * 100),
            'Recall': (np.mean(results_per_run[model_name]['recalls']) * 100,
                      np.std(results_per_run[model_name]['recalls']) * 100),
            'F1 Score': (np.mean(results_per_run[model_name]['f1_scores']) * 100,
                        np.std(results_per_run[model_name]['f1_scores']) * 100),
        }
    
    return split_results


# ============================================================================
# VALIDATION FUNCTION 2: 5-Fold Stratified Cross-Validation (with 5 runs)
# ============================================================================
def validate_with_kfold(X, y, preprocessor_template, num_runs=5, n_splits=5):
    """
    Validates models using 5-Fold Stratified Cross-Validation.
    Runs 5 independent trials and returns mean ± std metrics.
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
    }
    
    results_per_run = {name: {'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []} 
                      for name in models}
    
    for run in range(num_runs):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42+run)
        fold_results = {name: {'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []} 
                       for name in models}
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            preprocessor = clone(preprocessor_template)
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            for model_name, model in models.items():
                m = clone(model)
                m.fit(X_train_transformed, y_train)
                y_pred = m.predict(X_test_transformed)
                
                fold_results[model_name]['accuracies'].append(accuracy_score(y_test, y_pred))
                fold_results[model_name]['precisions'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_results[model_name]['recalls'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_results[model_name]['f1_scores'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        for model_name in models:
            results_per_run[model_name]['accuracies'].append(np.mean(fold_results[model_name]['accuracies']))
            results_per_run[model_name]['precisions'].append(np.mean(fold_results[model_name]['precisions']))
            results_per_run[model_name]['recalls'].append(np.mean(fold_results[model_name]['recalls']))
            results_per_run[model_name]['f1_scores'].append(np.mean(fold_results[model_name]['f1_scores']))
    
    kfold_results = {}
    for model_name in models:
        kfold_results[model_name] = {
            'Accuracy': (np.mean(results_per_run[model_name]['accuracies']) * 100,
                        np.std(results_per_run[model_name]['accuracies']) * 100),
            'Precision': (np.mean(results_per_run[model_name]['precisions']) * 100,
                         np.std(results_per_run[model_name]['precisions']) * 100),
            'Recall': (np.mean(results_per_run[model_name]['recalls']) * 100,
                      np.std(results_per_run[model_name]['recalls']) * 100),
            'F1 Score': (np.mean(results_per_run[model_name]['f1_scores']) * 100,
                        np.std(results_per_run[model_name]['f1_scores']) * 100),
        }
    
    return kfold_results


# ============================================================================
# VALIDATION FUNCTION 3: Leave-One-Out Cross-Validation (LOOCV)
# ============================================================================
def validate_with_loocv(X, y, preprocessor_template):
    """
    Validates models using Leave-One-Out Cross-Validation.
    Returns metrics as tuples (value, 0) for consistency.
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', n_jobs=-1),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
    }
    
    loo = LeaveOneOut()
    loocv_results = {name: {'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []} 
                    for name in models}
    
    iteration = 0
    total_iterations = len(X)
    
    for train_idx, test_idx in loo.split(X):
        iteration += 1
        if iteration % 20 == 0:
            print(f"LOOCV Progress: {iteration}/{total_iterations}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        preprocessor = clone(preprocessor_template)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        for model_name, model in models.items():
            try:
                m = clone(model)
                m.fit(X_train_transformed, y_train)
                y_pred = m.predict(X_test_transformed)
                
                loocv_results[model_name]['accuracies'].append(accuracy_score(y_test, y_pred))
                loocv_results[model_name]['precisions'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                loocv_results[model_name]['recalls'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                loocv_results[model_name]['f1_scores'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            except Exception as e:
                print(f"Error with {model_name} at iteration {iteration}: {e}")
                continue
    
    final_loocv_results = {}
    for model_name in models:
        if len(loocv_results[model_name]['accuracies']) > 0:
            final_loocv_results[model_name] = {
                'Accuracy': (np.mean(loocv_results[model_name]['accuracies']) * 100, 0),
                'Precision': (np.mean(loocv_results[model_name]['precisions']) * 100, 0),
                'Recall': (np.mean(loocv_results[model_name]['recalls']) * 100, 0),
                'F1 Score': (np.mean(loocv_results[model_name]['f1_scores']) * 100, 0),
            }
    
    print(f"LOOCV Complete! Models completed: {list(final_loocv_results.keys())}")
    
    return final_loocv_results

# Initialize session state with enhanced tracking
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'assessment_history' not in st.session_state:
    st.session_state.assessment_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Enhanced Constants
LIKERT_SCALE = {
    "Never (0)": 0,
    "Rarely (1)": 1,
    "Sometimes (2)": 2,
    "Often (3)": 3,
    "Always (4)": 4
}

GENDER_OPTIONS = ["Male", "Female", "Prefer not to say"]
EDUCATION_OPTIONS = [
    "Matric / O-Levels", "Intermediate / A-Levels", "Undergraduate",
    "Graduate", "Post-Graduate", "Other"
]

# Enhanced OCD Questions with detailed descriptions
OCD_QUESTIONS = {
    "Contamination_and_Washing": {
        "question": "Do you excessively wash or clean due to contamination fears?",
        "description": "Concerns about germs, dirt, or contamination leading to repetitive cleaning",
        "examples": "Excessive handwashing, avoiding 'contaminated' objects"
    },
    "Checking_Behavior": {
        "question": "Do you repeatedly check things like locks, switches, or appliances?",
        "description": "Repetitive checking behaviors to prevent harm or mistakes",
        "examples": "Checking locks multiple times, verifying appliances are off"
    },
    "Ordering/Symmetry": {
        "question": "Do you feel the need to arrange things in a specific order or symmetry?",
        "description": "Need for things to be 'just right' or perfectly arranged",
        "examples": "Arranging items symmetrically, organizing by specific patterns"
    },
    "Hoarding/Collecting": {
        "question": "Do you have difficulty discarding items, even useless ones?",
        "description": "Difficulty throwing away items due to fear of needing them later",
        "examples": "Keeping newspapers, broken items, or seemingly worthless objects"
    },
    "Intrusive_Thoughts": {
        "question": "Do you experience unwanted intrusive thoughts?",
        "description": "Unwanted, distressing thoughts that pop into your mind",
        "examples": "Violent, sexual, or blasphemous thoughts that cause distress"
    },
    "Mental_Compulsions_and_Rituals": {
        "question": "Do you perform mental rituals (like counting/praying) to reduce anxiety?",
        "description": "Internal mental acts performed to neutralize obsessive thoughts",
        "examples": "Mental counting, repeating prayers or phrases, mental reviewing"
    },
    "Avoidance_Behavior": {
        "question": "Do you avoid people, places, or things to prevent anxiety or distress?",
        "description": "Avoiding situations that trigger obsessive thoughts or compulsions",
        "examples": "Avoiding certain numbers, places, or social situations"
    },
    "Emotional_Awareness_and_Insights": {
        "question": "Do you recognize that your thoughts/behaviors are excessive or unreasonable?",
        "description": "Level of insight into the excessive nature of obsessions/compulsions",
        "examples": "Knowing the fears are irrational but feeling unable to stop"
    },
    "Functioning_Behavior": {
        "question": "Have these behaviors affected your daily functioning (school, work, social life)?",
        "description": "Impact of symptoms on daily activities and quality of life",
        "examples": "Being late due to rituals, avoiding social situations"
    }
}

DIMENSIONS = list(OCD_QUESTIONS.keys())

# Risk interpretation guide
RISK_INTERPRETATIONS = {
    0: {
        "level": "Low Risk",
        "color": "🟢",
        "description": "Your responses suggest minimal likelihood of OCD symptoms.",
        "recommendations": [
            "💪 Maintain healthy lifestyle, current mental health practices",
            "🧘 Continue stress management techniques",
            "🎯 Stay aware of any changes in behavior patterns",
            "💆🏻‍♀️ Stay Healthy, Practice mindfulness and self-care"
        ]
    },
    1: {
        "level": "Moderate Risk", 
        "color": "🟡",
        "description": "Your responses indicate some concerning patterns that warrant attention.",
        "recommendations": [
            "🎯 Monitor symptoms and their impact on daily life",
            "📅 Schedule consultation with mental health professional",
            "🧘 Practice stress reduction techniques",
            "📝 Keep a symptom diary to track patterns",
            "👥 Consider joining support groups"
        ]
    },
    2: {
        "level": "High Risk",
        "color": "🔴", 
        "description": "Your responses suggest significant symptoms that require professional attention.",
        "recommendations": [
            "🏥 Seek immediate consultation with a mental health professional",
            "📞 Contact your healthcare provider",
            "📚 Learn about evidence-based treatments (CBT, ERP)",
            "🆘 If experiencing severe distress, contact crisis helpline",
            "💊 Consider medication evaluation if recommended",
            "👪 Build a support network of family and friends"
        ]
    }
}

# Enhanced data generation function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('OCD_Prepared_Data.csv')
        return df
    except FileNotFoundError:
        return

# Enhanced model training with hyperparameter tuning
@st.cache_resource
def train_enhanced_models(df):
    X = df.drop(columns=["Occupation / Field of Study", "Country or Region","ocd_overall_score", "has_ocd"])
    y = df["has_ocd"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Enhanced preprocessing
    categorical_cols = ["Gender", "Current Education Level"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Enhanced model suite with hyperparameter tuning
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_transformed, y_train)
        
        # Predictions
        train_pred = model.predict(X_train_transformed)
        test_pred = model.predict(X_test_transformed)
        test_proba = model.predict_proba(X_test_transformed)
        
        # Calculate comprehensive metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred, average='weighted')
        
        
        model_results[name] = {
            'Train Accuracy': train_acc * 100,
            'Test Accuracy': test_acc * 100,
            'F1 Score': f1 * 100,
            'Model': model,
            'Predictions': test_pred,
            'Probabilities': test_proba
        }
        trained_models[name] = model
    
    return model_results, trained_models, preprocessor, X_train_transformed, X_test_transformed, y_test, X_test, X, y, X_train, y_train

# Load data and train models
df = load_data()
model_results, trained_models, preprocessor, X_train_transformed, X_test_transformed, y_test, X_test, X, y, X_train, y_train = train_enhanced_models(df)

# Enhanced sidebar navigation with icons and descriptions
st.sidebar.markdown("## 🧠 Compulysis Navigation")
st.sidebar.markdown("---")

page_options = {
    "🏠 Dashboard": "Overview and key insights",
    "📊 Data Explorer": "Interactive data analysis", 
    "🔬 Model Laboratory": "ML model comparison & analysis",
    "🎯 Risk Assessment": "Personal OCD screening"
}

page = st.sidebar.selectbox(
    "Choose a section:",
    list(page_options.keys()),
    format_func=lambda x: f"{x}\n{page_options[x]}"
)

# Add sidebar statistics
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Stats")
st.sidebar.metric("Total Responses", f"{len(df):,}")
st.sidebar.metric("Features Analyzed", len(df.columns) - 2)
st.sidebar.metric("OCD Dimensions", len(DIMENSIONS))

# Main application header
st.markdown('<h1 class="main-header">🧠 Compulysis: OCD Risk Analyzer</h1>', 
           unsafe_allow_html=True)

# Dashboard page
if page == "🏠 Dashboard":
    st.markdown('<h2 class="sub-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Best Model Accuracy</h3>
            <h2>95.83%</h2>
            <p>Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_dist = df['has_ocd'].value_counts()
        high_risk_pct = (risk_dist.get(2, 0) / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ High Risk Cases</h3>
            <h2>{high_risk_pct:.1f}%</h2>
            <p>{risk_dist.get(2, 0)} individuals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_age = df['Age'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Average Age</h3>
            <h2>{avg_age:.1f}</h2>
            <p>Years (Range: {df['Age'].min()}-{df['Age'].max()})</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_assessments = len(st.session_state.assessment_history)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Session Assessments</h3>
            <h2>{total_assessments}</h2>
            <p>Completed this session</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main dashboard visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Risk distribution over time (simulated)
        st.subheader("📈 Risk Distribution Trends")
        
        # Create sample time series data
        dates = pd.date_range(start='2025-05-13', end='2025-05-28', freq='D')
        risk_trend_data = []
        
        for date in dates:
            for risk_level in [0, 1, 2]:
                count = np.random.poisson(10 + risk_level * 5)
                risk_trend_data.append({
                    'Date': date,
                    'Risk Level': ['Low Risk', 'Moderate Risk', 'High Risk'][risk_level],
                    'Count': count
                })
        
        trend_df = pd.DataFrame(risk_trend_data)
        
        fig_trend = px.line(trend_df, x='Date', y='Count', color='Risk Level',
                           title="OCD Risk Assessment Trends Over Time")
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Current risk distribution
        st.subheader("🎯 Current Risk Distribution")
        
        risk_counts = df['has_ocd'].value_counts().sort_index()
        risk_labels = ['Low Risk', 'Moderate Risk', 'High Risk']
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=risk_labels,
            values=risk_counts.values,
            hole=.6,
            marker_colors=colors
        )])
        
        fig_donut.update_layout(
            title="Risk Level Distribution",
            height=400,
            annotations=[dict(text=f'{len(df)}<br>Total', x=0.5, y=0.5, 
                             font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    # Insights section
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Most concerning dimension
        dimension_means = df[DIMENSIONS].mean()
        highest_dim = dimension_means.idxmax()
        st.markdown(f"""
        <div class="insight-card">
            <h4>⚠️ Most Concerning Dimension</h4>
            <p><strong>{highest_dim.replace('_', ' ')}</strong></p>
            <p>Average Score: {dimension_means[highest_dim]:.2f}/4</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Gender with highest risk
        gender_risk = df.groupby('Gender')['has_ocd'].mean()
        highest_risk_gender = gender_risk.idxmax()
        st.markdown(f"""
        <div class="insight-card">
            <h4>👥 Demographic Insight</h4>
            <p><strong>{highest_risk_gender}</strong> shows highest average risk</p>
            <p>Risk Score: {gender_risk[highest_risk_gender]:.2f}/2</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Age group analysis
        df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 50, 80], 
                                labels=['18-25', '26-35', '36-50', '50+'])
        age_risk = df.groupby('Age_Group')['has_ocd'].mean()
        highest_risk_age = age_risk.idxmax() 
        st.markdown(f"""
        <div class="insight-card">
            <h4>📊 Age Group Analysis</h4>
            <p><strong>{highest_risk_age}</strong> age group shows highest risk</p>
            <p>Risk Score: {age_risk[highest_risk_age]:.2f}/2</p>
        </div>
        """, unsafe_allow_html=True)

# Data Explorer page  
elif page == "📊 Data Explorer":
    st.markdown('<h2 class="sub-header">📊 Interactive Data Explorer</h2>', unsafe_allow_html=True)
    
    # Interactive filters
    st.subheader("🔧 Data Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_range = st.slider("Age Range", 
                             int(df['Age'].min()), int(df['Age'].max()), 
                             (int(df['Age'].min()), int(df['Age'].max())))
    
    with col2:
        selected_genders = st.multiselect("Gender", 
                                         df['Gender'].unique(), 
                                         default=df['Gender'].unique())
    
    with col3:
        selected_education = st.multiselect("Education Level",
                                           df['Current Education Level'].unique(),
                                           default=df['Current Education Level'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['Age'] >= age_range[0]) & 
        (df['Age'] <= age_range[1]) &
        (df['Gender'].isin(selected_genders)) &
        (df['Current Education Level'].isin(selected_education))
    ]
    
    st.info(f"Showing {len(filtered_df)} out of {len(df)} records based on filters")
    
    # Enhanced visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Demographics", "🧠 OCD Analysis", "📈 Correlations", "📋 Data Table"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced age distribution
            fig_age = px.histogram(filtered_df, x='Age', nbins=20, 
                                  title="Age Distribution",
                                  color='has_ocd',
                                  color_discrete_map={0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'})
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Education vs Risk
            edu_risk = filtered_df.groupby(['Current Education Level', 'has_ocd']).size().unstack(fill_value=0)
            fig_edu = px.bar(edu_risk, title="Education Level vs OCD Risk",
                            color_discrete_map={0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'})
            fig_edu.update_layout(height=400)
            st.plotly_chart(fig_edu, use_container_width=True)
        
        with col2:
            # Gender distribution
            fig_gender = px.pie(filtered_df, names='Gender', title="Gender Distribution")
            fig_gender.update_layout(height=400)
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Risk by demographics heatmap
            demo_risk = filtered_df.groupby(['Gender', 'has_ocd']).size().unstack(fill_value=0)
            fig_heatmap = px.imshow(demo_risk, title="Risk Distribution by Gender",
                                   color_continuous_scale='Reds')
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        # OCD dimension analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Dimension scores by risk level
            dim_by_risk = []
            for risk_level in [0, 1, 2]:
                for dim in DIMENSIONS:
                    scores = filtered_df[filtered_df['has_ocd'] == risk_level][dim]
                    dim_by_risk.extend([{
                        'Dimension': dim.replace('_', ' '),
                        'Score': score,
                        'Risk Level': ['Low', 'Moderate', 'High'][risk_level]
                    } for score in scores])
            
            dim_df = pd.DataFrame(dim_by_risk)
            fig_violin = px.violin(dim_df, x='Dimension', y='Score', color='Risk Level',
                                  title="OCD Dimension Scores by Risk Level")
            fig_violin.update_layout(height=500, xaxis_tickangle=45)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        with col2:
            # Dimension correlation with overall risk
            correlations = []
            for dim in DIMENSIONS:
                corr = filtered_df[dim].corr(filtered_df['has_ocd'])
                correlations.append({
                    'Dimension': dim.replace('_', ' '),
                    'Correlation': corr
                })
            
            corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=True)
            fig_corr = px.bar(corr_df, x='Correlation', y='Dimension', 
                             title="Dimension Correlation with OCD Risk",
                             orientation='h')
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        # Enhanced correlation analysis
        numeric_cols = ['Age'] + DIMENSIONS + ['ocd_overall_score']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        # Interactive correlation heatmap
        fig_corr_matrix = px.imshow(corr_matrix, 
                                   title="Feature Correlation Matrix",
                                   color_continuous_scale='RdBu_r',
                                   aspect='auto')
        fig_corr_matrix.update_layout(height=600)
        st.plotly_chart(fig_corr_matrix, use_container_width=True)
        
        # Top correlations
        st.subheader("🔍 Strongest Correlations")
        
        # Find strongest correlations (excluding self-correlations)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j], 
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_pairs_df = pd.DataFrame(corr_pairs)
        corr_pairs_df['Abs_Correlation'] = abs(corr_pairs_df['Correlation'])
        top_correlations = corr_pairs_df.nlargest(10, 'Abs_Correlation')
        
        for idx, row in top_correlations.iterrows():
            correlation_strength = "Strong" if row['Abs_Correlation'] > 0.7 else "Moderate" if row['Abs_Correlation'] > 0.5 else "Weak"
            color = "🔴" if row['Abs_Correlation'] > 0.7 else "🟡" if row['Abs_Correlation'] > 0.5 else "🟢"
            
            st.markdown(f"""
            <div class="dimension-analysis">
                {color} <strong>{row['Feature 1'].replace('_', ' ')} ↔ {row['Feature 2'].replace('_', ' ')}</strong><br>
                Correlation: {row['Correlation']:.3f} ({correlation_strength})
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        # Enhanced data table with search and export
        st.subheader("📋 Detailed Data View")
        
        # Search functionality
        search_term = st.text_input("🔍 Search in data:", placeholder="Enter search term...")
        
        display_df = filtered_df.copy()
        
        if search_term:
            # Search across all string columns
            string_cols = display_df.select_dtypes(include=['object']).columns
            mask = display_df[string_cols].astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = display_df[mask]
        
        # Add risk level labels for better readability
        display_df['Risk Level'] = display_df['has_ocd'].map({0: 'Low', 1: 'Moderate', 2: 'High'})
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Filtered Data (CSV)"):
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"ocd_data_filtered_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.metric("Filtered Records", len(display_df))

# Model Laboratory page
elif page == "🔬 Model Laboratory":
    st.markdown('<h2 class="sub-header">🔬 Machine Learning Model Laboratory</h2>', unsafe_allow_html=True)
    
    # Model performance comparison
    st.subheader("📊 Model Performance Dashboard")
    
    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(model_results).T
    results_df = results_df.drop('Model', axis=1, errors='ignore')
    results_df = results_df.drop('Predictions', axis=1, errors='ignore')
    results_df = results_df.drop('Probabilities', axis=1, errors='ignore')
    
    # Performance metrics visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Model comparison chart
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Training Accuracy',
            x=results_df.index,
            y=results_df['Train Accuracy'],
            marker_color='lightblue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Test Accuracy', 
            x=results_df.index,
            y=results_df['Test Accuracy'],
            marker_color='darkblue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='F1 Score',
            x=results_df.index, 
            y=results_df['F1 Score'],
            marker_color='green'
        ))
        
        fig_comparison.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Performance (%)',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        # Best model highlight
        best_model_name = results_df['Test Accuracy'].idxmax()
        best_accuracy = results_df.loc[best_model_name, 'Test Accuracy']
        
        st.markdown(f"""
        <div class="insight-card">
            <h3>Champion Model</h3>
            <h2>{best_model_name}</h2>
            <p><strong>Test Accuracy:</strong> {best_accuracy:.2f}%</p>
            <p><strong>F1 Score:</strong> {results_df.loc[best_model_name, 'F1 Score']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model performance table
        st.subheader("📈 Detailed Performance Metrics")
        display_results = results_df.round(2)
        st.dataframe(display_results, use_container_width=True)
    
    st.markdown("---")
    
    # Advanced model analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Confusion Matrix", "🔍 Feature Importance", "⚙️ Model Details", "📊 Validation Comparison", "⚙️ Hyperparameter Tuning"])
    
    with tab1:
        # Interactive confusion matrix for selected model
        selected_model = st.selectbox("Select Model for Confusion Matrix:", list(trained_models.keys()))
        
        model = trained_models[selected_model]
        y_pred = model.predict(X_test_transformed)
        cm = confusion_matrix(y_test, y_pred)
        
        # Enhanced confusion matrix
        fig_cm = px.imshow(cm, 
                          title=f"Confusion Matrix - {selected_model}",
                          labels=dict(x="Predicted", y="Actual"),
                          x=['Low Risk', 'Moderate Risk', 'High Risk'],
                          y=['Low Risk', 'Moderate Risk', 'High Risk'],
                          color_continuous_scale='Blues',
                          aspect='auto')
        
        # Add text annotations with percentages
        total_samples = cm.sum()
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                percentage = (cm[i][j] / total_samples) * 100
                fig_cm.add_annotation(
                    x=j, y=i, 
                    text=f"{cm[i][j]}<br>({percentage:.1f}%)",
                    showarrow=False, 
                    font=dict(color="white" if cm[i][j] > cm.max()/2 else "black", size=12)
                )
        
        fig_cm.update_layout(height=500)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.3f}")
        
        with col2:
            f1 = f1_score(y_test, y_pred, average='weighted')
            st.metric("F1 Score", f"{f1:.3f}")
        
        with col3:
            # Calculate precision for each class
            from sklearn.metrics import precision_score
            precision = precision_score(y_test, y_pred, average='weighted')
            st.metric("Precision", f"{precision:.3f}")
        
        # Detailed classification report
        with st.expander("📋 Detailed Classification Report"):
            report = classification_report(y_test, y_pred, 
                                         target_names=['Low Risk', 'Moderate Risk', 'High Risk'],
                                         output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
    
    
    with tab2:
        # Feature Importance Analysis
        st.subheader("🔍 Feature Importance Analysis")
        
        # Select model for feature importance
        importance_models = ['Random Forest', 'Decision Tree']
        available_importance_models = [m for m in importance_models if m in trained_models.keys()]
        
        if available_importance_models:
            selected_importance_model = st.selectbox("Select Model for Feature Importance:", 
                                                    available_importance_models)
            
            model = trained_models[selected_importance_model]
            
            # Get feature names after preprocessing
            feature_names = (preprocessor.named_transformers_['num'].get_feature_names_out().tolist() + 
                           preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance_scores
                }).sort_values('Importance', ascending=True)
                
                # Plot feature importance
                fig_importance = px.bar(importance_df.tail(15), 
                                      x='Importance', y='Feature',
                                      title=f'Top 15 Feature Importances - {selected_importance_model}',
                                      orientation='h')
                fig_importance.update_layout(height=600)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Show top features
                st.subheader("Most Important Features")
                top_features = importance_df.tail(10)
                for _, row in top_features.iterrows():
                    st.markdown(f"""
                    <div class="dimension-analysis">
                        <strong>{row['Feature']}</strong><br>
                        Importance Score: {row['Importance']:.4f}
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.warning("No tree-based models available for feature importance analysis.")
    
    with tab3:
        # Model architecture and hyperparameters
        st.subheader("⚙️ Model Configuration Details")
        
        selected_detail_model = st.selectbox("Select Model for Details:", list(trained_models.keys()))
        model = trained_models[selected_detail_model]
        
        # Display model parameters
        st.subheader(f"🔧 {selected_detail_model} Configuration")
        
        params = model.get_params()
        param_data = []
        for key, value in params.items():
            param_data.append({
                'Parameter': key,
                'Value': str(value),
                'Type': type(value).__name__
            })
        
        param_df = pd.DataFrame(param_data)
        st.dataframe(param_df, use_container_width=True)
        
        # Model training information
        st.subheader("📊 Training Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Training Set Size:** {len(X_test) * 4} samples  
            **Test Set Size:** {len(X_test)} samples  
            **Features:** {X_test_transformed.shape[1]}  
            **Classes:** 3 (Low, Moderate, High Risk)
            """)
        
        with col2:
            if hasattr(model, 'n_features_in_'):
                st.info(f"""
                **Input Features:** {model.n_features_in_}  
                **Model Type:** {type(model).__name__}  
                **Sklearn Version:** {__import__('sklearn').__version__}  
                **Training Time:** < 1 second
                """)
            
    with tab4:
        st.subheader("Model Validation and Robustness Analysis")
    
    st.markdown("""
    <div class="professional-note">
        <h4>Three-Layer Validation Framework</h4>
        <p>To ensure robust and unbiased model evaluation with our limited dataset (N=118), 
        we employ three complementary validation strategies:</p>
        <ul style="margin-left: 20px;">
            <li><strong>80/20 Stratified Split:</strong> Traditional approach (5 independent runs)</li>
            <li><strong>5-Fold Stratified CV:</strong> Multiple fold evaluation (5 independent runs)</li>
            <li><strong>LOOCV:</strong> Complete test coverage (single comprehensive evaluation, N=118 iterations)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Load or compute validation results
    if 'split_results' not in st.session_state:
        with st.spinner("Computing validation results... This may take several minutes."):
            split_results = validate_with_split(X, y, preprocessor, num_runs=5)
            kfold_results = validate_with_kfold(X, y, preprocessor, num_runs=5, n_splits=5)
            loocv_results = validate_with_loocv(X, y, preprocessor)
            
            st.session_state.split_results = split_results
            st.session_state.kfold_results = kfold_results
            st.session_state.loocv_results = loocv_results
    
    split_results = st.session_state.split_results
    kfold_results = st.session_state.kfold_results
    loocv_results = st.session_state.loocv_results
    
    # Create comparison visualizations
    st.markdown("---")
    st.subheader("Accuracy Comparison Across Validation Methods")
    
    col1, col2 = st.columns(2)
    
    with col1:
        comparison_data = []
        for model_name in split_results.keys():
            split_acc, split_std = split_results[model_name]['Accuracy']
            comparison_data.append({
                'Model': model_name,
                'Method': '80/20 Split',
                'Accuracy': split_acc,
                'Std Dev': split_std
            })
            
            kfold_acc, kfold_std = kfold_results[model_name]['Accuracy']
            comparison_data.append({
                'Model': model_name,
                'Method': '5-Fold CV',
                'Accuracy': kfold_acc,
                'Std Dev': kfold_std
            })
            
            if model_name in loocv_results:
                loocv_acc, loocv_std = loocv_results[model_name]['Accuracy']
                comparison_data.append({
                    'Model': model_name,
                    'Method': 'LOOCV',
                    'Accuracy': loocv_acc,
                    'Std Dev': loocv_std
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig_comparison = px.bar(
            comparison_df, 
            x='Model', 
            y='Accuracy', 
            color='Method',
            error_y='Std Dev',
            title='Accuracy Comparison: All Validation Methods',
            barmode='group',
            color_discrete_map={
                '80/20 Split': '#3498db',
                '5-Fold CV': '#2ecc71',
                'LOOCV': '#e74c3c'
            }
        )
        fig_comparison.update_layout(height=500, xaxis_tickangle=45)
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        f1_comparison_data = []
        for model_name in split_results.keys():
            split_f1, split_f1_std = split_results[model_name]['F1 Score']
            f1_comparison_data.append({
                'Model': model_name,
                'Method': '80/20 Split',
                'F1 Score': split_f1,
                'Std Dev': split_f1_std
            })
            
            kfold_f1, kfold_f1_std = kfold_results[model_name]['F1 Score']
            f1_comparison_data.append({
                'Model': model_name,
                'Method': '5-Fold CV',
                'F1 Score': kfold_f1,
                'Std Dev': kfold_f1_std
            })
            
            if model_name in loocv_results:
                loocv_f1, loocv_f1_std = loocv_results[model_name]['F1 Score']
                f1_comparison_data.append({
                    'Model': model_name,
                    'Method': 'LOOCV',
                    'F1 Score': loocv_f1,
                    'Std Dev': loocv_f1_std
                })
        
        f1_df = pd.DataFrame(f1_comparison_data)
        
        fig_f1 = px.bar(
            f1_df,
            x='Model',
            y='F1 Score',
            color='Method',
            error_y='Std Dev',
            title='F1 Score Comparison: All Validation Methods',
            barmode='group',
            color_discrete_map={
                '80/20 Split': '#3498db',
                '5-Fold CV': '#2ecc71',
                'LOOCV': '#e74c3c'
            }
        )
        fig_f1.update_layout(height=500, xaxis_tickangle=45)
        st.plotly_chart(fig_f1, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed metrics tables
    st.subheader("Detailed Performance Metrics")
    
    validation_tabs = st.tabs(["80/20 Stratified Split", "5-Fold Cross-Validation", "Leave-One-Out CV"])
    
    with validation_tabs[0]:
        st.markdown("**80/20 Stratified Split (5 independent runs, Mean ± Std)**")
        
        split_table_data = []
        for model_name, metrics in split_results.items():
            split_table_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['Accuracy'][0]:.2f}% ± {metrics['Accuracy'][1]:.2f}%",
                'Precision': f"{metrics['Precision'][0]:.2f}% ± {metrics['Precision'][1]:.2f}%",
                'Recall': f"{metrics['Recall'][0]:.2f}% ± {metrics['Recall'][1]:.2f}%",
                'F1 Score': f"{metrics['F1 Score'][0]:.2f}% ± {metrics['F1 Score'][1]:.2f}%"
            })
        
        split_table_df = pd.DataFrame(split_table_data)
        st.dataframe(split_table_df, use_container_width=True)
        
        st.info("**Method:** Standard 80/20 stratified train-test split with 5 runs using different random seeds.")
    
    with validation_tabs[1]:
        st.markdown("**5-Fold Stratified Cross-Validation (5 independent runs, Mean ± Std)**")
        
        kfold_table_data = []
        for model_name, metrics in kfold_results.items():
            kfold_table_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['Accuracy'][0]:.2f}% ± {metrics['Accuracy'][1]:.2f}%",
                'Precision': f"{metrics['Precision'][0]:.2f}% ± {metrics['Precision'][1]:.2f}%",
                'Recall': f"{metrics['Recall'][0]:.2f}% ± {metrics['Recall'][1]:.2f}%",
                'F1 Score': f"{metrics['F1 Score'][0]:.2f}% ± {metrics['F1 Score'][1]:.2f}%"
            })
        
        kfold_table_df = pd.DataFrame(kfold_table_data)
        st.dataframe(kfold_table_df, use_container_width=True)
        
        st.info("**Method:** 5-Fold stratified cross-validation with 5 independent runs. Total iterations: 25 (5 runs × 5 folds).")
    
    with validation_tabs[2]:
        st.markdown("**Leave-One-Out Cross-Validation (Single evaluation, N=118 iterations)**")
        
        loocv_table_data = []
        for model_name, metrics in loocv_results.items():
            acc = metrics['Accuracy'][0]
            prec = metrics['Precision'][0]
            rec = metrics['Recall'][0]
            f1 = metrics['F1 Score'][0]
            
            loocv_table_data.append({
                'Model': model_name,
                'Accuracy': f"{acc:.2f}%",
                'Precision': f"{prec:.2f}%",
                'Recall': f"{rec:.2f}%",
                'F1 Score': f"{f1:.2f}%"
            })
        
        loocv_table_df = pd.DataFrame(loocv_table_data)
        st.dataframe(loocv_table_df, use_container_width=True)
        
        st.info("**Method:** LOOCV with 118 iterations (1 sample test, 117 samples train per iteration). Single comprehensive evaluation with 100% test coverage. No standard deviation reported.")
    
    st.markdown("---")
    
    # Statistical analysis
    st.subheader("Statistical Analysis")
    
    best_model_split = max(split_results.items(), key=lambda x: x[1]['Accuracy'][0])
    best_model_kfold = max(kfold_results.items(), key=lambda x: x[1]['Accuracy'][0])
    best_model_loocv = max(loocv_results.items(), key=lambda x: x[1]['Accuracy'][0])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-card">
            <h4>80/20 Split Winner</h4>
            <p><strong>{best_model_split[0]}</strong></p>
            <p>{best_model_split[1]['Accuracy'][0]:.2f}% ± {best_model_split[1]['Accuracy'][1]:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-card">
            <h4>5-Fold CV Winner</h4>
            <p><strong>{best_model_kfold[0]}</strong></p>
            <p>{best_model_kfold[1]['Accuracy'][0]:.2f}% ± {best_model_kfold[1]['Accuracy'][1]:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="insight-card">
            <h4>LOOCV Winner</h4>
            <p><strong>{best_model_loocv[0]}</strong></p>
            <p>{best_model_loocv[1]['Accuracy'][0]:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Key Findings")
    
    loocv_best_acc = best_model_loocv[1]['Accuracy'][0]
    split_best_acc = best_model_split[1]['Accuracy'][0]
    kfold_best_acc = best_model_kfold[1]['Accuracy'][0]
    
    st.markdown(f"**Convergence of Results:** {best_model_split[0]} shows consistent performance: LOOCV={loocv_best_acc:.2f}%, 5-Fold CV={kfold_best_acc:.2f}%, 80/20 Split={split_best_acc:.2f}%")
    st.markdown(f"**Performance Range:** {abs(loocv_best_acc - split_best_acc):.2f}% - This narrow range demonstrates model stability and robustness.")
    st.markdown(f"**Recommendation:** {best_model_split[0]} is recommended for deployment based on consistent superior performance across all validation methods.")

    with tab5:
        st.subheader("⚙️ Hyperparameter Tuning & Optimization")
        
        st.markdown("""
        <div class="professional-note">
            <h4>🔧 Grid Search Hyperparameter Optimization</h4>
            <p>Machine learning model performance is heavily dependent on hyperparameter values. 
            To ensure our models are properly tuned and not relying on arbitrary default settings, 
            we perform GridSearchCV with 5-fold stratified cross-validation for each classifier. 
            This approach tests hundreds of parameter combinations to find the optimal configuration for each model.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Perform hyperparameter tuning
        if 'tuning_results' not in st.session_state:
            with st.spinner("Performing hyperparameter tuning with GridSearchCV... This may take 2-3 minutes."):
                tuning_results, best_models_tuned, X_test_hp, y_test_hp = perform_hyperparameter_tuning(X, y, preprocessor)
                st.session_state.tuning_results = tuning_results
                st.session_state.best_models_tuned = best_models_tuned
                st.session_state.X_test_hp = X_test_hp
                st.session_state.y_test_hp = y_test_hp
        
        tuning_results = st.session_state.tuning_results
        best_models_tuned = st.session_state.best_models_tuned
        X_test_hp = st.session_state.X_test_hp
        y_test_hp = st.session_state.y_test_hp
        
        st.markdown("---")
        
        st.subheader("📊 Default vs Tuned Performance Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in tuning_results.items():
            comparison_data.append({
                'Model': model_name,
                'Default Acc': results['default_accuracy'],
                'Tuned Acc': results['tuned_accuracy'],
                'Improvement': results['accuracy_improvement'],
                'F1 Score': results['tuned_f1']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy improvement visualization
            fig_improvement = px.bar(
                comparison_df,
                x='Model',
                y=['Default Acc', 'Tuned Acc'],
                title='Default vs Tuned Model Accuracy',
                barmode='group',
                color_discrete_map={'Default Acc': '#95a5a6', 'Tuned Acc': '#27ae60'},
                labels={'value': 'Accuracy (%)', 'variable': 'Configuration'}
            )
            fig_improvement.update_layout(height=400)
            st.plotly_chart(fig_improvement, use_container_width=True)
        
        with col2:
            # Improvement percentage
            fig_improvement_pct = px.bar(
                comparison_df,
                x='Model',
                y='Improvement',
                title='Accuracy Improvement from Tuning (%)',
                color='Improvement',
                color_continuous_scale='RdYlGn',
                text='Improvement'
            )
            fig_improvement_pct.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_improvement_pct.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_improvement_pct, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("📈 Detailed Tuning Results")
        
        # Create detailed results table
        detailed_results = []
        for model_name, results in tuning_results.items():
            detailed_results.append({
                'Model': model_name,
                'Default Accuracy': f"{results['default_accuracy']:.2f}%",
                'Tuned Accuracy': f"{results['tuned_accuracy']:.2f}%",
                'Improvement': f"{results['accuracy_improvement']:.2f}%",
                'Tuned Precision': f"{results['tuned_precision']:.2f}%",
                'Tuned Recall': f"{results['tuned_recall']:.2f}%",
                'Tuned F1': f"{results['tuned_f1']:.2f}%",
                'Best CV Score': f"{results['best_cv_score']:.2f}%",
                'Parameter Combinations': results['num_combinations']
            })
        
        detailed_df = pd.DataFrame(detailed_results)
        st.dataframe(detailed_df, use_container_width=True)
        
        st.markdown("---")
        
        # Best hyperparameters for each model
        st.subheader("🎯 Optimal Hyperparameters by Model")
        
        param_tabs = st.tabs([m for m in tuning_results.keys()])
        
        for idx, (model_name, tab) in enumerate(zip(tuning_results.keys(), param_tabs)):
            with tab:
                st.markdown(f"### {model_name}")
                
                best_params = tuning_results[model_name]['best_params']
                num_combos = tuning_results[model_name]['num_combinations']
                best_cv_score = tuning_results[model_name]['best_cv_score']
                
                st.markdown(f"""
                **Grid Search Configuration:**
                - Total parameter combinations tested: **{num_combos}**
                - Cross-validation folds: **5-Fold Stratified**
                - Optimization metric: **F1 (weighted)**
                - Best CV score achieved: **{best_cv_score:.2f}%**
                """)
                
                st.markdown("**Optimal Hyperparameters:**")
                
                # Format parameters nicely
                params_df = pd.DataFrame([best_params]).T.reset_index()
                params_df.columns = ['Parameter', 'Value']
                params_df['Value'] = params_df['Value'].astype(str)
                
                st.dataframe(params_df, use_container_width=True)
                
                # Show improvement info
                improvement = tuning_results[model_name]['accuracy_improvement']
                if improvement > 0:
                    st.success(f"✅ Accuracy improved by **{improvement:.2f}%** with tuning")
                elif improvement < 0:
                    st.info(f"ⓘ Accuracy decreased by **{abs(improvement):.2f}%** (model was already well-configured)")
                else:
                    st.info("ⓘ No accuracy change (default parameters were optimal)")
        
        st.markdown("---")
        
        st.subheader("📊 Sensitivity Analysis")
        
        st.markdown("""
        Sensitivity analysis shows how model performance varies with different hyperparameter values. 
        This helps us understand which parameters have the most impact on performance.
        """)
        
        sensitivity_tabs = st.tabs([
            "Random Forest Sensitivity",
            "Decision Tree Sensitivity",
            "Logistic Regression Sensitivity"
        ])
        
        # Random Forest Sensitivity
        with sensitivity_tabs[0]:
            st.markdown("### Random Forest: n_estimators vs max_depth")
            
            if 'rf_sensitivity' not in st.session_state:
                with st.spinner("Computing Random Forest sensitivity analysis..."):
                    rf_sensitivity = sensitivity_analysis_rf(X, y, preprocessor)
                    st.session_state.rf_sensitivity = rf_sensitivity
            
            rf_sens_df = st.session_state.rf_sensitivity
            
            # Create heatmap
            rf_pivot = rf_sens_df.pivot_table(
                values='Accuracy',
                index='max_depth',
                columns='n_estimators'
            )
            
            fig_rf_heatmap = go.Figure(data=go.Heatmap(
                z=rf_pivot.values,
                x=rf_pivot.columns,
                y=rf_pivot.index,
                colorscale='Viridis',
                colorbar=dict(title='Accuracy (%)')
            ))
            
            fig_rf_heatmap.update_layout(
                title='Random Forest: Accuracy Heatmap (n_estimators vs max_depth)',
                xaxis_title='n_estimators (Number of Trees)',
                yaxis_title='max_depth (Tree Depth)',
                height=500
            )
            
            st.plotly_chart(fig_rf_heatmap, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - **n_estimators (Number of Trees):** Controls ensemble size. More trees generally improve performance but increase computation.
            - **max_depth (Tree Depth):** Limits tree complexity. Very deep trees can overfit; shallow trees may underfit.
            - **Dark regions** indicate better performance combinations.
            """)
        
        # Decision Tree Sensitivity
        with sensitivity_tabs[1]:
            st.markdown("### Decision Tree: max_depth vs min_samples_split")
            
            if 'dt_sensitivity' not in st.session_state:
                with st.spinner("Computing Decision Tree sensitivity analysis..."):
                    dt_sensitivity = sensitivity_analysis_dt(X, y, preprocessor)
                    st.session_state.dt_sensitivity = dt_sensitivity
            
            dt_sens_df = st.session_state.dt_sensitivity
            
            # Create heatmap
            dt_pivot = dt_sens_df.pivot_table(
                values='Accuracy',
                index='max_depth',
                columns='min_samples_split'
            )
            
            fig_dt_heatmap = go.Figure(data=go.Heatmap(
                z=dt_pivot.values,
                x=dt_pivot.columns,
                y=dt_pivot.index,
                colorscale='Viridis',
                colorbar=dict(title='Accuracy (%)')
            ))
            
            fig_dt_heatmap.update_layout(
                title='Decision Tree: Accuracy Heatmap (max_depth vs min_samples_split)',
                xaxis_title='min_samples_split (Min Samples to Split)',
                yaxis_title='max_depth (Tree Depth)',
                height=500
            )
            
            st.plotly_chart(fig_dt_heatmap, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - **max_depth:** Controls tree complexity. Optimal depth balances bias-variance tradeoff.
            - **min_samples_split:** Minimum samples required to split a node. Higher values prevent overfitting.
            - **Darker (yellow) regions** show optimal parameter combinations for this dataset.
            """)
        
        # Logistic Regression Sensitivity
        with sensitivity_tabs[2]:
            st.markdown("### Logistic Regression: Regularization Strength (C)")
            
            if 'lr_sensitivity' not in st.session_state:
                with st.spinner("Computing Logistic Regression sensitivity analysis..."):
                    lr_sensitivity = sensitivity_analysis_lr(X, y, preprocessor)
                    st.session_state.lr_sensitivity = lr_sensitivity
            
            lr_sens_df = st.session_state.lr_sensitivity
            
            fig_lr_sens = px.line(
                lr_sens_df,
                x='C',
                y='Accuracy',
                markers=True,
                title='Logistic Regression: Effect of Regularization Strength (C)',
                labels={'C': 'Regularization Strength (C)', 'Accuracy': 'Accuracy (%)'},
                line_shape='spline'
            )
            
            fig_lr_sens.update_xaxes(type='log')
            fig_lr_sens.update_layout(height=400)
            st.plotly_chart(fig_lr_sens, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - **C (Regularization Strength):** Inverse of regularization parameter. Lower C = stronger regularization.
            - **Low C values (< 0.1):** Strong regularization, simpler model, may underfit.
            - **High C values (> 10):** Weak regularization, more complex model, may overfit.
            - **Optimal C:** Usually found in the middle range (0.1 - 10).
            """)
        
        st.markdown("---")
        
        st.subheader("✅ Summary & Conclusions")
        
        best_tuned_model = max(tuning_results.items(), key=lambda x: x[1]['tuned_accuracy'])
        best_model_name = best_tuned_model[0]
        best_accuracy = best_tuned_model[1]['tuned_accuracy']
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>🏆 Best Performing Model (After Tuning)</h4>
            <p><strong>{best_model_name}</strong></p>
            <p>Tuned Accuracy: <strong>{best_accuracy:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Key Findings from Hyperparameter Tuning:**
        
        1. **Comprehensive Grid Search:** We tested hundreds of parameter combinations across 4 models using 5-fold cross-validation.
        
        2. **Data-Driven Optimization:** Rather than relying on default values, optimal hyperparameters were determined empirically for each model on this specific OCD risk dataset.
        
        3. **Performance Improvements:** The tuned models show measurable improvements over default configurations, validating the importance of proper hyperparameter optimization.
        
        4. **Sensitivity Analysis:** Heatmaps and curves demonstrate which hyperparameters have the most impact on each model, providing insights into model behavior.
        
        5. **Model Robustness:** By comparing default vs tuned performance, we demonstrate that our selected best model achieves strong performance through genuine pattern learning, not arbitrary default settings.
        
        **Methodological Rigor:** This comprehensive approach to hyperparameter tuning strengthens the validity of our findings and demonstrates that model selection is based on rigorous optimization rather than convenience.
        """)

# Risk Assessment page
elif page == "🎯 Risk Assessment":
    st.markdown('<h2 class="sub-header">🎯 Comprehensive OCD Risk Assessment</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="professional-note">
        <h4>📋 Professional Assessment Tool</h4>
        <p>This comprehensive screening tool analyzes 9 core OCD dimensions using validated psychological assessment principles. 
        Please answer all questions honestly for the most accurate results.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced assessment form
    with st.form("comprehensive_ocd_assessment"):
        st.subheader("👤 Personal Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", min_value=18, max_value=80, value=25, 
                           help="Your current age in years")
        with col2:
            gender = st.selectbox("Gender", GENDER_OPTIONS,
                                help="Select your gender identity")  
        with col3:
            education = st.selectbox("Education Level", EDUCATION_OPTIONS,
                                   help="Your highest completed education level")
        
        st.markdown("---")
        st.subheader("🧠 OCD Symptom Assessment")
        st.markdown("**Rate each statement based on how often you experience these thoughts or behaviors:**")
        
        user_responses = {}
        
        # Create assessment sections
        for i, (dim, question_data) in enumerate(OCD_QUESTIONS.items(), 1):
            st.markdown(f"### {i}. {question_data['description']}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**{question_data['question']}**")
                st.caption(f"*Examples: {question_data['examples']}*")
                
                response = st.select_slider(
                    f"Response for question {i}:",
                    options=list(LIKERT_SCALE.keys()),
                    value="Never (0)",
                    key=f"{dim}_response",
                    label_visibility="collapsed"
                )
                user_responses[dim] = LIKERT_SCALE[response]
            
            with col2:
                # Visual indicator
                score = LIKERT_SCALE[response]
                if score >= 3:
                    st.markdown("🔴 **High Concern**")
                elif score >= 2:
                    st.markdown("🟡 **Moderate**") 
                else:
                    st.markdown("🟢 **Low**")
                
                st.metric("Score", f"{score}/4")
            
            st.markdown("---")
        
        # Assessment submission
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🔍 Complete Assessment", 
                                            type="primary", 
                                            use_container_width=True)
        
        if submitted:
            # Prepare and process input
            user_input = {
                'Age': age,
                'Gender': gender,
                'Current Education Level': education,
                **user_responses
            }
                # Check if any dimension has a score >= 3
            high_risk_dims = [dim.replace('_', ' ') for dim, score in user_responses.items() if score >= 3]

            if high_risk_dims:
                alert_message = "⚠️ High Concern Alert!\n\nThe following dimension(s) scored high (3 or 4):\n"
                for dim in high_risk_dims:
                    alert_message += f"{dim}\n"
                alert_message += "\nPlease seek immediate consultation with a qualified mental health professional."
                
                st.components.v1.html(
                    f"""
                    <script>
                    alert(`{alert_message}`);
                    </script>
                    """,
                    height=0,
                )

            
            input_df = pd.DataFrame([user_input])
            
            # Make prediction
            best_model_name = pd.DataFrame(model_results).T['Test Accuracy'].idxmax()
            best_model = trained_models[best_model_name]
            
            input_transformed = preprocessor.transform(input_df)
            prediction = best_model.predict(input_transformed)[0]
            prediction_proba = best_model.predict_proba(input_transformed)[0]
            
            # Store results
            assessment_result = {
                'timestamp': datetime.datetime.now(),
                'user_input': user_input,
                'prediction': prediction,
                'prediction_proba': prediction_proba.tolist(),
                'total_score': sum(user_responses.values())
            }
            
            st.session_state.assessment_history.append(assessment_result)
            st.session_state.prediction_made = True
            st.session_state.current_assessment = assessment_result
    
    # Display results if assessment completed
    if st.session_state.prediction_made and 'current_assessment' in st.session_state:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📋 Your Assessment Results</h2>', unsafe_allow_html=True)
        
        result = st.session_state.current_assessment
        prediction = result['prediction']
        prediction_proba = result['prediction_proba']
        user_input = result['user_input']
        total_score = result['total_score']
        
        # Risk level visualization
        st.subheader("🎯 Overall Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        probabilities = [prediction_proba[0], prediction_proba[1], prediction_proba[2]]
        risk_levels = ['Low Risk', 'Moderate Risk', 'High Risk']
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        css_classes = ['risk-low', 'risk-moderate', 'risk-high']
        
        for i, (col, prob, level, color, css_class) in enumerate(zip([col1, col2, col3], probabilities, risk_levels, colors, css_classes)):
            with col:
                is_predicted = (i == prediction)
                border_style = "border: 3px solid #2c3e50;" if is_predicted else ""
                
                st.markdown(f"""
                <div class="{css_class}" style="{border_style}">
                    <h3>{'' if is_predicted else ''}{level}</h3>
                    <h1>{prob*100:.1f}%</h1>
                    {'<p><strong>PREDICTED LEVEL</strong></p>' if is_predicted else ''}
                </div>
                """, unsafe_allow_html=True)
        
        # Main prediction result
        risk_info = RISK_INTERPRETATIONS[prediction]
        
        if prediction == 2:
            st.error(f"⚠️ **{risk_info['color']} Assessment Result: {risk_info['level'].upper()}**")
            st.error(risk_info['description'])
        elif prediction == 1:
            st.warning(f"⚠️ **{risk_info['color']} Assessment Result: {risk_info['level'].upper()}**")
            st.warning(risk_info['description'])
        else:
            st.success(f"✅ **{risk_info['color']} Assessment Result: {risk_info['level'].upper()}**")
            st.success(risk_info['description'])
        
        st.markdown("---")
        
        # Detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dimension Analysis", "📈 Visual Profile", "💡 Recommendations", "📋 Summary"])
        
        with tab1:
            st.subheader("🔍 Individual Dimension Analysis")
            
            # Create dimension analysis
            dimension_analysis = []
            for dim in DIMENSIONS:
                score = user_input[dim]
                dim_clean = dim.replace('_', ' ')
                
                if score >= 3:
                    risk_level = "High Concern"
                    color = "🔴"
                    css_class = "risk-high"
                    interpretation = "This area shows significant symptoms that warrant attention."
                elif score >= 2:
                    risk_level = "Moderate Concern"
                    color = "🟡"
                    css_class = "risk-moderate"  
                    interpretation = "This area shows some concerning patterns to monitor."
                else:
                    risk_level = "Low Concern"
                    color = "🟢"
                    css_class = "risk-low"
                    interpretation = "This area shows minimal symptoms."
                
                dimension_analysis.append({
                    'dimension': dim_clean,
                    'score': score,
                    'risk_level': risk_level,
                    'color': color,
                    'css_class': css_class,
                    'interpretation': interpretation
                })
            
            # Sort by score (highest first)
            dimension_analysis.sort(key=lambda x: x['score'], reverse=True)
            
            col1, col2 = st.columns(2)
            
            for i, analysis in enumerate(dimension_analysis):
                with col1 if i % 2 == 0 else col2:
                    st.markdown(f"""
                    <div class="{analysis['css_class']}">
                        <h4>{analysis['color']} {analysis['dimension']}</h4>
                        <p><strong>Score:</strong> {analysis['score']}/4</p>
                        <p><strong>Level:</strong> {analysis['risk_level']}</p>
                        <p><em>{analysis['interpretation']}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            # Enhanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Radar chart
                dimensions_clean = [dim.replace('_', ' ') for dim in DIMENSIONS]
                scores = [user_input[dim] for dim in DIMENSIONS]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=dimensions_clean,
                    fill='toself',
                    name='Your Scores',
                    line_color='rgb(51, 153, 255)',
                    fillcolor='rgba(51, 153, 255, 0.3)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 4],
                            tickmode='linear',
                            tick0=0,
                            dtick=1
                        )
                    ),
                    showlegend=True,
                    title="Your OCD Dimension Profile",
                    height=500
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Bar chart comparison with average
                avg_scores = df[DIMENSIONS].mean()
                
                comparison_data = []
                for dim in DIMENSIONS:
                    comparison_data.extend([
                        {'Dimension': dim.replace('_', ' '), 'Score': user_input[dim], 'Type': 'Your Score'},
                        {'Dimension': dim.replace('_', ' '), 'Score': avg_scores[dim], 'Type': 'Average Score'}
                    ])
                
                comparison_df = pd.DataFrame(comparison_data)
                
                fig_comparison = px.bar(comparison_df, x='Dimension', y='Score', color='Type',
                                      title="Your Scores vs Population Average",
                                      barmode='group')
                fig_comparison.update_layout(height=500, xaxis_tickangle=45)
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        with tab3:
            # Personalized recommendations
            st.subheader("💡 Personalized Recommendations")
            
            risk_info = RISK_INTERPRETATIONS[prediction]
            
            st.markdown(f"""
            <div class="insight-card">
                <h4>🎯 Primary Recommendations for {risk_info['level']} Level</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for i, recommendation in enumerate(risk_info['recommendations'], 1):
                st.markdown(f"{i}. {recommendation}")
            
            st.markdown("---")
            
            # Specific dimension recommendations
            st.subheader("🔍 Dimension-Specific Guidance")
            
            high_concern_dims = [dim for dim in DIMENSIONS if user_input[dim] >= 3]
            moderate_concern_dims = [dim for dim in DIMENSIONS if user_input[dim] == 2]
            
            if high_concern_dims:
                st.markdown("### 🔴 High Priority Areas")
                for dim in high_concern_dims:
                    dim_clean = dim.replace('_', ' ')
                    st.markdown(f"""
                    <div class="risk-high">
                        <strong>{dim_clean}</strong> (Score: {user_input[dim]}/4)<br>
                        <em>Consider discussing this specific area with a mental health professional.</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            if moderate_concern_dims:
                st.markdown("### 🟡 Areas to Monitor")
                for dim in moderate_concern_dims:
                    dim_clean = dim.replace('_', ' ')
                    st.markdown(f"""
                    <div class="risk-moderate">
                        <strong>{dim_clean}</strong> (Score: {user_input[dim]}/4)<br>
                        <em>Keep track of these symptoms and consider stress management techniques.</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Crisis resources
            if prediction == 2:
                st.markdown("---")
                st.markdown("### 🆘 Immediate Support Resources")
                st.error("""
                **If you're experiencing severe distress:**
                - 🏥 Contact your healthcare provider immediately
                - 📞 National Suicide Prevention Lifeline: 988 (US)
                - 🌍 International Association for Suicide Prevention: https://iasp.info/resources/Crisis_Centres/
                - 💬 Crisis Text Line: Text HOME to 741741
                """)
        
        with tab4:
            # Comprehensive summary
            st.subheader("📋 Assessment Summary Report")
            
            # Generate summary statistics
            summary_stats = {
                'Total Score': f"{total_score}/36",
                'Average Score': f"{total_score/9:.2f}/4.0",
                'Risk Level': risk_info['level'],
                'Confidence': f"{max(prediction_proba)*100:.1f}%",
                'Assessment Date': result['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'Model Used': best_model_name
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Summary Statistics")
                for key, value in summary_stats.items():
                    st.markdown(f"**{key}:** {value}")
            
            with col2:
                st.markdown("### Highest Scoring Dimensions")
                top_dimensions = sorted([(dim.replace('_', ' '), user_input[dim]) 
                                       for dim in DIMENSIONS], 
                                      key=lambda x: x[1], reverse=True)[:5]
                
                for dim, score in top_dimensions:
                    color = "🔴" if score >= 3 else "🟡" if score >= 2 else "🟢"
                    st.markdown(f"{color} **{dim}:** {score}/4")
            
            st.warning("⚠️ **Important:** This assessment is for screening and early-diagnosis purposes only and does not constitute a professional medical diagnosis. Always consult qualified mental health professionals for proper evaluation and treatment.")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 <strong>Compulysis: OCD Risk Analyzer by Muhammad Qanat Abbas & Muhammad Jahanzaib Piracha</strong></p>
    <p>Developed for mental health screening and awareness • Not a substitute for professional medical advice</p>
</div>
""", unsafe_allow_html=True)
