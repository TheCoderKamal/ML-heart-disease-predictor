# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_auc_score, precision_score, recall_score, f1_score, 
                            roc_curve, precision_recall_curve)
from imblearn.over_sampling import ADASYN, SMOTE
from datetime import datetime

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_and_explore_data(file_path):
    """
    Load the dataset and perform basic exploratory analysis
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Basic statistics
        print("\n--- Dataset Summary ---")
        print(df.describe().T)
        
        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\n--- Missing Values ---")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found in the dataset")
        
        # Class distribution
        print("\n--- Class Distribution ---")
        target_col = "Heart Disease Status"
        class_dist = df[target_col].value_counts(normalize=True) * 100
        print(class_dist)
        
        # Check for imbalance
        if class_dist.min() < 30:
            print(f"Warning: Dataset is imbalanced. Minority class is only {class_dist.min():.2f}%")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, target_col="Heart Disease Status"):
    """
    Preprocess the data by handling missing values, encoding categorical features,
    and scaling numerical features
    """
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Encode the target variable first
    label_encoder_target = LabelEncoder()
    df_processed[target_col] = label_encoder_target.fit_transform(df_processed[target_col])
    print(f"Encoded {target_col}: {dict(zip(label_encoder_target.classes_, label_encoder_target.transform(label_encoder_target.classes_)))}")
    
    # Separate features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    print(f"Numerical columns: {len(num_cols)}")
    print(f"Categorical columns: {len(cat_cols)}")
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        # Fill numerical columns with median
        imputer_num = SimpleImputer(strategy='median')
        X[num_cols] = imputer_num.fit_transform(X[num_cols])
        
        # Fill categorical columns with mode
        if len(cat_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
        
        print("Missing values handled successfully")
    
    # Encode categorical features
    encoder_dict = {}
    if len(cat_cols) > 0:
        for col in cat_cols:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])
            encoder_dict[col] = encoder
            print(f"Encoded {col}: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
    
    # Scale numerical features
    scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    print("Numerical features scaled using MinMaxScaler")
    
    # Add target encoder to the encoder dictionary
    encoder_dict['target'] = label_encoder_target
    
    return X, y, scaler, encoder_dict

def handle_imbalanced_data(X, y, method="SMOTE", k_neighbors=5):
    """
    Handle imbalanced dataset using various resampling techniques
    """
    original_class_dist = pd.Series(y).value_counts(normalize=True) * 100
    print(f"\nOriginal class distribution: {dict(original_class_dist.map(lambda x: f'{x:.2f}%'))}")
    
    if method == "SMOTE":
        resampler = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
    elif method == "ADASYN":
        resampler = ADASYN(random_state=RANDOM_SEED, n_neighbors=k_neighbors)
    elif method == "BorderlineSMOTE":
        from imblearn.over_sampling import BorderlineSMOTE
        resampler = BorderlineSMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
    elif method == "None":
        print("Skipping resampling, will use class weights instead")
        return X, y
    else:
        print(f"Warning: Unknown resampling method '{method}'. Using SMOTE as default.")
        resampler = SMOTE(random_state=RANDOM_SEED)
    
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    
    new_class_dist = pd.Series(y_resampled).value_counts(normalize=True) * 100
    print(f"After {method}, class distribution: {dict(new_class_dist.map(lambda x: f'{x:.2f}%'))}")
    print(f"Dataset size: Before={X.shape[0]}, After={X_resampled.shape[0]}")
    
    return X_resampled, y_resampled

def train_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and evaluate their performance
    """
    
    # Initialize models with class weights where supported
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(),  # KNN doesn't support class weights
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED),
        "SVM": SVC(probability=True, class_weight='balanced', random_state=RANDOM_SEED),
        "Naïve Bayes": GaussianNB(),  # NB doesn't support class weights
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_SEED),  # Uses weak learners with their own weighting
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),  # Internal handling of imbalance
        "XGBoost": XGBClassifier(use_label_encoder=False, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), eval_metric='logloss', random_state=RANDOM_SEED)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Performance metrics
        acc = accuracy_score(y_test, y_pred)
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = None
        
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        
        print(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {f'{auc:.4f}' if auc is not None else 'N/A'}")
    
    return results

def perform_cross_validation(models, X, y, cv=5):
    """
    Perform cross-validation for all models
    """
    cv_results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    
    for name, model in models.items():
        print(f"\nPerforming {cv}-fold cross-validation for {name}...")
        
        # Cross-validation for accuracy
        cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        # Cross-validation for AUC if model supports probability predictions
        if hasattr(model, "predict_proba"):
            cv_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
            cv_results[name] = {
                'mean_accuracy': cv_accuracy.mean(),
                'std_accuracy': cv_accuracy.std(),
                'mean_auc': cv_auc.mean(),
                'std_auc': cv_auc.std()
            }
            print(f"{name} - CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}, CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
        else:
            cv_results[name] = {
                'mean_accuracy': cv_accuracy.mean(),
                'std_accuracy': cv_accuracy.std()
            }
            print(f"{name} - CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    
    return cv_results

def tune_hyperparameters(models, X_train, y_train):
    """
    Perform hyperparameter tuning for selected models
    """
    # Define parameter grids for selected models
    param_grids = {
        "Random Forest": {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "SVM": {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        },
        "K-Nearest Neighbors": {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1: Manhattan, 2: Euclidean
        },
        "XGBoost": {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
            'subsample': [0.6, 0.8, 1.0]
        }
    }
    
    best_models = {}
    for name, param_grid in param_grids.items():
        print(f"\nTuning hyperparameters for {name}...")
        
        # Skip if model is not in the provided list
        if name not in models:
            print(f"Skipping {name} as it's not in the provided models")
            continue
        
        base_model = models[name]['model']
        
        # Use RandomizedSearchCV for efficiency
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=20,
            scoring='f1',  # Using F1 score for balanced precision and recall
            cv=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        print(f"Best parameters for {name}: {search.best_params_}")
        print(f"Best CV score for {name}: {search.best_score_:.4f}")
        
        best_models[name] = {
            'model': search.best_estimator_,
            'params': search.best_params_,
            'score': search.best_score_
        }
    
    return best_models

def evaluate_best_models(best_models, X_test, y_test):
    """
    Evaluate the performance of the best models after hyperparameter tuning
    """
    results = {}
    for name, model_info in best_models.items():
        model = model_info['model']
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Performance metrics
        acc = accuracy_score(y_test, y_pred)
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = None
        
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Classification report and confusion matrix
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'report': report,
            'confusion_matrix': cm
        }
        
        print(f"\n--- Evaluation for {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        if auc is not None:
            print(f"AUC-ROC: {auc:.4f}")
        else:
            print("AUC-ROC: N/A")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

def analyze_feature_importance(X, best_model):
    """
    Analyze feature importance for tree-based models
    """
    # Check if the model has feature_importances_ attribute
    if hasattr(best_model, 'feature_importances_'):
        # Get feature importances
        importances = best_model.feature_importances_
        
        # Create a DataFrame to store feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        print("\n--- Feature Importance ---")
        print(feature_importance_df)
        
        # Create a bar plot of feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        
        # Create models/plots directory if it doesn't exist
        if not os.path.exists('models/plots'):
            os.makedirs('models/plots')
            
        plt.savefig('models/plots/feature_importance.png')
        print("Feature importance plot saved to models/plots/feature_importance.png")
        
        return feature_importance_df
    else:
        print("The selected model doesn't support feature importance analysis")
        return None

def save_model(model, model_name, scaler=None, encoders=None):
    """
    Save the trained model and preprocessing objects
    """
    # Create a directory to save models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Create a timestamp for the model version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a dictionary with all objects to save
    model_data = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'model_name': model_name,
        'created_at': timestamp,
        'features': list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else None
    }
    
    # Save the model and preprocessing objects
    filename = f"models/{model_name}_{timestamp}.pkl"
    with open(filename, "wb") as file:
        pickle.dump(model_data, file)
    
    print(f"Model saved successfully: {filename}")
    return filename

def plot_roc_curves(best_models, X_test, y_test):
    """
    Plot ROC curves for all best models
    """
    plt.figure(figsize=(12, 8))
    
    for name, model_info in best_models.items():
        model = model_info['model']
        
        if hasattr(model, "predict_proba"):
            # Get predicted probabilities
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
    
    # Plot diagonal line for reference (random classifier)
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Set plot details
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Best Models')
    plt.legend(loc='lower right')
    
    # Create models/plots directory if it doesn't exist
    if not os.path.exists('models/plots'):
        os.makedirs('models/plots')
        
    plt.savefig('models/plots/roc_curves.png')
    print("ROC curves plot saved to models/plots/roc_curves.png")

def main():
    """
    Main function to execute the heart disease prediction pipeline
    """
    # Step 1: Load and explore data
    file_path = "Indian_heart_disease.csv"  # Update path if necessary
    df = load_and_explore_data(file_path)
    if df is None:
        print("Error: Could not load the dataset")
        return
    
    # Step 2: Preprocess data
    X, y, scaler, encoders = preprocess_data(df)
    
    # Step 3: Handle imbalanced data
    X_resampled, y_resampled = handle_imbalanced_data(X, y, method="SMOTE")
    
    # Step 4: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=RANDOM_SEED, stratify=y_resampled
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Step 5: Train and evaluate multiple models
    model_results = train_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Step 6: Perform cross-validation
    cv_results = perform_cross_validation(
        {name: model_info['model'] for name, model_info in model_results.items()}, 
        X_resampled, y_resampled
    )
    
    # Step 7: Select the best models for hyperparameter tuning
    # Choose models with high CV scores or specific models of interest
    models_to_tune = {
        "Random Forest": model_results["Random Forest"],
        "XGBoost": model_results["XGBoost"],
        "SVM": model_results["SVM"],
        "K-Nearest Neighbors": model_results["K-Nearest Neighbors"]
    }
    
    # Step 8: Tune hyperparameters
    best_models = tune_hyperparameters(models_to_tune, X_train, y_train)
    
    # Step 9: Evaluate best models
    best_model_results = evaluate_best_models(best_models, X_test, y_test)
    
    # Step 10: Plot ROC curves for best models
    plot_roc_curves(best_models, X_test, y_test)
    
    # Step 11: Select the final model (e.g., the one with highest F1 score)
    best_model_name = max(best_model_results, key=lambda x: best_model_results[x]['f1_score'])
    final_model = best_models[best_model_name]['model']
    print(f"\nFinal selected model: {best_model_name}")
    
    # Step 12: Analyze feature importance for the final model
    feature_importance = analyze_feature_importance(X, final_model)
    
    # Step 13: Save the final model
    save_model(final_model, best_model_name, scaler, encoders)
    
    print("\nHeart disease prediction model development completed successfully!")

if __name__ == "__main__":
    main()