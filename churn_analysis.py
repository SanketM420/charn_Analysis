# churn_analysis.py
# Full churn analysis pipeline with visualizations.
# Place this file in your project folder and run: python churn_analysis.py

import os
import sys
import numpy as np
import pandas as pd

# Safe imports with friendly error message if packages missing
try:
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
        roc_curve,
    )
except ModuleNotFoundError as e:
    print("A required Python package is missing:", e.name)
    print("Install required packages with:")
    print("    pip install pandas numpy scikit-learn matplotlib")
    sys.exit(1)


def create_synthetic_dataset(path, n=1000, random_state=42):
    """Create and save a synthetic churn dataset to `path`."""
    np.random.seed(random_state)
    customerID = [f"CUST{10000+i}" for i in range(n)]
    gender = np.random.choice(['Male','Female'], size=n, p=[0.48,0.52])
    senior = np.random.choice([0,1], size=n, p=[0.88,0.12])
    partner = np.random.choice(['Yes','No'], size=n, p=[0.42,0.58])
    dependents = np.random.choice(['Yes','No'], size=n, p=[0.25,0.75])
    tenure = np.random.exponential(scale=24, size=n).astype(int)
    tenure = np.clip(tenure, 0, 72)
    phone_service = np.random.choice(['Yes','No'], size=n, p=[0.9,0.1])
    multiple_lines = np.where(phone_service=='No','No', np.random.choice(['No','Yes'], size=n, p=[0.65,0.35]))
    internet = np.random.choice(['DSL','Fiber optic','No'], size=n, p=[0.35,0.45,0.20])
    online_security = np.where(internet=='No','No', np.random.choice(['No','Yes'], size=n, p=[0.6,0.4]))
    online_backup = np.where(internet=='No','No', np.random.choice(['No','Yes'], size=n, p=[0.6,0.4]))
    device_protection = np.where(internet=='No','No', np.random.choice(['No','Yes'], size=n, p=[0.7,0.3]))
    tech_support = np.where(internet=='No','No', np.random.choice(['No','Yes'], size=n, p=[0.75,0.25]))
    streaming_tv = np.where(internet=='No','No', np.random.choice(['No','Yes'], size=n, p=[0.6,0.4]))
    contract = np.random.choice(['Month-to-month','One year','Two year'], size=n, p=[0.55,0.25,0.20])
    paperless = np.random.choice(['Yes','No'], size=n, p=[0.6,0.4])
    payment = np.random.choice(['Electronic check','Mailed check','Bank transfer','Credit card'], size=n, p=[0.35,0.25,0.2,0.2])

    base = np.random.normal(loc=20, scale=5, size=n)
    internet_add = np.where(internet=='No', 0, np.where(internet=='DSL', 20, 35))
    extras = (online_security=='Yes')*3 + (online_backup=='Yes')*2 + (device_protection=='Yes')*2 + (tech_support=='Yes')*3 + (streaming_tv=='Yes')*4
    monthly_charges = np.round(base + internet_add + extras + np.random.normal(scale=3, size=n), 2)
    total_charges = np.round(monthly_charges * (tenure + 1) + np.random.normal(scale=10, size=n), 2)
    total_charges = np.where(tenure == 0, monthly_charges + np.random.normal(scale=5, size=n), total_charges)

    prob = (
        0.15
        + (contract == 'Month-to-month') * 0.25
        + (monthly_charges > 70) * 0.15
        + (tenure < 6) * 0.12
        + (paperless == 'Yes') * 0.05
        + (payment == 'Electronic check') * 0.05
        - (senior == 1) * 0.03
    )
    prob = np.clip(prob, 0, 0.9)
    churn = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'customerID': customerID,
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': np.where(churn == 1, 'Yes', 'No')
    })

    # Introduce small missingness
    for col in ['TotalCharges', 'OnlineSecurity', 'TechSupport']:
        idx = np.random.choice(df.index, size=max(1, int(0.01 * len(df))), replace=False)
        df.loc[idx, col] = np.nan

    df.to_csv(path, index=False)
    print(f"Created synthetic dataset at: {path}")
    return df


def load_data(path):
    return pd.read_csv(path)


def preprocess(df):
    df = df.copy()
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * (df['tenure'] + 1))
    df['OnlineSecurity'] = df['OnlineSecurity'].fillna('No')
    df['TechSupport'] = df['TechSupport'].fillna('No')
    df['Churn_flag'] = (df['Churn'] == 'Yes').astype(int)
    return df


def prepare_features(df):
    features = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges',
        'Contract', 'PaperlessBilling', 'PaymentMethod',
        'InternetService', 'OnlineSecurity', 'TechSupport', 'StreamingTV'
    ]
    X = df[features].copy()
    y = df['Churn_flag']
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    num_cols = [c for c in ['tenure', 'MonthlyCharges', 'SeniorCitizen'] if c in X.columns]
    if len(num_cols) > 0:
        X[num_cols] = scaler.fit_transform(X[num_cols])
    return X, y, scaler


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'y_test': y_test,
        'y_proba': y_proba,
        'model': model,
        'X_test': X_test
    }
    return results


def save_report(results, out_dir):
    report = f"""Churn Analysis Report
-----------------------
Accuracy: {results['accuracy']:.4f}
Precision: {results['precision']:.4f}
Recall: {results['recall']:.4f}
F1: {results['f1']:.4f}
ROC AUC: {results['roc_auc']:.4f}

Confusion Matrix:
{results['confusion_matrix']}
"""
    with open(os.path.join(out_dir, 'churn_report.txt'), 'w') as f:
        f.write(report)
    print("Saved churn_report.txt")


def make_and_save_plots(df, results, out_dir):
    # 1) Churn count
    plt.figure(figsize=(6, 4))
    df['Churn'].value_counts().plot(kind='bar')
    plt.title('Churn Count')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'churn_count.png'))
    plt.close()

    # 2) Tenure distribution - churned
    plt.figure(figsize=(8, 4))
    df[df['Churn'] == 'Yes']['tenure'].plot(kind='hist', bins=20)
    plt.title('Tenure distribution (Churned)')
    plt.xlabel('Tenure (months)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tenure_hist_churn.png'))
    plt.close()

    # 3) Tenure distribution - retained
    plt.figure(figsize=(8, 4))
    df[df['Churn'] == 'No']['tenure'].plot(kind='hist', bins=20)
    plt.title('Tenure distribution (Retained)')
    plt.xlabel('Tenure (months)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tenure_hist_retained.png'))
    plt.close()

    # 4) Monthly charges boxplot
    plt.figure(figsize=(6, 4))
    df.boxplot(column='MonthlyCharges', by='Churn')
    plt.title('Monthly Charges by Churn')
    plt.suptitle('')
    plt.xlabel('Churn')
    plt.ylabel('Monthly Charges')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'monthly_charges_boxplot.png'))
    plt.close()

    # 5) ROC curve
    try:
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_proba'])
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
        plt.close()
    except Exception:
        print("Skipping ROC curve (not enough class variety).")

    # 6) Feature importance (top coefficients)
    coef_df = pd.DataFrame({'feature': results['X_test'].columns, 'coefficient': results['model'].coef_[0]}).sort_values(by='coefficient', ascending=False)
    plt.figure(figsize=(8, 6))
    top = coef_df.head(12)
    plt.barh(top['feature'][::-1], top['coefficient'][::-1])
    plt.title('Feature Importance (Logistic Regression)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_importance.png'))
    plt.close()

    print("Saved plots: churn_count.png, tenure_hist_churn.png, tenure_hist_retained.png, monthly_charges_boxplot.png, roc_curve.png (if applicable), feature_importance.png")


def main():
    script_folder = os.path.dirname(os.path.abspath(__file__))
    out_dir = script_folder
    csv_path = os.path.join(out_dir, 'churn_synthetic.csv')

    # If CSV missing, create it
    if not os.path.exists(csv_path):
        print("churn_synthetic.csv not found — creating synthetic dataset...")
        create_synthetic_dataset(csv_path)
    else:
        print(f"Found dataset at: {csv_path}")

    # Run pipeline
    df = load_data(csv_path)
    print('Loaded data shape:', df.shape)
    df = preprocess(df)
    print('Missing values after fill (total):', int(df.isna().sum().sum()))
    X, y, scaler = prepare_features(df)
    results = train_and_evaluate(X, y)
    print('Model metrics: Accuracy', results['accuracy'])
    save_report(results, out_dir)

    # Save coefficients CSV
    coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': results['model'].coef_[0]}).sort_values(by='coefficient', ascending=False)
    coef_path = os.path.join(out_dir, 'feature_coefficients.csv')
    coef_df.to_csv(coef_path, index=False)
    print("Saved feature coefficients to:", coef_path)

    # Save plots
    make_and_save_plots(df, {**results, 'X_test': X}, out_dir)
    print("All done. Check the files in:", out_dir)


if __name__ == '__main__':
    main()
