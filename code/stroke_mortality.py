from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import lightgbm as lgbm


load_dotenv()


DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)


# SQL queries

# get stroke patients into a table called stroke cohort 
drop_stroke_cohort = """DROP TABLE IF EXISTS stroke_cohort;"""
create_stroke_cohort = """
SELECT DISTINCT icu.subject_id, icu.hadm_id, icu.icustay_id, d.icd9_code
INTO stroke_cohort
FROM mimiciii.diagnoses_icd d
JOIN mimiciii.icustays icu ON d.hadm_id = icu.hadm_id
WHERE d.icd9_code = '430' OR d.icd9_code = '431'
   OR d.icd9_code = '433' OR d.icd9_code = '434';
"""

# get the basic demographics of stroke patients into a table called stroke demographics

# for age calculation --> https://stackoverflow.com/questions/1572110/how-to-calculate-age-in-years-based-on-date-of-birth-and-getdate
drop_demographics = """DROP TABLE IF EXISTS stroke_demographics;"""
create_demographics = """
SELECT sc.subject_id, p.gender, p.dob, a.ethnicity, a.admittime,
       ROUND(EXTRACT(YEAR FROM age(a.admittime, p.dob))) AS age
INTO stroke_demographics
FROM stroke_cohort sc
JOIN mimiciii.patients p ON sc.subject_id = p.subject_id
JOIN mimiciii.admissions a ON sc.hadm_id = a.hadm_id;
"""

# get whether or not select stroke patients have died or not into a table called stroke_labels
drop_labels = """DROP TABLE IF EXISTS stroke_labels;"""
create_labels = """
SELECT sc.subject_id, sc.hadm_id,
       CASE WHEN a.hospital_expire_flag = 1 THEN 1 ELSE 0 END AS mortality
INTO stroke_labels
FROM stroke_cohort sc
JOIN mimiciii.admissions a ON sc.hadm_id = a.hadm_id;
"""

# Prescriptions (antiplatelets and anticoagulants)
drop_prescriptions = """DROP TABLE IF EXISTS stroke_prescriptions;"""
create_prescriptions = """
SELECT DISTINCT sc.subject_id, sc.hadm_id,
       CASE
         WHEN LOWER(p.drug) LIKE '%aspirin%' OR LOWER(p.drug) LIKE '%clopidogrel%' THEN 1
         ELSE 0
       END AS antiplatelet,
       CASE
         WHEN LOWER(p.drug) LIKE '%heparin%' OR LOWER(p.drug) LIKE '%warfarin%' OR LOWER(p.drug) LIKE '%enoxaparin%' THEN 1
         ELSE 0
       END AS anticoagulant
INTO stroke_prescriptions
FROM stroke_cohort sc
JOIN mimiciii.prescriptions p ON sc.hadm_id = p.hadm_id;
"""

# sql queries for lab events, first one we are extracting coagulation labs such as platelet count, aptt and inr test
create_coag_lab = """
SELECT le.subject_id, le.hadm_id, le.itemid, le.charttime, le.valuenum, le.valueuom
FROM mimiciii.labevents le
JOIN mimiciii.icustays icu ON le.hadm_id = icu.hadm_id
WHERE le.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
  AND le.itemid IN (
    51237, 
    51275, 
    51265  
  );
"""

# sql queries for lab events, we are extracting metabolic panel sodium and potassium
create_metabolic_lab = """

SELECT le.subject_id, le.hadm_id, le.itemid, le.charttime, le.valuenum, le.valueuom
FROM mimiciii.labevents le
JOIN mimiciii.icustays icu ON le.hadm_id = icu.hadm_id
WHERE le.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
  AND le.itemid IN (
    50983, 
    50971  
  );
"""

# sql queries for lab events, we are extracting cbc such as hemoglobin and wbc
create_cbc_lab = """

SELECT le.subject_id, le.hadm_id, le.itemid, le.charttime, le.valuenum, le.valueuom
FROM mimiciii.labevents le
JOIN mimiciii.icustays icu ON le.hadm_id = icu.hadm_id
WHERE le.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
  AND le.itemid IN (
    51222, 
    51300  
  );

"""


# testing sql queries
with engine.connect() as conn:
    
    # stroke_cohort
    conn.execute(text(drop_stroke_cohort))
    conn.execute(text(create_stroke_cohort))
    stroke_df = pd.read_sql(text("SELECT * FROM stroke_cohort;"), conn)
    print("Stroke Cohort:\n", stroke_df)
    stroke_df.to_csv("stroke_df_dataset.csv", index=False)

    # stroke_demographics
    conn.execute(text(drop_demographics))
    conn.execute(text(create_demographics))
    demo_df = pd.read_sql(text("SELECT * FROM stroke_demographics;"), conn)
    print("Demographics:\n", demo_df)
    demo_df.to_csv("demo_df_dataset.csv", index=False)

    # stroke_labels
    conn.execute(text(drop_labels))
    conn.execute(text(create_labels))
    labels_df = pd.read_sql(text("SELECT * FROM stroke_labels;"), conn)
    print("Labels:\n", labels_df)
    labels_df.to_csv("labels_df_dataset.csv", index=False)

    # stroke_prescriptions
    conn.execute(text(drop_prescriptions))
    conn.execute(text(create_prescriptions))
    rx_df = pd.read_sql(text("SELECT * FROM stroke_prescriptions;"), conn)
    print("Prescriptions:\n", rx_df)
    rx_df.to_csv("rx_df.csv", index=False)

    # lab events coag
    coag_df = pd.read_sql(text(create_coag_lab), conn)
    print("Coagulation Labs:\n", coag_df.head())
    coag_df.to_csv("coag_df.csv", index=False)

    # lab events metabolic
    metabolic_df = pd.read_sql(text(create_metabolic_lab), conn)
    print("Metabolic Panel Labs:\n", metabolic_df.head())
    metabolic_df.to_csv("metabolic_df.csv", index=False)

    # lab events cbc
    cbc_df = pd.read_sql(text(create_cbc_lab), conn)
    print("CBC Labs:\n", cbc_df.head())
    cbc_df.to_csv("cbc_df_dataset.csv", index=False)


# now its time for training classification models
# we want to merge all tables into one based on hadm id and subject id 
def process_lab_data(lab_df, itemid_map):
    """
    Process lab data to get first value for each test per patient
    itemid_map: Dictionary mapping itemid to test name
    """
    lab_df = lab_df[lab_df['itemid'].isin(itemid_map.keys())]
    
    lab_df = lab_df.sort_values(['subject_id', 'hadm_id', 'itemid', 'charttime'])
    
    lab_df = lab_df.groupby(['subject_id', 'hadm_id', 'itemid']).first().reset_index()
    
    lab_df['test_name'] = lab_df['itemid'].map(itemid_map)
    lab_pivot = lab_df.pivot_table(
        index=['subject_id', 'hadm_id'],
        columns='test_name',
        values='valuenum',
        aggfunc='first'
    ).reset_index()
    
    return lab_pivot

coag_map = {
    51237: 'inr',
    51275: 'aptt',
    51265: 'platelet_count'
}

metabolic_map = {
    50983: 'sodium',
    50971: 'potassium'
}

cbc_map = {
    51222: 'hemoglobin',
    51300: 'wbc'
}


# helper func to process ethnicities

def process_ethnicity(df):
    df = df.copy()
    
    value_counts = df['ethnicity'].value_counts()
    rare_categories = value_counts[value_counts < 20].index.tolist()
    
    df['ethnicity_processed'] = df['ethnicity'].apply(
        lambda x: 'OTHER' if x in rare_categories else x
    )
    
    # one hot encoding 
    ethnicity_dummies = pd.get_dummies(
        df['ethnicity_processed'],
        prefix='ethnicity'
    )
    
    df = df.drop(columns=['ethnicity', 'ethnicity_processed'])
    
    return pd.concat([df, ethnicity_dummies], axis=1)

# Process each lab DataFrame
coag_processed = process_lab_data(coag_df, coag_map)
metabolic_processed = process_lab_data(metabolic_df, metabolic_map)
cbc_processed = process_lab_data(cbc_df, cbc_map)


final_df = stroke_df.copy()

# Merge demographics
final_df = final_df.merge(
    demo_df[['subject_id', 'gender', 'age', 'ethnicity']],
    on='subject_id',
    how='left'
)

# add ethnicity
final_df = final_df.drop(columns=[col for col in final_df.columns if col.startswith('ethnicity_')])
final_df = process_ethnicity(final_df)


# Merge mortality labels
final_df = final_df.merge(
    labels_df[['subject_id', 'hadm_id', 'mortality']],
    on=['subject_id', 'hadm_id'],
    how='left'
)

# Merge prescriptions
final_df = final_df.merge(
    rx_df[['subject_id', 'hadm_id', 'antiplatelet', 'anticoagulant']],
    on=['subject_id', 'hadm_id'],
    how='left'
)

final_df[['antiplatelet', 'anticoagulant']] = final_df[['antiplatelet', 'anticoagulant']].fillna(0)

# Merge lab data one by one
final_df = final_df.merge(
    coag_processed,
    on=['subject_id', 'hadm_id'],
    how='left'
)

final_df = final_df.merge(
    metabolic_processed,
    on=['subject_id', 'hadm_id'],
    how='left'
)

final_df = final_df.merge(
    cbc_processed,
    on=['subject_id', 'hadm_id'],
    how='left'
)

final_df['icd9_code'] = final_df['icd9_code'].str.extract('(\d+)').astype(float) #apparently xgboost treats icd9 code as string/object so we converting it 


# Convert gender to binary (0/1)
final_df['gender'] = final_df['gender'].map({'M': 0, 'F': 1})

print("Missing values per column:")
print(final_df.isnull().sum())

# filling missing values for median
for col in ['platelet_count', 'aptt', 'inr', 'potassium', 'sodium', 'hemoglobin', 'wbc']:
    final_df[col].fillna(final_df[col].median(), inplace=True)
final_df.dropna(inplace=True)

# verifying final dataset
print("\nFinal dataset shape:", final_df.shape)
display(final_df.head())

final_df.to_csv("final_stroke_dataset.csv", index=False)


# time to train
ethnicity_cols = [col for col in final_df.columns if col.startswith('ethnicity_')]
print(final_df.filter(like='ethnicity').head())


X = final_df[['gender', 'age', 'icd9_code', 'antiplatelet', 'anticoagulant',
              'aptt', 'inr', 'platelet_count', 'potassium', 'sodium',
              'hemoglobin', 'wbc'] + ethnicity_cols]
y = final_df['mortality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    if y_probs is not None:
        print(f"ROC AUC: {roc_auc_score(y_test, y_probs):.3f}")
        print(f"Average Precision: {average_precision_score(y_test, y_probs):.3f}")
        
        # Plot PR curve
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + "="*50 + "\n")


# logistic regression

print("Training Logistic Regression...")
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, X_test, y_test)


# https://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work
# random forest
print("Training Random Forest...")
classes = np.unique(y_train)
weights = class_weight.compute_class_weight(
    'balanced', classes=classes, y=y_train
)
class_weights = dict(zip(classes, weights))

rf_model = RandomForestClassifier(
    class_weight=class_weights,
    random_state=42,
    n_estimators=200,
    max_depth=10,
    min_samples_split=5
)
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, X_test, y_test)


# lightgbm
print("\nTraining LightGBM...")
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
lgbm_model = LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    num_leaves=31,           
    max_depth=-1,            
    learning_rate=0.05,      
    n_estimators=200,        
    class_weight='balanced', 
    reg_alpha=0.1,           
    reg_lambda=0.1,          
    random_state=42,
    n_jobs=-1,               
    importance_type='gain'   
)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='aucpr',     
    callbacks=[
        lgbm.early_stopping(stopping_rounds=20, verbose=True),
        lgbm.log_evaluation(period=20)
    ]
)
evaluate_model(lgbm_model, X_test, y_test)


# feature importance analysis
models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "LightGBM": lgbm_model
}

for name, model in models.items():
    print(f"\n{name} Feature Importance:")
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        print(importances.sort_values(ascending=False).head(10))
    elif hasattr(model, 'coef_'):
        coef = pd.Series(model.coef_[0], index=X.columns)
        print(coef.sort_values(key=abs, ascending=False).head(10))