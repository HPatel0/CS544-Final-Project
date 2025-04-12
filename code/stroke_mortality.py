from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv
import os
from IPython.display import display


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
SELECT sc.subject_id, p.gender, p.dob, a.admittime,
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
