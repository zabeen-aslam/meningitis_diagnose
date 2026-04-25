from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="🧠 Meningitis Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Feature Names ─────────────────────────────────────────────
FEATURE_NAMES = [
    'Age', 'Gender', 'Vaccination_Status', 'Has_Comorbidity',
    'Is_Diabetic', 'Is_HIV', 'Is_Hypertensive',
    'Previous_Meningitis_History', 'Petechiae', 'Seizures',
    'Altered_Mental_Status', 'GCS_Score', 'Procalcitonin',
    'CRP_Level', 'Blood_WBC_Count', 'CSF_WBC_Count',
    'CSF_Glucose', 'CSF_Protein', 'CSF_to_Blood_Glucose_Ratio',
    'CSF_Neutrophils_%', 'CSF_Lymphocytes_%', 'CSF_Culture_Result',
    'Protein_to_Glucose_Ratio', 'WBC_Blood_to_CSF_Ratio',
    'Neutrophil_Lymphocyte_Ratio', 'CRP_to_Procalcitonin_Ratio'
]

# ── Patient Input Schema ──────────────────────────────────────
class PatientInput(BaseModel):
    Age: float
    Gender: str
    Vaccination_Status: str
    Comorbidities: str
    Previous_Meningitis_History: str
    Petechiae: str
    Seizures: str
    Altered_Mental_Status: str
    GCS_Score: float
    Procalcitonin: float
    CRP_Level: float
    Blood_WBC_Count: float
    CSF_WBC_Count: float
    CSF_Glucose: float
    CSF_Protein: float
    CSF_to_Blood_Glucose_Ratio: float
    CSF_Neutrophils_pct: float
    CSF_Lymphocytes_pct: float
    CSF_Culture_Result: str

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 36,
                "Gender": "Female",
                "Vaccination_Status": "Full",
                "Comorbidities": "Diabetes",
                "Previous_Meningitis_History": "No",
                "Petechiae": "No",
                "Seizures": "Yes",
                "Altered_Mental_Status": "Yes",
                "GCS_Score": 8,
                "Procalcitonin": 0.36,
                "CRP_Level": 25.0,
                "Blood_WBC_Count": 13700,
                "CSF_WBC_Count": 460,
                "CSF_Glucose": 51.0,
                "CSF_Protein": 115.0,
                "CSF_to_Blood_Glucose_Ratio": 0.55,
                "CSF_Neutrophils_pct": 30.0,
                "CSF_Lymphocytes_pct": 70.0,
                "CSF_Culture_Result": "Negative"
            }
        }

# ── Preprocess ────────────────────────────────────────────────
def preprocess(df, fit_encoders=False):
    df = df.copy()
    df['Comorbidities'].fillna('None', inplace=True)
    df['Stage_Prediction'].fillna('Stage II', inplace=True)
    df.loc[df['CSF_Neutrophils_%'] < 0,   'CSF_Neutrophils_%'] = 0
    df.loc[df['CSF_Lymphocytes_%'] > 100, 'CSF_Lymphocytes_%'] = 100
    df.loc[df['Procalcitonin'] < 0,       'Procalcitonin']     = 0.01
    df['Protein_to_Glucose_Ratio']    = df['CSF_Protein']       / (df['CSF_Glucose'] + 0.01)
    df['WBC_Blood_to_CSF_Ratio']      = df['Blood_WBC_Count']   / (df['CSF_WBC_Count'] + 1)
    df['Neutrophil_Lymphocyte_Ratio'] = df['CSF_Neutrophils_%'] / (df['CSF_Lymphocytes_%'] + 0.01)
    df['CRP_to_Procalcitonin_Ratio']  = df['CRP_Level']         / (df['Procalcitonin'] + 0.01)
    for col, mapping in {
        'Gender':                      {'Male': 1, 'Female': 0},
        'Vaccination_Status':          {'Full': 2, 'Partial': 1, 'Unknown': 0},
        'Previous_Meningitis_History': {'Yes': 1, 'No': 0},
        'Petechiae':                   {'Yes': 1, 'No': 0},
        'Seizures':                    {'Yes': 1, 'No': 0},
        'Altered_Mental_Status':       {'Yes': 1, 'No': 0},
        'CSF_Culture_Result':          {'Positive': 1, 'Negative': 0}
    }.items():
        df[col] = df[col].map(mapping)
    df['Has_Comorbidity'] = (df['Comorbidities'] != 'None').astype(int)
    df['Is_Diabetic']     = (df['Comorbidities'] == 'Diabetes').astype(int)
    df['Is_HIV']          = (df['Comorbidities'] == 'HIV').astype(int)
    df['Is_Hypertensive'] = (df['Comorbidities'] == 'Hypertension').astype(int)
    return df[FEATURE_NAMES], df

# ── ClinicalValidator ─────────────────────────────────────────
class ClinicalValidator:
    def validate_bacterial(self, r):
        s, w = 0, 0
        s += 3 if r['Procalcitonin'] > 2.0 else 2 if r['Procalcitonin'] > 0.235 else 0; w += 3
        s += 2 if r['CRP_Level'] > 50 else 0;               w += 2
        s += 2 if r['CSF_WBC_Count'] > 1500 else 0;         w += 2
        s += 2 if r['CSF_Neutrophils_%'] > 50 else 0;       w += 2
        s += 1.5 if r['CSF_Glucose'] < 40 else 0;           w += 1.5
        s += 1.5 if r['CSF_to_Blood_Glucose_Ratio'] < 0.4 else 0; w += 1.5
        return s / w if w > 0 else 0

    def validate_viral(self, r):
        s, w = 0, 0
        s += 3 if r['Procalcitonin'] < 0.5 else 0;          w += 3
        s += 2 if r['CRP_Level'] < 20 else 0;               w += 2
        s += 1.5 if r['CSF_WBC_Count'] < 500 else 0;        w += 1.5
        s += 2 if r['CSF_Lymphocytes_%'] > 50 else 0;       w += 2
        s += 1.5 if r['CSF_Glucose'] >= 40 else 0;          w += 1.5
        s += 1 if r['GCS_Score'] >= 13 else 0;              w += 1
        return s / w if w > 0 else 0

    def validate_tb(self, r):
        s, w = 0, 0
        s += 3 if r['CSF_Protein'] > 500 else 2 if r['CSF_Protein'] > 100 else 0; w += 3
        s += 2 if r['Procalcitonin'] < 1.0 else 0;          w += 2
        s += 2 if r['CSF_Glucose'] < 45 else 0;             w += 2
        s += 2 if r['CSF_to_Blood_Glucose_Ratio'] < 0.5 else 0; w += 2
        s += 1.5 if r['CSF_Lymphocytes_%'] > 50 else 0;     w += 1.5
        return s / w if w > 0 else 0

    def get_stage(self, r):
        g = r['GCS_Score']
        return 'Stage I' if g >= 14 else 'Stage II' if g >= 10 else 'Stage III'

    def predict(self, r):
        scores = {
            'Bacterial':   self.validate_bacterial(r),
            'Viral':       self.validate_viral(r),
            'Tuberculous': self.validate_tb(r)
        }
        d = max(scores, key=scores.get)
        return {'diagnosis': d, 'stage': self.get_stage(r),
                'confidence': scores[d], 'all_scores': scores}
# ── Fix for Railway deployment ────────────────────────────────
import sys
sys.modules['__main__'].ClinicalValidator = ClinicalValidator
# ─────────────────────────────────────────────────────────────
# ── Load Model ────────────────────────────────────────────────
bundle = joblib.load('meningitis_model_final.pkl')
print('✅ Model loaded successfully')

# ── Risk Helper ───────────────────────────────────────────────
def get_risk(stage: str) -> dict:
    s = stage.lower()
    if "3" in s or "iii" in s:
        return {
            "risk_level":   "High",
            "risk_color":   "red",
            "risk_heading": "High Risk — Immediate medical attention required",
            "risk_message": "Advanced-stage meningitis detected. Please go to the nearest emergency room immediately."
        }
    elif "2" in s or "ii" in s:
        return {
            "risk_level":   "Moderate",
            "risk_color":   "blue",
            "risk_heading": "Moderate Risk — Please visit your doctor soon",
            "risk_message": "Moderate-stage meningitis indicators detected. Please consult a doctor promptly."
        }
    else:
        return {
            "risk_level":   "Low",
            "risk_color":   "green",
            "risk_heading": "Low Risk — Early stage detected",
            "risk_message": "Early stage indicators. Monitor closely and follow doctor's advice."
        }

# ── predict_patient ───────────────────────────────────────────
def predict_patient(patient_dict, bundle):
    sc   = bundle['scaler']
    le_d = bundle['label_encoder_diagnosis']
    le_s = bundle['label_encoder_stage']
    m_d  = bundle['model_diagnosis']
    m_s  = bundle['model_stage']
    val  = bundle['clinical_validator']

    df_p = pd.DataFrame([patient_dict])
    df_p['Meningitis_Diagnosis'] = 'Bacterial'
    df_p['Stage_Prediction']     = 'Stage I'
    X_p, df_pp = preprocess(df_p)
    X_p  = X_p.fillna(0)
    X_sc = sc.transform(X_p)

    diag_proba = m_d.predict_proba(X_sc)[0]
    diag_idx   = int(np.argmax(diag_proba))
    ml_diag    = le_d.classes_[diag_idx]
    ml_conf    = float(diag_proba[diag_idx])

    stage_proba = m_s.predict_proba(X_sc)[0]
    stage_idx   = int(np.argmax(stage_proba))
    ml_stage    = le_s.classes_[stage_idx]

    row  = df_pp.iloc[0]
    clin = val.predict(row)

    # GCS Stage Override
    gcs = patient_dict.get('GCS_Score', 15)
    if gcs >= 14:
        final_stage    = 'Stage I'
        stage_override = True
    elif gcs < 10:
        final_stage    = 'Stage III'
        stage_override = True
    else:
        clin_stage  = clin['stage']
        final_stage = 'Stage II' if (ml_stage == 'Stage II' or clin_stage == 'Stage II') else clin_stage
        stage_override = False

    # Diagnosis Consensus
    agree = (ml_diag == clin['diagnosis'])
    if agree and ml_conf > 0.6 and clin['confidence'] > 0.5:
        final_diag, final_conf, consensus = ml_diag, (ml_conf + clin['confidence']) / 2, 'STRONG'
    elif agree:
        final_diag, final_conf, consensus = ml_diag, (ml_conf + clin['confidence']) / 2, 'MODERATE'
    elif ml_conf > 0.70 and clin['confidence'] < 0.45:
        final_diag, final_conf, consensus = ml_diag, ml_conf, 'ML_DOMINANT'
    elif clin['confidence'] >= 0.60:
        final_diag, final_conf, consensus = clin['diagnosis'], clin['confidence'], 'CLINICAL_DOMINANT'
    else:
        final_diag, final_conf, consensus = ml_diag, max(ml_conf, clin['confidence']), 'UNCERTAIN'

    # ML Interpretation
    if agree:
        ml_interpretation = '✅ ML aur clinical rules dono agree — high reliability'
    elif ml_conf < 0.50:
        ml_interpretation = 'ℹ️  ML uncertain — clinical rules pe zyada bharosa karo'
    elif not agree and clin['confidence'] >= 0.60:
        ml_interpretation = '⚠️  ML disagree kar raha — clinical evidence strong, extra tests recommend'
    else:
        ml_interpretation = '🚨 Strong disagreement — specialist review zaroori'

    # Flags
    flags = []
    if not agree:
        flags.append('⚠️ ML and clinical rules DISAGREE — expert review required')
    if ml_conf < 0.5:
        flags.append('⚠️ Low ML confidence — borderline case')
    if clin['confidence'] < 0.4:
        flags.append('⚠️ Weak clinical signal — atypical presentation')
    if gcs < 8:
        flags.append('🚨 CRITICAL: GCS < 8 — immediate ICU admission')
    if patient_dict.get('Procalcitonin', 0) > 2.0 and patient_dict.get('CSF_Lymphocytes_%', 0) > 70:
        flags.append('⚠️ Conflicting biomarkers: high PCT + lymphocytic dominance')
    if stage_override:
        flags.append(f'ℹ️ Stage determined by GCS ({gcs}) — ML stage overridden')

    # Recommendations
    recs = []
    if final_diag == 'Bacterial':
        recs += ['🚨 START empiric antibiotics NOW (vancomycin + ceftriaxone)',
                 'Take blood & CSF cultures before antibiotics if possible',
                 'Watch for septic shock, raised ICP']
    elif final_diag == 'Viral':
        recs += ['Supportive care + close neurological monitoring',
                 'Consider acyclovir if HSV cannot be excluded',
                 'Repeat LP if patient deteriorates']
    elif final_diag == 'Tuberculous':
        recs += ['Start 4-drug anti-TB: INH + Rifampin + PZA + Ethambutol',
                 'Add dexamethasone (reduces mortality)',
                 'Treatment duration: 9-12 months',
                 'Screen close contacts for TB']
    if final_stage == 'Stage III':
        recs.append('🚨 Stage III — ICU admission, aggressive monitoring')
    elif final_stage == 'Stage II':
        recs.append('Monitor closely — risk of rapid progression')
    if consensus == 'UNCERTAIN':
        recs.append('⚠️ UNCERTAIN — do NOT act on this alone, call specialist immediately')

    return {
        'final_diagnosis':   final_diag,
        'final_stage':       final_stage,
        'confidence':        round(final_conf, 3),
        'consensus_level':   consensus,
        'clinical_scores':   {k: round(v, 3) for k, v in clin['all_scores'].items()},
        'ml_interpretation': ml_interpretation,
        'agreement':         agree,
        'uncertainty_flags': flags,
        'recommendations':   recs
    }

# ── Routes ────────────────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'meningitis_model_final.pkl'}

@app.post('/predict')
def predict(patient: PatientInput):
    try:
        patient_data = patient.dict()

        # % wale fields rename karo
        patient_data['CSF_Neutrophils_%'] = patient_data.pop('CSF_Neutrophils_pct')
        patient_data['CSF_Lymphocytes_%'] = patient_data.pop('CSF_Lymphocytes_pct')

        result = predict_patient(patient_data, bundle)
        risk   = get_risk(result['final_stage'])

        return {
            'success':         True,
            'diagnosis':       result['final_diagnosis'],
            'stage':           result['final_stage'],
            'confidence':      result['confidence'],
            'consensus':       result['consensus_level'],
            'ml_vs_rules':     'AGREE' if result['agreement'] else 'DISAGREE',
            'interpretation':  result['ml_interpretation'],
            'clinical_scores': result['clinical_scores'],
            'flags':           result['uncertainty_flags'],
            'recommendations': result['recommendations'],
            'risk_level':      risk['risk_level'],
            'risk_color':      risk['risk_color'],
            'risk_heading':    risk['risk_heading'],
            'risk_message':    risk['risk_message'],
        }

    except Exception as e:
        return {'error': str(e)}

# ── Run ───────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)