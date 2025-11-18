import streamlit as st
import anthropic
import base64
import json
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="ECG Analysis System - Logical Sequential Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2563eb 0%, #4f46e5 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .logic-flow {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .decision-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .positive-finding {
        background-color: #fee2e2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .negative-finding {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .measurement-box {
        background-color: #e0e7ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6366f1;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

def encode_image(image_file):
    """Convert uploaded file to base64"""
    bytes_data = image_file.getvalue()
    return base64.b64encode(bytes_data).decode('utf-8')

def analyze_ecg_with_claude(image_file, api_key):
    """Analyze ECG using Claude API with strict logical sequence"""
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Determine file type and prepare content
    file_type = image_file.type
    base64_data = encode_image(image_file)
    
    analysis_prompt = """You are an expert cardiologist performing ECG analysis using STRICT LOGICAL SEQUENCE.

CRITICAL: ALL measurements must be based on STANDARD ECG GRID:
- Small square (horizontal): 0.04 seconds (40 milliseconds)
- Small square (vertical): 1 mm = 0.1 mV
- Large square (horizontal): 0.20 seconds (5 small squares)
- Large square (vertical): 5 mm = 0.5 mV
- Standard calibration: 10 mm = 1 mV
- Paper speed: 25 mm/s (unless otherwise marked)

Count the squares on the ECG grid for ALL measurements. Be precise.

═══════════════════════════════════════════════════════════════════════════════
LOGICAL SEQUENCE OF ANALYSIS - Follow this EXACT order with decision trees:
═══════════════════════════════════════════════════════════════════════════════

STEP 1: CALCULATE HEART RATE (First Priority)
═══════════════════════════════════════════════════════════════════════════════
Method 1 (Regular rhythm): 300 ÷ (number of large squares between R-R intervals)
Method 2: 1500 ÷ (number of small squares between R-R intervals)
Method 3: Count QRS complexes in 6 seconds × 10

→ COUNT the squares on the grid between consecutive R waves
→ MEASURE: R-R interval in seconds
→ CALCULATE: Heart rate in bpm
→ CLASSIFY: 
   • Bradycardia (<60 bpm)
   • Normal (60-100 bpm)
   • Tachycardia (>100 bpm)

STEP 2: EXAMINE P WAVES (Foundation of Rhythm Diagnosis)
═══════════════════════════════════════════════════════════════════════════════
Look at EVERY lead systematically. Count squares to measure.

A. P WAVE PRESENCE:
   → Are P waves visible? (Check ALL 12 leads)
   → If NO P waves visible:
      ├─ Look for f waves (atrial fibrillation): irregular baseline, 300-600/min
      ├─ Look for F waves (atrial flutter): sawtooth, 250-350/min
      ├─ Look for P waves hidden in QRS or T waves
      └─ Consider: Junctional rhythm, ventricular rhythm
   
   → If P waves present: PROCEED to detailed analysis

B. P WAVE MEASUREMENTS (Count squares on grid):
   → Duration: Normal <0.12s (<3 small squares)
   → Amplitude: Normal <0.25 mV (<2.5 small squares vertically)
   → COUNT: How many small squares wide and tall

C. P WAVE MORPHOLOGY BY LEAD:
   Lead II (Reference standard):
      → Upright and smooth = Normal sinus
      → Notched/biphasic = Consider left atrial enlargement
      → Tall peaked (>2.5mm) = Right atrial enlargement
   
   Lead V1 (Critical for atrial assessment):
      → Positive deflection first = Right atrium
      → Negative deflection second = Left atrium
      → Deeply negative terminal portion (>1mm deep, >0.04s wide) = LAE
      → Tall positive = RAE
   
   Leads I, aVL, V5-V6:
      → Should be upright in normal sinus rhythm
   
   Lead aVR:
      → Should be negative in normal sinus rhythm
      → Positive P in aVR = Ectopic rhythm or lead reversal

D. P WAVE AXIS:
   → Normal sinus: 0° to +75° (positive in II, III, aVF)
   → Left atrial rhythm: Negative in I, positive in aVR
   → Low atrial rhythm: Negative in II, III, aVF

E. P WAVE RATE (Count P-P interval):
   → <60/min = Sinus bradycardia (if sinus morphology)
   → 60-100/min = Normal sinus (if sinus morphology)
   → 100-150/min = Sinus tachycardia
   → 150-250/min = Atrial tachycardia
   → 250-350/min = Atrial flutter
   → 350-600/min = Atrial fibrillation

═══════════════════════════════════════════════════════════════════════════════
DECISION POINT 1: Based on P wave analysis, determine ATRIAL ACTIVITY:
═══════════════════════════════════════════════════════════════════════════════
→ Normal sinus P waves? → PROCEED to Step 3
→ Ectopic P waves? → Diagnose atrial rhythm, PROCEED to Step 3
→ Absent P waves? → Diagnose atrial fibrillation OR junctional rhythm
→ Flutter waves? → Diagnose atrial flutter
→ Then PROCEED to analyze ventricular response

STEP 3: P-QRS RELATIONSHIP (Critical for Rhythm & Conduction)
═══════════════════════════════════════════════════════════════════════════════
A. COUNT P waves and QRS complexes:
   → How many P waves per QRS?
      • 1:1 = Normal conduction (if PR constant) OR AV nodal reentry
      • 2:1, 3:1, 4:1 = AV block (if more P than QRS)
      • Variable = Mobitz II or high-degree AV block
      • More QRS than P = Junctional or ventricular rhythm

B. MEASURE PR INTERVAL (Count squares from start of P to start of QRS):
   → Normal: 0.12-0.20s (3-5 small squares)
   → Short (<0.12s or <3 squares):
      ├─ With delta wave = WPW syndrome
      └─ Without delta wave = Enhanced AV conduction or junctional
   
   → Prolonged (>0.20s or >5 squares):
      └─ 1st degree AV block
   
   → Variable PR interval:
      ├─ Progressively lengthening then dropped QRS = Mobitz I (Wenckebach)
      ├─ Suddenly dropped QRS with constant PR = Mobitz II
      └─ Completely variable = 3rd degree (complete) heart block

C. PR SEGMENT:
   → Depression in inferior leads + elevation in aVR = Acute pericarditis
   → Elevation may indicate atrial infarction

═══════════════════════════════════════════════════════════════════════════════
DECISION POINT 2: Based on P-QRS relationship, determine CONDUCTION:
═══════════════════════════════════════════════════════════════════════════════
→ 1:1 with normal PR → Normal AV conduction, PROCEED to QRS analysis
→ Prolonged PR → 1st degree AV block, PROCEED to QRS analysis
→ Progressive PR lengthening → Mobitz I, PROCEED with caution
→ Dropped QRS with constant PR → Mobitz II, HIGH RISK
→ Dissociated P and QRS → Complete heart block, EMERGENT

STEP 4: QRS COMPLEX ANALYSIS (Ventricular Depolarization)
═══════════════════════════════════════════════════════════════════════════════
A. MEASURE QRS DURATION (Count squares from start to end of QRS):
   → Normal: 0.06-0.10s (1.5-2.5 small squares)
   → Borderline: 0.10-0.12s (2.5-3 small squares)
   → Wide: ≥0.12s (≥3 small squares)

═══════════════════════════════════════════════════════════════════════════════
DECISION POINT 3: If QRS is WIDE (≥0.12s), determine CAUSE:
═══════════════════════════════════════════════════════════════════════════════

IF QRS ≥ 0.12s → Analyze MORPHOLOGY to differentiate:

→ Check V1 and V6 first:

PATTERN A - LBBB (Left Bundle Branch Block):
   V1: Predominantly NEGATIVE (deep S wave, may have small r)
   V6: Predominantly POSITIVE (broad monophasic R, no Q wave)
   Lead I: Broad notched R wave
   
   Count squares for QRS duration:
   → Standard LBBB: QRS ≥120ms (≥3 small squares)
   → Strauss LBBB Criteria (more specific for "true" LBBB):
      • QRS ≥140ms in men (≥3.5 small squares)
      • QRS ≥130ms in women (≥3.25 small squares)
      • Mid-QRS notching/slurring in ≥2 of: I, aVL, V5, V6
      • No Q waves in I, V5, V6
   
   → R-peak time in V6: Measure from QRS start to R peak
      • >60ms (>1.5 small squares) suggests poor LV function
   
   IF LBBB PRESENT:
      ├─ Cannot diagnose LVH reliably on voltage criteria
      ├─ But can use: QRS >160ms, Modified Sokolow-Lyon ≥45mm, R in aVL ≥11mm
      ├─ ST-T changes expected (appropriate discordance)
      ├─ New LBBB + chest pain = STEMI equivalent
      └─ LBBB + EF≤35% + QRS≥150ms = Consider CRT

PATTERN B - RBBB (Right Bundle Branch Block):
   V1: rsR' or rSR' pattern ("M-shaped", terminal R wave)
   V6: Wide terminal S wave
   Lead I: Wide terminal S wave
   
   Count squares:
   → QRS ≥120ms (≥3 small squares)
   → Measure terminal R' in V1
   
   Appropriate discordance:
   → ST depression and T inversion in V1-V3 (expected, not ischemic)
   
   IF RBBB PRESENT:
      ├─ CAN diagnose RVH (unlike LBBB where LVH diagnosis difficult)
      ├─ Look for: RAD >110°, very tall R' in V1 (>15mm), R/S in V1 >2.5
      ├─ Check for fascicular blocks (RBBB + LAD = bifascicular block)
      └─ Acute RBBB may indicate septal MI

PATTERN C - Ventricular Rhythm:
   V1: Concordance (all positive or all negative across precordium)
   Wide, bizarre QRS morphology
   → AV dissociation (P waves independent of QRS)
   → Capture beats or fusion beats
   → If rate >100: Ventricular tachycardia (EMERGENT)

PATTERN D - WPW (Pre-excitation):
   → Short PR (<0.12s)
   → Delta wave (slurred upstroke of QRS)
   → Wide QRS
   → Secondary ST-T changes

PATTERN E - Paced Rhythm:
   → Pacing spikes before QRS
   → Wide QRS (unless biventricular pacing)

═══════════════════════════════════════════════════════════════════════════════
IF QRS is NORMAL WIDTH (0.06-0.10s):
═══════════════════════════════════════════════════════════════════════════════

B. Q WAVE ANALYSIS (Lead by lead):
   Pathologic Q waves indicate myocardial infarction:
   → Width: ≥0.04s (≥1 small square) OR
   → Depth: ≥25% of R wave height (≥1/4 of R wave)
   
   Small "septal" Q waves are normal in:
   → I, aVL, V5, V6 (from septal depolarization)
   
   Location of pathologic Q waves:
   → II, III, aVF = Inferior MI (look for reciprocal changes in I, aVL)
   → V1-V3 = Anterior MI
   → V4-V6, I, aVL = Lateral MI
   → V1-V2 with tall R/S >1 = Posterior MI (reciprocal of Q)

C. R WAVE PROGRESSION (V1 to V6):
   Normal pattern (measure R wave amplitude in mm):
   → V1: Small r (1-3mm)
   → V2: R wave starts increasing
   → V3-V4: R wave = S wave (transition zone)
   → V5-V6: Tall R wave, small s wave
   
   Abnormal patterns:
   → Poor R progression: R wave in V3 ≤3mm
      Causes: Anterior MI, LBBB, LVH, COPD, lead misplacement
   → Early transition: Tall R in V1-V2
      Causes: Posterior MI, RVH, WPW, normal variant
   → Late transition: Persistent S wave into V5-V6
      Causes: LVH, normal variant

D. VOLTAGE CRITERIA FOR HYPERTROPHY (Measure in mm on grid):

LEFT VENTRICULAR HYPERTROPHY (LVH):
   Sokolow-Lyon Criteria:
   → S wave in V1 + R wave in V5 or V6 ≥35mm
      (Count: depth of S in V1 + height of R in V5/V6)
   
   Cornell Criteria:
   → Men: S in V3 + R in aVL >28mm
   → Women: S in V3 + R in aVL >20mm
   
   Additional LVH criteria:
   → R wave in aVL ≥11mm
   → R wave in lead I ≥14mm
   
   Supportive features:
   → Left axis deviation
   → R wave peak time in V5-V6 >50ms
   → LV strain pattern: ST depression + T inversion in I, aVL, V5-V6
   
   CRITICAL: Cannot diagnose LVH in presence of LBBB by voltage
   But in LBBB, suggestive findings:
   → QRS >160ms
   → Modified Sokolow-Lyon: S in V2 + R in V6 ≥45mm
   → R in aVL ≥11mm

RIGHT VENTRICULAR HYPERTROPHY (RVH):
   → Right axis deviation (>+110°)
   → Tall R wave in V1 (≥7mm OR R/S ratio ≥1)
   → Deep S wave in V5-V6
   → qR pattern in V1
   → RV strain: ST depression + T inversion in V1-V3
   
   Causes: Pulmonary hypertension, COPD, congenital heart disease
   
   RVH WITH RBBB (Can be diagnosed):
   → RAD >110°
   → Very tall R' in V1 (>15mm)
   → R/S ratio in V1 >2.5
   → ST-T strain more than expected for RBBB alone

E. QRS AXIS DETERMINATION (in frontal plane):
   
   Use leads I and aVF (easiest method):
   → Lead I positive + aVF positive = Normal axis (0° to +90°)
   → Lead I positive + aVF negative = Left axis deviation (-30° to -90°)
   → Lead I negative + aVF positive = Right axis deviation (+90° to +180°)
   → Lead I negative + aVF negative = Extreme axis (-90° to -180°)
   
   More precise: Find isoelectric lead (equal positive and negative)
   → Axis is perpendicular to this lead
   
   LEFT AXIS DEVIATION (-30° to -90°) causes:
   → Left anterior fascicular block (LAFB)
   → LVH
   → Inferior MI
   → WPW
   → Normal variant (especially with age)
   
   RIGHT AXIS DEVIATION (+90° to +180°) causes:
   → RVH
   → Left posterior fascicular block (LPFB)
   → Lateral MI
   → RBBB
   → Normal in young/tall individuals
   → Dextrocardia

STEP 5: ST SEGMENT ANALYSIS (Ischemia & Injury)
═══════════════════════════════════════════════════════════════════════════════
Reference: Use TP segment as baseline (NOT PR segment)

Measure from J point (junction of QRS and ST segment):

A. ST ELEVATION (Count mm above baseline):
   → ≥1mm (1 small square) in ≥2 contiguous leads = Significant
   → ≥2mm in V2-V3 in men, ≥1.5mm in women = Significant
   
   Morphology:
   → Concave (normal variant, early repolarization, pericarditis)
   → Convex or horizontal (acute MI, more concerning)
   
   Distribution determines territory:
   → II, III, aVF = Inferior STEMI (RCA or LCx)
      Check: Reciprocal ST depression in I, aVL
      Do: Right-sided leads (V4R) for RV involvement
   
   → V1-V4 = Anterior STEMI (LAD)
      Check: Reciprocal ST depression in inferior leads
   
   → I, aVL, V5-V6 = Lateral STEMI (LCx or diagonal)
   
   → V1-V6, I, aVL = Anterolateral STEMI (proximal LAD)
   
   → Tall R in V1-V2 + ST depression V1-V3 = Posterior MI
      Do: Posterior leads (V7-V9)
   
   → Diffuse ST elevation + PR depression = Acute pericarditis
   
   → ST elevation in aVR > other leads = Left main or severe 3-vessel disease

B. ST DEPRESSION (Count mm below baseline):
   → ≥0.5mm (0.5 small square) horizontal or downsloping = Significant
   → ≥1mm if upsloping (measured 60-80ms after J point)
   
   Types:
   → Horizontal or downsloping = Ischemia
   → Upsloping with tachycardia = May be normal (atrial repolarization)
   
   Distribution:
   → Diffuse = Subendocardial ischemia
   → Reciprocal to ST elevation = Helps confirm STEMI
   → V1-V3 with tall R = Posterior MI

C. J POINT:
   → J point elevation with rapid upsloping ST = Early repolarization (benign)
   → J point depression with upsloping ST in sinus tachycardia = Normal

STEP 6: T WAVE ANALYSIS (Repolarization)
═══════════════════════════════════════════════════════════════════════════════
A. T WAVE DIRECTION (Should be concordant with QRS):
   → Normally upright in: I, II, V3-V6
   → Normally inverted in: aVR
   → Variable in: III, aVL, aVF, V1-V2
   
   Concordance rule:
   → If QRS predominantly positive, T should be upright
   → If QRS predominantly negative, T should be inverted

B. T WAVE INVERSION (Abnormal if discordant):
   Deep symmetric T inversion:
   → V2-V4 = Wellens syndrome (critical LAD stenosis)
   → Inferior leads = Inferior ischemia/evolving MI
   → I, aVL, V5-V6 = Lateral ischemia
   
   → With appropriate discordance in BBB = Expected (not ischemic)

C. T WAVE MORPHOLOGY:
   → Peaked T waves: Hyperkalemia (>6.5 mEq/L), hyperacute MI
   → Flattened T waves: Ischemia, hypokalemia, digitalis
   → Biphasic T waves: Ischemia
   → Asymmetric T waves: Usually normal
   → Symmetric deeply inverted T: Ischemia, evolving MI, CNS event

D. T WAVE AMPLITUDE:
   → Normal: <10mm in precordial leads, <5mm in limb leads
   → Tall: Hyperkalemia, early MI, LVH, normal variant

STEP 7: QT INTERVAL (Repolarization Time)
═══════════════════════════════════════════════════════════════════════════════
Measure from beginning of QRS to end of T wave:

Count squares: QT interval in seconds

Correct for heart rate (QTc):
Bazett's formula: QTc = QT / √(RR interval)

Normal values:
→ Men: QTc ≤440ms (≤11 small squares at 60 bpm)
→ Women: QTc ≤460ms (≤11.5 small squares at 60 bpm)

PROLONGED QT (>500ms = high risk):
Causes:
→ Congenital long QT syndrome
→ Drugs: Antiarrhythmics, antipsychotics, antibiotics (macrolides, fluoroquinolones)
→ Electrolytes: Hypokalemia, hypomagnesemia, hypocalcemia
→ Bradycardia
→ Acute MI

Risk: Torsades de pointes → Ventricular fibrillation

SHORT QT (<340ms):
→ Hypercalcemia
→ Digitalis effect
→ Congenital short QT syndrome

NOTE: If QRS is wide (BBB), QT is artificially prolonged
→ Use JT interval (QT minus QRS) or
→ QTc - QRS = JTc

STEP 8: U WAVE (If Present)
═══════════════════════════════════════════════════════════════════════════════
→ Small positive deflection after T wave
→ Best seen in V2-V4
→ Normal amplitude: <2mm

Prominent U waves:
→ Hypokalemia (<3 mEq/L)
→ Bradycardia
→ Drugs: Digitalis, class IA or III antiarrhythmics

Inverted U waves:
→ Ischemia
→ LVH

═══════════════════════════════════════════════════════════════════════════════
STEP 9: CHAMBER ENLARGEMENT IN ATRIAL FIBRILLATION
═══════════════════════════════════════════════════════════════════════════════
When P waves absent due to AF, use f-wave analysis:

LEFT ATRIAL ENLARGEMENT (LAE):
→ Dominantly negative f-waves in V1 (>1mm deep, >40ms wide)
→ Broad, notched fibrillatory waves
→ Leftward atrial vector: negative f-waves in V1, broad f-waves in II

RIGHT ATRIAL ENLARGEMENT (RAE):
→ Tall positive f-waves in V1
→ Large f-waves in II, III, aVF
→ Rightward vector dominance

═══════════════════════════════════════════════════════════════════════════════
FINAL SYNTHESIS - OVERALL INTERPRETATION:
═══════════════════════════════════════════════════════════════════════════════

Return comprehensive analysis in this JSON structure:

{
  "gridMeasurements": {
    "paperSpeed": "25 mm/s or other",
    "calibration": "10mm = 1mV or other",
    "measurementNote": "All measurements counted on standard ECG grid"
  },
  
  "step1_heartRate": {
    "rrIntervalSmallSquares": number,
    "rrIntervalSeconds": number,
    "heartRate": number,
    "calculationMethod": "method used",
    "classification": "bradycardia/normal/tachycardia",
    "regularity": "regular/regularly irregular/irregularly irregular"
  },
  
  "step2_pWaveAnalysis": {
    "pWavesPresent": true/false,
    "ifAbsent": {
      "fWaves": true/false,
      "flutterWaves": true/false,
      "diagnosis": "atrial fibrillation/atrial flutter/junctional/ventricular"
    },
    "ifPresent": {
      "durationSmallSquares": number,
      "durationSeconds": number,
      "amplitudeSmallSquares": number,
      "amplitudeMv": number,
      "morphologyByLead": {
        "leadII": "description with normal/abnormal",
        "leadV1": "description",
        "leadI_aVL": "description",
        "lea
