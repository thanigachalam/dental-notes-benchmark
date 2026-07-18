
import json, random, re, statistics, zipfile
from pathlib import Path
from collections import Counter

random.seed(20260313)

OUT = Path("/mnt/data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl")
STATS = Path("/mnt/data/dentalnote_bench_20000_icd10_internalproc_v20_stats.json")
SCRIPT_OUT = Path("/mnt/data/generate_dentalnote_bench_20000_icd10_internalproc_v20.py")
ZIP_OUT = Path("/mnt/data/dentalnote_bench_20000_icd10_internalproc_v20_bundle.zip")
ICD_CSV = Path("/mnt/data/icd10_cm_diagnosis_mappings_v20.csv")
PROC_CSV = Path("/mnt/data/internal_procedure_code_definitions_v20.csv")
ICD_MD = Path("/mnt/data/icd10_cm_diagnosis_mappings_v20.md")
PROC_MD = Path("/mnt/data/internal_procedure_code_definitions_v20.md")

DIAG_SYSTEM = "ICD-10-CM"
PROC_SYSTEM = "DPO_INTERNAL_V1"

PROCEDURE_CODEBOOK = {
    "PROC_LOCAL_ANES": ("Local anesthesia", "anesthesia", "Local infiltration or block anesthesia for dental treatment"),
    "PROC_COMPOSITE": ("Composite restoration", "restorative", "Direct composite restoration for carious or defective tooth structure"),
    "PROC_BWXR": ("Bitewing radiograph", "diagnostic", "Bitewing imaging for proximal caries/restoration assessment"),
    "PROC_OCCLUSAL_ADJ": ("Occlusal adjustment", "restorative", "Minor adjustment of occlusion after restoration"),
    "PROC_PERIO_MAINT": ("Periodontal maintenance", "periodontics", "Periodontal maintenance/debridement visit"),
    "PROC_RAD_REVIEW": ("Radiographic review", "diagnostic", "Review of available radiographs"),
    "PROC_OHI": ("Oral hygiene instruction", "preventive", "Counseling on brushing, flossing, and interdental cleaning"),
    "PROC_PA_XR": ("Periapical radiograph", "diagnostic", "Periapical imaging of a specific tooth"),
    "PROC_PULP_TEST": ("Pulp vitality testing", "endodontics", "Thermal/electric pulp testing"),
    "PROC_PULPOTOMY": ("Emergency pulpotomy", "endodontics", "Pulpotomy for urgent pain control"),
    "PROC_TEMP_REST": ("Temporary restoration", "endodontics", "Temporary sealing/restoration"),
    "PROC_TRAUMA_EVAL": ("Dental trauma evaluation", "emergency", "Focused evaluation of traumatic dental injury"),
    "PROC_PERIAPICAL_XR": ("Periapical radiograph", "diagnostic", "Periapical imaging for trauma or endodontic review"),
    "PROC_REPOSITION": ("Repositioning of injured tooth", "trauma", "Repositioning of displaced traumatized tooth"),
    "PROC_FLEX_SPLINT": ("Flexible splint", "trauma", "Flexible stabilization splint for traumatized teeth"),
    "PROC_PROPHY": ("Dental prophylaxis", "preventive", "Routine prophylaxis/cleaning"),
    "PROC_FLUORIDE": ("Topical fluoride varnish", "preventive", "Topical fluoride application"),
    "PROC_SEALANT": ("Sealant placement", "preventive", "Placement of pit and fissure sealant"),
    "PROC_IMPLANT_CHECK": ("Implant follow-up evaluation", "implant", "Follow-up assessment of implant site"),
    "PROC_OCCLUSION_CHECK": ("Occlusion check", "prosthodontics", "Assessment of implant or prosthetic occlusion"),
    "PROC_EXT_SIMPLE": ("Simple extraction", "oral_surgery", "Extraction of erupted tooth"),
    "PROC_EXT_SURGICAL": ("Surgical extraction", "oral_surgery", "Surgical removal of tooth"),
    "PROC_POSTOP": ("Post-operative instructions", "oral_surgery", "Post-operative counseling and precautions"),
    "PROC_CROWN_PREP": ("Crown preparation", "prosthodontics", "Tooth preparation for full-coverage restoration"),
    "PROC_IMPRESSION": ("Impression or digital scan", "prosthodontics", "Impression or digital scan for prosthetic fabrication"),
    "PROC_TEMP_CROWN": ("Provisional crown", "prosthodontics", "Placement or adjustment of provisional crown"),
    "PROC_CEMENT_RESEAT": ("Recementation of existing crown", "prosthodontics", "Recementation or reseat of crown"),
    "PROC_EMERG_EVAL": ("Emergency evaluation", "emergency", "Focused evaluation for urgent complaint"),
    "PROC_I_AND_D": ("Incision and drainage", "emergency", "Incision and drainage of odontogenic swelling"),
    "PROC_PRESCRIBE": ("Prescription management", "emergency", "Medication prescribing/review for urgent care"),
    "PROC_EXAM": ("Periodic oral evaluation", "diagnostic", "Recall/periodic oral examination"),
}

def d(code, official, tag, summary=None):
    return {
        "code": code,
        "official_text": official,
        "tag": tag,
        "summary_text": summary or official.lower(),
    }

SCENARIOS = [
    {
        "id": "REST",
        "prompt_id": "P_RESTORATIVE_CARIES",
        "visit_type": "restorative",
        "specialty": "restorative_dentistry",
        "age_groups": ["adult", "older_adult", "adolescent"],
        "sexes": ["female", "male"],
        "emergency_flag": False,
        "chief_complaints": [
            "cold sensitivity on an upper molar with sweets",
            "food trapping and sensitivity around a back tooth",
            "pain when chewing on a lower molar for the past 10 days",
            "broken filling noticed while flossing",
            "lingering sensitivity around an old filling",
            "rough area catching floss on a back tooth"
        ],
        "teeth": ["3","4","5","12","13","14","19","20","29","30"],
        "surfaces": [["M","O"],["D","O"],["O"],["M","O","D"],["B"],["root"]],
        "diagnosis_options": [
            [d("K02.52","Dental caries on pit and fissure surface penetrating into dentin","pit_dentin","pit and fissure caries into dentin")],
            [d("K02.51","Dental caries on pit and fissure surface limited to enamel","pit_enamel","early pit and fissure caries limited to enamel")],
            [d("K02.62","Dental caries on smooth surface penetrating into dentin","smooth_dentin","smooth surface caries into dentin")],
            [d("K02.61","Dental caries on smooth surface limited to enamel","smooth_enamel","smooth surface caries limited to enamel")],
            [d("K02.7","Dental root caries","root_caries","root caries")],
            [d("K08.51","Open restoration margins of tooth","open_margin","open restoration margin")],
            [d("K08.531","Fractured dental restorative material with loss of material","fractured_restoration","fractured restoration with loss of material")],
            [d("K03.81","Cracked tooth","cracked_tooth","cracked tooth")],
            [d("K02.52","Dental caries on pit and fissure surface penetrating into dentin","pit_dentin","pit and fissure caries into dentin"),
             d("K08.51","Open restoration margins of tooth","open_margin","open restoration margin")],
        ],
        "procedure_templates": [
            ("PROC_LOCAL_ANES", "Local anesthesia"),
            ("PROC_COMPOSITE", "Composite restoration"),
            ("PROC_BWXR", "Bitewing radiograph"),
            ("PROC_OCCLUSAL_ADJ", "Occlusal adjustment")
        ],
        "materials": ["composite_resin", "etch_bonding_agent"],
        "risk_pool": ["high_caries_risk", "fair_oral_hygiene", "frequent_snacking", "xerostomia", "history_of_large_restorations"],
        "sdoh_pool": ["transportation_barrier", "none", "financial_constraints"],
        "followups": ["6_month_recall", "watch_for_postop_sensitivity", "routine_hygiene_recall", "crown_eval_if_crack_persists"],
    },
    {
        "id": "PERIO",
        "prompt_id": "P_PERIO_MAINT",
        "visit_type": "perio_maintenance",
        "specialty": "periodontics",
        "age_groups": ["adult", "older_adult"],
        "sexes": ["female", "male"],
        "emergency_flag": False,
        "chief_complaints": [
            "bleeding while brushing",
            "maintenance visit with intermittent sensitivity",
            "gums feel sore and puffy in the back",
            "follow-up for periodontal maintenance",
            "food packing between back teeth with gum bleeding",
            "ongoing concern about deeper pockets"
        ],
        "teeth": ["3","14","19","30"],
        "surfaces": [["B","L"],["M"],["D"],["root"]],
        "diagnosis_options": [
            [d("K05.00","Acute gingivitis, plaque induced","acute_gingivitis","acute plaque-induced gingivitis")],
            [d("K05.10","Chronic gingivitis, plaque induced","chronic_gingivitis","chronic plaque-induced gingivitis")],
            [d("K05.312","Chronic periodontitis, localized, moderate","perio_loc_mod","localized moderate chronic periodontitis")],
            [d("K05.313","Chronic periodontitis, localized, severe","perio_loc_sev","localized severe chronic periodontitis")],
            [d("K05.322","Chronic periodontitis, generalized, moderate","perio_gen_mod","generalized moderate chronic periodontitis")],
            [d("K05.323","Chronic periodontitis, generalized, severe","perio_gen_sev","generalized severe chronic periodontitis")],
            [d("K06.0","Gingival recession","gingival_recession","gingival recession")],
            [d("K05.5","Other periodontal diseases","other_perio","other periodontal disease")],
            [d("K05.322","Chronic periodontitis, generalized, moderate","perio_gen_mod","generalized moderate chronic periodontitis"),
             d("K06.0","Gingival recession","gingival_recession","gingival recession")],
        ],
        "procedure_templates": [
            ("PROC_PERIO_MAINT", "Periodontal maintenance"),
            ("PROC_RAD_REVIEW", "Radiographic review"),
            ("PROC_OHI", "Oral hygiene instruction")
        ],
        "materials": ["ultrasonic_scaler", "hand_curettes", "topical_fluoride"],
        "risk_pool": ["diabetes_type_2", "smoker", "fair_oral_hygiene", "plaque_accumulation", "history_of_periodontal_treatment"],
        "sdoh_pool": ["limited_dental_access_history", "none", "financial_constraints"],
        "followups": ["3_month_recall", "periodontal_recheck_6_weeks", "reinforce_interdental_cleaning", "periodontal_maintenance_3_months"],
    },
    {
        "id": "ENDO",
        "prompt_id": "P_ENDO_CONSULT",
        "visit_type": "endo_consult",
        "specialty": "endodontics",
        "age_groups": ["adult", "older_adult", "adolescent"],
        "sexes": ["female", "male"],
        "emergency_flag": False,
        "chief_complaints": [
            "severe lingering pain to cold on a lower molar",
            "spontaneous throbbing pain in a back tooth",
            "pain waking the patient at night",
            "biting pain and sensitivity to temperature",
            "deep toothache not controlled by over-the-counter medication",
            "sharp pain with chewing and cold exposure"
        ],
        "teeth": ["3","14","19","30"],
        "surfaces": [["O"],["M","O"],["D","O"]],
        "diagnosis_options": [
            [d("K04.01","Reversible pulpitis","rev_pulp","reversible pulpitis")],
            [d("K04.02","Irreversible pulpitis","irr_pulp","irreversible pulpitis"),
             d("K04.4","Acute apical periodontitis of pulpal origin","acute_apical","acute apical periodontitis of pulpal origin")],
            [d("K04.1","Necrosis of the pulp","necrotic_pulp","pulp necrosis"),
             d("K04.7","Periapical abscess without sinus","abscess_nosinus","periapical abscess without sinus")],
            [d("K04.5","Chronic apical periodontitis","chronic_apical","chronic apical periodontitis")],
            [d("K04.6","Periapical abscess with sinus","abscess_sinus","periapical abscess with sinus")],
            [d("K04.8","Radicular cyst","radicular_cyst","radicular cyst")],
        ],
        "procedure_templates": [
            ("PROC_PA_XR", "Periapical radiograph"),
            ("PROC_PULP_TEST", "Pulp vitality testing"),
            ("PROC_PULPOTOMY", "Emergency pulpotomy"),
            ("PROC_TEMP_REST", "Temporary restoration")
        ],
        "materials": ["temporary_restorative_material", "calcium_hydroxide_liner"],
        "risk_pool": ["poor_sleep_due_to_pain", "delayed_dental_care", "high_caries_risk"],
        "sdoh_pool": ["transportation_barrier", "none", "financial_constraints"],
        "followups": ["urgent_endo_within_1_week", "root_canal_completion_2_weeks", "monitor_if_symptoms_resolve", "definitive_endo_referral"],
    },
    {
        "id": "PEDTRAUMA",
        "prompt_id": "P_PED_TRAUMA",
        "visit_type": "emergency",
        "specialty": "pediatric_dentistry",
        "age_groups": ["child", "adolescent"],
        "sexes": ["female", "male"],
        "emergency_flag": True,
        "chief_complaints": [
            "front tooth pushed back after a bicycle fall",
            "loose front tooth after playground injury",
            "broken front tooth after sports injury",
            "lip swelling and tooth pain after a fall",
            "parent reports tooth feels out of place after collision",
            "pain and mobility in a front tooth after recess injury"
        ],
        "teeth": ["8","9","7","10"],
        "surfaces": [["F"],["facial"]],
        "diagnosis_options": [
            [d("S03.2XXA","Dislocation of tooth: initial encounter","dislocation","tooth dislocation")],
            [d("S02.5XXA","Fracture of tooth (traumatic): initial encounter closed","tooth_fracture","traumatic tooth fracture")],
            [d("S03.2XXA","Dislocation of tooth: initial encounter","dislocation","tooth dislocation"),
             d("S01.512A","Laceration without foreign body of oral cavity","oral_laceration","oral cavity laceration")],
            [d("S02.5XXA","Fracture of tooth (traumatic): initial encounter closed","tooth_fracture","traumatic tooth fracture"),
             d("S09.93XA","Unspecified injury of face: initial encounter","facial_injury","facial injury")],
        ],
        "procedure_templates": [
            ("PROC_TRAUMA_EVAL", "Dental trauma evaluation"),
            ("PROC_PERIAPICAL_XR", "Periapical radiograph"),
            ("PROC_REPOSITION", "Repositioning of injured tooth"),
            ("PROC_FLEX_SPLINT", "Flexible splint"),
            ("PROC_LOCAL_ANES", "Local anesthesia")
        ],
        "materials": ["flexible_splint_material", "composite_resin"],
        "risk_pool": ["sports_injury_history"],
        "sdoh_pool": ["none"],
        "followups": ["2_week_and_4_6_week_trauma_followup", "urgent_trauma_recheck_1_week", "monitor_pulpal_status"],
    },
    {
        "id": "PEDRECALL",
        "prompt_id": "P_PED_RECALL",
        "visit_type": "pediatric_recall",
        "specialty": "pediatric_dentistry",
        "age_groups": ["child"],
        "sexes": ["female", "male"],
        "emergency_flag": False,
        "chief_complaints": [
            "routine cleaning and checkup",
            "parent concerned about new cavities",
            "occasional sensitivity on lower left when eating candy",
            "needs school dental form and recall exam",
            "parent reports difficulty with nighttime brushing",
            "wants fluoride and routine exam"
        ],
        "teeth": ["A","B","I","J","K","L","S","T","3","14","19","30"],
        "surfaces": [["O"],["M","O"],["D","O"],["B"]],
        "diagnosis_options": [
            [d("Z01.20","Encounter for routine dental examination","routine_exam","routine dental examination")],
            [d("Z01.21","Encounter for dental examination with abnormal findings","exam_abnormal","dental examination with abnormal findings"),
             d("K03.6","Deposits on teeth","deposits","deposits on teeth")],
            [d("K02.51","Dental caries on pit and fissure surface limited to enamel","pit_enamel","early pit and fissure caries")],
            [d("K02.52","Dental caries on pit and fissure surface penetrating into dentin","pit_dentin","pit and fissure caries into dentin")],
            [d("K02.61","Dental caries on smooth surface limited to enamel","smooth_enamel","smooth surface caries limited to enamel")],
            [d("K02.62","Dental caries on smooth surface penetrating into dentin","smooth_dentin","smooth surface caries into dentin")],
            [d("K00.6","Disturbances in tooth eruption","eruption_disturbance","disturbance in tooth eruption")],
        ],
        "procedure_templates": [
            ("PROC_PROPHY", "Child prophylaxis"),
            ("PROC_FLUORIDE", "Topical fluoride varnish"),
            ("PROC_BWXR", "Bitewing radiograph"),
            ("PROC_SEALANT", "Sealant placement")
        ],
        "materials": ["fluoride_varnish", "sealant_material"],
        "risk_pool": ["high_caries_risk", "frequent_snacking", "inconsistent_brushing"],
        "sdoh_pool": ["language_barrier", "none", "transportation_barrier"],
        "followups": ["6_month_recall", "sealant_followup", "restorative_visit_soon", "eruption_monitoring"],
    },
    {
        "id": "IMPLANT",
        "prompt_id": "P_IMPLANT_FU",
        "visit_type": "implant_followup",
        "specialty": "implant_dentistry",
        "age_groups": ["adult", "older_adult"],
        "sexes": ["female", "male"],
        "emergency_flag": False,
        "chief_complaints": [
            "implant healing check",
            "mild soreness around implant site",
            "follow-up after implant placement",
            "concern about pressure when chewing near implant",
            "wants confirmation that implant is healing normally",
            "implant site feels tight but not painful"
        ],
        "teeth": ["19","30","3","14"],
        "surfaces": [["implant_site"]],
        "diagnosis_options": [
            [d("K08.49","Partial loss of teeth due to other specified cause, unspecified class","missing_tooth_other","partial tooth loss status after prior extraction")],
            [d("K08.42","Partial loss of teeth due to periodontal diseases","partial_loss_perio","partial loss of teeth due to periodontal disease")],
            [d("K08.43","Partial loss of teeth due to caries","partial_loss_caries","partial loss of teeth due to caries")],
            [d("M27.61","Osseointegration failure of dental implant","implant_early_failure","osseointegration failure of implant")],
            [d("M27.62","Post-osseointegration biological failure of implant","implant_bio_failure","biological failure of implant")],
            [d("M27.63","Post-osseointegration mechanical failure of implant","implant_mech_failure","mechanical failure of implant")],
            [d("M27.69","Other endosseous dental implant failure","implant_other_failure","other implant failure")],
        ],
        "procedure_templates": [
            ("PROC_IMPLANT_CHECK", "Implant follow-up evaluation"),
            ("PROC_PA_XR", "Periapical radiograph"),
            ("PROC_OCCLUSION_CHECK", "Occlusion check")
        ],
        "materials": ["chlorhexidine_rinse", "implant_healing_abutment"],
        "risk_pool": ["history_of_periodontitis", "smoker", "none"],
        "sdoh_pool": ["none"],
        "followups": ["implant_followup_4_weeks", "restorative_phase_planning", "implant_hygiene_review", "implant_failure_workup"],
    },
    {
        "id": "SURG",
        "prompt_id": "P_ORALSURG_EXT",
        "visit_type": "oral_surgery",
        "specialty": "oral_surgery",
        "age_groups": ["adult", "older_adult", "adolescent"],
        "sexes": ["female", "male"],
        "emergency_flag": False,
        "chief_complaints": [
            "broken tooth with pain and swelling",
            "referred for extraction of non-restorable tooth",
            "intermittent pain around wisdom tooth",
            "painful fractured molar",
            "recurrent infection around lower third molar",
            "tooth deemed non-restorable by referring dentist"
        ],
        "teeth": ["1","16","17","32","19","30"],
        "surfaces": [["O"],["D"],["M","O"]],
        "diagnosis_options": [
            [d("K01.1","Impacted teeth","impacted","impacted tooth")],
            [d("K01.0","Embedded teeth","embedded","embedded tooth")],
            [d("K08.3","Retained dental root","retained_root","retained root")],
            [d("K03.81","Cracked tooth","cracked_tooth","cracked tooth")],
            [d("K04.7","Periapical abscess without sinus","abscess_nosinus","periapical abscess without sinus")],
            [d("K05.5","Other periodontal diseases","other_perio","other periodontal disease")],
        ],
        "procedure_templates": [
            ("PROC_EXT_SIMPLE", "Simple extraction"),
            ("PROC_EXT_SURGICAL", "Surgical extraction"),
            ("PROC_LOCAL_ANES", "Local anesthesia"),
            ("PROC_PA_XR", "Periapical radiograph"),
            ("PROC_POSTOP", "Post-operative instructions")
        ],
        "materials": ["gelfoam", "sutures", "saline_irrigation"],
        "risk_pool": ["smoker", "anxiety", "delayed_dental_care"],
        "sdoh_pool": ["financial_constraints", "none"],
        "followups": ["postop_check_1_week", "prn_for_pain_swelling", "surgical_site_recheck"],
    },
    {
        "id": "PROSTHO",
        "prompt_id": "P_PROSTHO_CROWN",
        "visit_type": "prosthodontic_restoration",
        "specialty": "prosthodontics",
        "age_groups": ["adult", "older_adult"],
        "sexes": ["female", "male"],
        "emergency_flag": False,
        "chief_complaints": [
            "crown came off while eating",
            "broken tooth needing full coverage",
            "old crown feels loose",
            "needs crown after root canal",
            "temporary crown broke over the weekend",
            "crown margin catching floss and food"
        ],
        "teeth": ["3","14","19","30","8","9"],
        "surfaces": [["full_coverage"]],
        "diagnosis_options": [
            [d("K08.50","Unsatisfactory restoration of tooth, unspecified","unsat_restoration","unsatisfactory restoration")],
            [d("K08.51","Open restoration margins of tooth","open_margin","open restoration margin")],
            [d("K08.531","Fractured dental restorative material with loss of material","fractured_restoration","fractured restoration with loss of material")],
            [d("K08.59","Other unsatisfactory restoration of existing tooth","other_unsat_restoration","other unsatisfactory restoration")],
            [d("K03.81","Cracked tooth","cracked_tooth","cracked tooth")],
            [d("K08.43","Partial loss of teeth due to caries","partial_loss_caries","partial loss of teeth due to caries")],
            [d("K08.42","Partial loss of teeth due to periodontal diseases","partial_loss_perio","partial loss of teeth due to periodontal disease")],
        ],
        "procedure_templates": [
            ("PROC_CROWN_PREP", "Crown preparation"),
            ("PROC_IMPRESSION", "Final impression or digital scan"),
            ("PROC_TEMP_CROWN", "Provisional crown"),
            ("PROC_CEMENT_RESEAT", "Recementation of existing crown")
        ],
        "materials": ["temporary_cement", "retraction_cord", "impression_material"],
        "risk_pool": ["bruxism", "history_of_large_restorations", "none"],
        "sdoh_pool": ["none"],
        "followups": ["crown_delivery_2_weeks", "watch_temp_crown", "return_if_crown_loose", "definitive_restoration_followup"],
    },
    {
        "id": "EMERG",
        "prompt_id": "P_EMERG_ABSCESS",
        "visit_type": "emergency",
        "specialty": "general_dentistry",
        "age_groups": ["adult", "older_adult"],
        "sexes": ["female", "male"],
        "emergency_flag": True,
        "chief_complaints": [
            "swelling and severe tooth pain since yesterday",
            "facial swelling with bad taste in mouth",
            "sharp pain with swelling around lower molar",
            "unable to sleep due to throbbing tooth pain",
            "worsening pressure and swelling around a back tooth",
            "painful tooth with swelling and bad taste for two days"
        ],
        "teeth": ["14","19","30","3"],
        "surfaces": [["B"],["O"],["D"]],
        "diagnosis_options": [
            [d("K04.4","Acute apical periodontitis of pulpal origin","acute_apical","acute apical periodontitis of pulpal origin")],
            [d("K04.6","Periapical abscess with sinus","abscess_sinus","periapical abscess with sinus")],
            [d("K04.7","Periapical abscess without sinus","abscess_nosinus","periapical abscess without sinus")],
            [d("K12.2","Cellulitis and abscess of mouth","mouth_abscess","cellulitis and abscess of mouth")],
            [d("L03.90","Cellulitis: unspecified","cellulitis_unspecified","cellulitis")],
            [d("K03.81","Cracked tooth","cracked_tooth","cracked tooth")],
            [d("K04.7","Periapical abscess without sinus","abscess_nosinus","periapical abscess without sinus"),
             d("K12.2","Cellulitis and abscess of mouth","mouth_abscess","cellulitis and abscess of mouth")],
        ],
        "procedure_templates": [
            ("PROC_EMERG_EVAL", "Emergency evaluation"),
            ("PROC_PA_XR", "Periapical radiograph"),
            ("PROC_I_AND_D", "Incision and drainage"),
            ("PROC_PRESCRIBE", "Prescription management")
        ],
        "materials": ["saline_irrigation", "drain_strip"],
        "risk_pool": ["delayed_dental_care", "smoker", "poor_access_to_care"],
        "sdoh_pool": ["financial_constraints", "transportation_barrier", "none"],
        "followups": ["urgent_endo_or_ext", "recheck_48_hours", "emergency_precautions", "same_day_return_if_swelling_worsens"],
    },
    {
        "id": "RECALL",
        "prompt_id": "P_HYGIENE_RECALL",
        "visit_type": "recall",
        "specialty": "general_dentistry",
        "age_groups": ["adult", "older_adult"],
        "sexes": ["female", "male"],
        "emergency_flag": False,
        "chief_complaints": [
            "routine cleaning and exam",
            "mild staining and occasional bleeding",
            "no major dental complaints today",
            "sensitivity near lower premolar when brushing",
            "would like fluoride and routine maintenance",
            "noticed light bleeding when flossing"
        ],
        "teeth": ["21","28","12","5"],
        "surfaces": [["B"],["C"],["root"]],
        "diagnosis_options": [
            [d("Z01.20","Encounter for routine dental examination","routine_exam","routine dental examination")],
            [d("Z01.21","Encounter for dental examination with abnormal findings","exam_abnormal","dental examination with abnormal findings"),
             d("K03.6","Deposits on teeth","deposits","deposits on teeth")],
            [d("K05.10","Chronic gingivitis, plaque induced","chronic_gingivitis","chronic plaque-induced gingivitis")],
            [d("K03.89","Other specified diseases of hard tissues of teeth","root_sensitivity","hard tissue sensitivity")],
            [d("K03.1","Abrasion of teeth","abrasion","tooth abrasion")],
            [d("G47.63","Bruxism, sleep related","bruxism","sleep-related bruxism")],
            [d("K03.81","Cracked tooth","cracked_tooth","cracked tooth")],
        ],
        "procedure_templates": [
            ("PROC_PROPHY", "Adult prophylaxis"),
            ("PROC_EXAM", "Periodic oral evaluation"),
            ("PROC_FLUORIDE", "Topical fluoride")
        ],
        "materials": ["prophy_paste", "fluoride_varnish"],
        "risk_pool": ["none", "mild_plaque_accumulation", "bruxism"],
        "sdoh_pool": ["none"],
        "followups": ["6_month_recall", "watch_sensitive_area", "nightguard_consideration"],
    },
]

SEX_WORD = {"female": "female", "male": "male", "other":"other"}
DURATIONS = ["2 days", "3 days", "1 week", "10 days", "2 weeks", "1 month", "several months"]
INJURY_MECHS = ["a bicycle fall", "a playground fall", "a sports injury", "an accidental collision at school", "a fall from a scooter"]
CARIES_RISKS = ["high", "moderate", "low"]
PLAQUE_STATES = ["mild", "moderate", "heavy"]
PERC_STATES = ["present", "absent", "mildly positive"]
ANESTHETICS = [
    "2% lidocaine with 1:100,000 epinephrine",
    "4% articaine with 1:100,000 epinephrine",
    "3% mepivacaine plain"
]
ALLERGY_POOL = [
    [],
    [{"name":"penicillin","reaction":"rash"}],
    [{"name":"latex","reaction":"itching"}],
    [{"name":"codeine","reaction":"nausea"}],
]
MEDICATION_POOL = [
    [],
    ["Ibuprofen"],
    ["Acetaminophen"],
    ["Metformin"],
    ["Lisinopril"],
    ["Amoxicillin"],
    ["Chlorhexidine rinse"],
    ["Ibuprofen", "Acetaminophen"],
]
NOTE_STYLE_POOL = ["classic_soap","abbrev_soap","mixed_narrative","structured_bullets","assessment_forward","ehr_shortform"]
SUMMARY_STYLE_POOL = ["compact_soap","narrative_clinical","billing_oriented","handoff_brief"]

def choose(xs):
    return random.choice(xs)

def sample_some(xs, max_k=2):
    k = random.randint(0, min(max_k, len(xs)))
    return random.sample(xs, k=k)

def random_demo(age, sex):
    if age == "child":
        return f"{random.randint(6,11)}-year-old {SEX_WORD[sex]}"
    if age == "adolescent":
        return f"{random.randint(12,17)}-year-old {SEX_WORD[sex]}"
    if age == "adult":
        return f"{random.randint(18,64)}-year-old {SEX_WORD[sex]}"
    return f"{random.randint(65,84)}-year-old {SEX_WORD[sex]}"

def normalize_surface_list(surfaces):
    out=[]
    for s in surfaces:
        out.append("facial" if s == "F" else s)
    return out

def surface_text(surfaces):
    return "/".join(normalize_surface_list(surfaces))

def has_tag(dx_list, tag):
    return any(dx["tag"] == tag for dx in dx_list)

def has_code(dx_list, code):
    return any(dx["code"] == code for dx in dx_list)

def findings_for_scenario(scenario, dx_list, tooth, surfaces):
    visit = scenario["visit_type"]
    surface_str = surface_text(surfaces)
    if scenario["id"] == "REST":
        if has_tag(dx_list, "open_margin"):
            return choose([
                f"open restorative margin with plaque retention on the {surface_str} surfaces",
                f"marginal breakdown and recurrent staining on the {surface_str} aspect",
            ])
        if has_tag(dx_list, "fractured_restoration"):
            return choose([
                f"fractured restoration with loss of material on the {surface_str} surfaces",
                f"restorative fracture and food trap involving the {surface_str} surfaces",
            ])
        if has_tag(dx_list, "cracked_tooth"):
            return choose([
                "crack line with biting tenderness and no obvious cusp separation",
                "crack-related pain on release with localized cusp tenderness",
            ])
        if has_tag(dx_list, "root_caries"):
            return choose([
                "soft root-surface caries with cervical plaque retention",
                "cervical/root caries with localized root sensitivity",
            ])
        if has_tag(dx_list, "smooth_enamel"):
            return choose([
                f"non-cavitated smooth-surface caries on the {surface_str} surfaces",
                f"early enamel involvement on the {surface_str} aspect",
            ])
        if has_tag(dx_list, "smooth_dentin"):
            return choose([
                f"smooth-surface caries extending into dentin on the {surface_str} surfaces",
                f"cavitated smooth-surface lesion with dentinal involvement on the {surface_str} aspect",
            ])
        if has_tag(dx_list, "pit_enamel"):
            return choose([
                f"incipient pit and fissure lesion on the {surface_str} surfaces",
                f"early occlusal enamel demineralization involving the {surface_str} surfaces",
            ])
        return choose([
            f"cavitated dentinal caries on the {surface_str} surfaces",
            f"carious breakdown involving the {surface_str} surfaces into dentin",
        ])
    if scenario["id"] == "PERIO":
        if has_tag(dx_list, "acute_gingivitis"):
            return choose([
                "erythematous marginal gingiva with plaque-induced bleeding on probing",
                "acute plaque-related gingival inflammation with tenderness",
            ])
        if has_tag(dx_list, "gingival_recession"):
            return choose([
                "localized recession with root exposure and cervical sensitivity",
                "recession defects with root exposure and plaque retention",
            ])
        if has_tag(dx_list, "perio_loc_mod"):
            return choose([
                "localized 4-5 mm pockets with bleeding on probing",
                "site-specific moderate attachment loss with posterior bleeding",
            ])
        if has_tag(dx_list, "perio_loc_sev"):
            return choose([
                "localized 6-7 mm pockets with attachment loss and posterior bleeding",
                "severe localized pocketing with furcation concern in posterior sites",
            ])
        if has_tag(dx_list, "perio_gen_sev"):
            return choose([
                "generalized 5-7 mm probing depths with bleeding on probing and bone loss",
                "generalized severe periodontal inflammation with broad attachment loss",
            ])
        return choose([
            "generalized 4-6 mm probing depths with bleeding on probing in posterior sextants",
            "generalized plaque and calculus with localized 5-6 mm pockets",
        ])
    if scenario["id"] == "ENDO":
        if has_tag(dx_list, "rev_pulp"):
            return choose([
                "short non-lingering thermal response and deep caries without swelling",
                "provoked cold sensitivity with quick relief and no periapical swelling",
            ])
        if has_tag(dx_list, "irr_pulp"):
            return choose([
                "lingering cold response with spontaneous pain and percussion tenderness",
                "prolonged thermal pain with deep caries and chewing discomfort",
            ])
        if has_tag(dx_list, "necrotic_pulp"):
            return choose([
                "no cold response with percussion sensitivity and biting pain",
                "non-vital pulpal findings with apical tenderness and reduced vitality response",
            ])
        if has_tag(dx_list, "abscess_sinus"):
            return choose([
                "draining sinus tract with localized apical tenderness",
                "soft tissue drainage adjacent to the involved tooth with apical discomfort",
            ])
        if has_tag(dx_list, "abscess_nosinus"):
            return choose([
                "apical tenderness with swelling and no sinus tract observed",
                "localized apical swelling with percussion pain and no visible drainage",
            ])
        if has_tag(dx_list, "radicular_cyst"):
            return choose([
                "radiolucent periapical lesion suggestive of cystic change with chronic symptoms",
                "persistent periapical radiolucency suspicious for radicular cyst",
            ])
        return choose([
            "apical tenderness with symptomatic pulpal origin pain",
            "deep caries with apical periodontal widening and pain on chewing",
        ])
    if scenario["id"] == "PEDTRAUMA":
        if has_tag(dx_list, "dislocation"):
            return choose([
                "palatal displacement with mobility and tenderness to percussion",
                "traumatic displacement with mobility and localized gingival injury",
            ])
        if has_tag(dx_list, "tooth_fracture"):
            return choose([
                "incisal fracture without obvious pulpal exposure",
                "enamel-dentin fracture of the incisal edge without visible pulpal involvement",
            ])
        return choose([
            "soft tissue injury of the lip/oral cavity with tenderness around the involved tooth",
            "oral soft tissue laceration with traumatic tenderness in the anterior segment",
        ])
    if scenario["id"] == "PEDRECALL":
        if has_tag(dx_list, "routine_exam"):
            return choose([
                "routine preventive findings with no major cavitation noted today",
                "age-appropriate exam findings with minor plaque accumulation",
            ])
        if has_tag(dx_list, "exam_abnormal"):
            return choose([
                "abnormal preventive exam with plaque accumulation and localized areas of concern",
                "preventive exam notable for deposits and early disease findings",
            ])
        if has_tag(dx_list, "eruption_disturbance"):
            return choose([
                "delayed eruption pattern with space monitoring needs",
                "eruption sequence variation requiring clinical monitoring",
            ])
        if has_tag(dx_list, "smooth_enamel") or has_tag(dx_list, "pit_enamel"):
            return choose([
                f"non-cavitated enamel demineralization on the {surface_str} surfaces",
                f"early lesion on the {surface_str} surfaces with no frank cavitation",
            ])
        return choose([
            f"cavitated carious lesion involving the {surface_str} surfaces",
            f"active caries affecting the {surface_str} surfaces",
        ])
    if scenario["id"] == "IMPLANT":
        if has_tag(dx_list, "implant_early_failure"):
            return choose([
                "implant site tenderness with radiographic concern for failed osseointegration",
                "mobility concern at the implant site with early osseointegration failure features",
            ])
        if has_tag(dx_list, "implant_bio_failure"):
            return choose([
                "peri-implant inflammation with bleeding and early tissue breakdown",
                "biologic peri-implant changes with soft tissue inflammation and discomfort",
            ])
        if has_tag(dx_list, "implant_mech_failure"):
            return choose([
                "mechanical instability or component-related discomfort at the implant site",
                "occlusal/mechanical concern associated with the implant restoration complex",
            ])
        if has_tag(dx_list, "implant_other_failure"):
            return choose([
                "implant site symptoms requiring additional workup for endosseous failure",
                "persistent discomfort around the implant site with unclear endosseous failure pattern",
            ])
        return choose([
            "stable soft tissue contour and radiographic healing at the implant site",
            "implant replacement site with acceptable tissue response and no acute complication",
        ])
    if scenario["id"] == "SURG":
        if has_tag(dx_list, "impacted"):
            return choose([
                "partially erupted posterior tooth with pericoronal inflammation and tenderness",
                "impacted posterior tooth with tissue irritation and food trap",
            ])
        if has_tag(dx_list, "embedded"):
            return choose([
                "embedded tooth identified radiographically with extraction indication",
                "embedded posterior tooth associated with intermittent pain",
            ])
        if has_tag(dx_list, "retained_root"):
            return choose([
                "retained root structure with localized irritation and extraction need",
                "retained root fragment with recurrent discomfort",
            ])
        if has_tag(dx_list, "cracked_tooth"):
            return choose([
                "fractured cusp with deep crack line and pain on biting",
                "posterior crack with poor restorative prognosis and chewing pain",
            ])
        if has_tag(dx_list, "abscess_nosinus"):
            return choose([
                "swelling and apical tenderness associated with the non-restorable tooth",
                "odontogenic infection signs around the involved tooth without sinus tract",
            ])
        return choose([
            "extensive structural breakdown and poor restorative prognosis",
            "non-restorable structural loss with deep decay and fracture lines",
        ])
    if scenario["id"] == "PROSTHO":
        if has_tag(dx_list, "open_margin"):
            return choose([
                "open crown margin with localized recurrent decay concern",
                "marginal opening of existing crown with food trap",
            ])
        if has_tag(dx_list, "fractured_restoration"):
            return choose([
                "fractured crown/restorative material with loss of contour",
                "restorative fracture with missing material and sensitivity",
            ])
        if has_tag(dx_list, "other_unsat_restoration"):
            return choose([
                "existing restoration no longer clinically serviceable",
                "restoration showing deterioration and poor long-term prognosis",
            ])
        if has_tag(dx_list, "cracked_tooth"):
            return choose([
                "crack-related symptomatology requiring full-coverage planning",
                "structural crack with loss of confidence in direct restoration",
            ])
        if has_tag(dx_list, "partial_loss_caries"):
            return choose([
                "partial tooth loss and missing posterior support related to caries history",
                "missing tooth space consistent with prior caries-related loss",
            ])
        if has_tag(dx_list, "partial_loss_perio"):
            return choose([
                "partial tooth loss associated with periodontal history requiring prosthetic planning",
                "edentulous space related to periodontal tooth loss",
            ])
        return choose([
            "dislodged or failing crown requiring prosthodontic management",
            "full-coverage restoration need due to compromised tooth structure",
        ])
    if scenario["id"] == "EMERG":
        if has_tag(dx_list, "abscess_sinus"):
            return choose([
                "draining odontogenic infection with apical tenderness",
                "sinus tract with localized swelling and tooth pain",
            ])
        if has_tag(dx_list, "abscess_nosinus"):
            return choose([
                "localized swelling with percussion pain and tenderness to palpation",
                "fluctuant vestibular swelling with marked percussion sensitivity",
            ])
        if has_tag(dx_list, "mouth_abscess") or has_tag(dx_list, "cellulitis_unspecified"):
            return choose([
                "facial or vestibular swelling with diffuse tenderness adjacent to the symptomatic tooth",
                "soft tissue infection pattern with swelling and pressure near the involved tooth",
            ])
        if has_tag(dx_list, "cracked_tooth"):
            return choose([
                "sharp biting pain with crack-like symptoms and localized tenderness",
                "pain on release with suspected structural crack and apical irritation",
            ])
        return choose([
            "severe percussion sensitivity with periapical tenderness",
            "apical tenderness and pain on chewing without gross swelling",
        ])
    if scenario["id"] == "RECALL":
        if has_tag(dx_list, "routine_exam"):
            return choose([
                "light plaque and stain with no urgent restorative concerns",
                "routine preventive findings with no acute odontogenic symptoms",
            ])
        if has_tag(dx_list, "deposits"):
            return choose([
                "generalized supragingival deposits with mild gingival inflammation",
                "plaque and calculus accumulation with localized bleeding on brushing",
            ])
        if has_tag(dx_list, "chronic_gingivitis"):
            return choose([
                "mild localized gingival erythema and bleeding on brushing",
                "plaque-related gingival inflammation in localized posterior sites",
            ])
        if has_tag(dx_list, "abrasion"):
            return choose([
                "cervical wear facets consistent with abrasion",
                "non-carious cervical wear with mild sensitivity",
            ])
        if has_tag(dx_list, "bruxism"):
            return choose([
                "occlusal wear facets suggestive of parafunctional activity",
                "wear pattern and muscle history consistent with sleep bruxism",
            ])
        if has_tag(dx_list, "cracked_tooth"):
            return choose([
                "localized crack symptoms with cold sensitivity on recall exam",
                "possible crack line noted with biting discomfort history",
            ])
        return choose([
            "localized cervical/root sensitivity without active cavitation",
            "hard tissue sensitivity without obvious decay progression",
        ])
    return "documented clinical findings"

def xray_text(scenario, dx_list, tooth):
    sid = scenario["id"]
    if sid == "REST":
        return choose([
            "Bitewing radiograph was reviewed to assess lesion depth and existing restoration margins.",
            "Radiographic review supports the restorative finding without obvious pulpal exposure.",
        ])
    if sid == "PERIO":
        return choose([
            "Radiographs demonstrate periodontal changes consistent with the current maintenance needs.",
            "Films were reviewed for bone levels and periodontal stability.",
        ])
    if sid == "ENDO":
        if has_tag(dx_list, "radicular_cyst"):
            return choose([
                "Periapical radiograph shows a persistent periapical radiolucency suspicious for cystic change.",
                "PA radiograph demonstrates a well-defined periapical radiolucency compatible with cystic change.",
            ])
        if has_tag(dx_list, "abscess_sinus") or has_tag(dx_list, "abscess_nosinus"):
            return choose([
                "Periapical radiograph demonstrates periapical change consistent with odontogenic infection.",
                "PA film supports periapical pathology at the symptomatic tooth.",
            ])
        return choose([
            "Periapical radiograph supports pulpal/periapical diagnosis at the involved tooth.",
            "PA radiograph shows changes consistent with the symptomatic endodontic diagnosis.",
        ])
    if sid == "PEDTRAUMA":
        return choose([
            f"Periapical radiograph of tooth #{tooth} was reviewed for displacement or root fracture.",
            f"Trauma radiograph of tooth #{tooth} showed no obvious root fracture unless otherwise noted.",
        ])
    if sid == "PEDRECALL":
        return choose([
            "Age-appropriate radiographs were obtained as tolerated to assess proximal surfaces and eruption.",
            "Bitewing radiographs were reviewed when cooperative for interproximal and eruption assessment.",
        ])
    if sid == "IMPLANT":
        return choose([
            "Periapical radiograph was reviewed for crestal bone support and implant stability.",
            "Implant imaging was reviewed to evaluate osseointegration and surrounding bone levels.",
        ])
    if sid == "SURG":
        return choose([
            "Imaging was reviewed to assess root morphology and surgical difficulty.",
            "Radiographic review confirmed the extraction indication and anatomic considerations.",
        ])
    if sid == "PROSTHO":
        return choose([
            "Radiographic review was completed to assess tooth structure and periapical status.",
            "Imaging supported prosthodontic planning and evaluation of remaining support.",
        ])
    if sid == "EMERG":
        return choose([
            "Periapical radiograph supported the urgent odontogenic diagnosis at the involved site.",
            "PA radiograph was reviewed for apical pathology and source of infection.",
        ])
    if sid == "RECALL":
        return choose([
            "Routine radiographic review was completed based on preventive interval and risk.",
            "Radiographs were updated as indicated for recall and screening purposes.",
        ])
    return "Radiographic review completed."

def make_perio_metrics(scenario, dx_list):
    if scenario["id"] == "PERIO":
        if has_tag(dx_list, "perio_gen_sev"):
            return {"generalized_or_localized":"generalized","worst_pd_mm":7,"worst_cal_mm":6,"bop_percent":45,"bone_loss_percent":35,"perio_stage":"III","perio_grade":"B"}
        if has_tag(dx_list, "perio_loc_sev"):
            return {"generalized_or_localized":"localized","worst_pd_mm":7,"worst_cal_mm":5,"bop_percent":36,"bone_loss_percent":28,"perio_stage":"III","perio_grade":"B"}
        if has_tag(dx_list, "perio_gen_mod"):
            return {"generalized_or_localized":"generalized","worst_pd_mm":6,"worst_cal_mm":4,"bop_percent":30,"bone_loss_percent":22,"perio_stage":"II","perio_grade":"B"}
        if has_tag(dx_list, "perio_loc_mod"):
            return {"generalized_or_localized":"localized","worst_pd_mm":5,"worst_cal_mm":3,"bop_percent":24,"bone_loss_percent":18,"perio_stage":"II","perio_grade":"B"}
        if has_tag(dx_list, "gingival_recession"):
            return {"generalized_or_localized":"localized","worst_pd_mm":4,"worst_cal_mm":2,"bop_percent":18,"bone_loss_percent":8,"perio_stage":"I","perio_grade":"A"}
        return {"generalized_or_localized":"generalized","worst_pd_mm":4,"worst_cal_mm":1,"bop_percent":20,"bone_loss_percent":5,"perio_stage":"I","perio_grade":"A"}
    return {"generalized_or_localized":"none","worst_pd_mm":3,"worst_cal_mm":0,"bop_percent":0,"bone_loss_percent":0,"perio_stage":"none","perio_grade":"none"}

def make_tooth_findings(dx_list):
    findings = []
    for dx in dx_list:
        findings.append({
            "finding_code": "F_" + dx["code"].replace(".","_").replace(":","").replace("-","_"),
            "finding_text": dx["summary_text"].capitalize()
        })
    return findings

def make_prior_restorations(scenario):
    if scenario["visit_type"] in ["restorative","endo_consult","prosthodontic_restoration"]:
        return choose([[], ["occlusal_restoration"], ["MO_restoration"], ["existing_crown"], ["large_composite"]])
    return []

def proc_records(scenario, tooth):
    procs=[]
    for pc, pt in scenario["procedure_templates"]:
        title, category, desc = PROCEDURE_CODEBOOK[pc]
        txt = pt if any(k in pc for k in ["RAD_REVIEW","EXAM","OHI"]) else f"{pt}, tooth #{tooth}"
        if pc in ["PROC_IMPLANT_CHECK", "PROC_OCCLUSION_CHECK"]:
            txt = f"{pt}, site #{tooth}"
        procs.append({
            "procedure_code_system": PROC_SYSTEM,
            "procedure_code": pc if pc in ["PROC_PROPHY","PROC_EXAM","PROC_OHI","PROC_PERIO_MAINT","PROC_RAD_REVIEW","PROC_FLUORIDE"] else f"{pc}_{tooth}",
            "procedure_text": txt,
            "procedure_category": category
        })
    return procs

def procedures_here(procs):
    return [p["procedure_code"] for p in procs[: min(2, len(procs))]]

def render_risk_clause(risks):
    if not risks:
        return choose([
            "Medical history is otherwise non-contributory.",
            "Medical history was reviewed and is non-contributory for today's treatment.",
            "No additional systemic concerns affecting today's care were reported.",
        ])
    human = ", ".join(r.replace("_", " ") for r in risks)
    return choose([
        f"Relevant history includes {human}.",
        f"Risk review notable for {human}.",
        f"Pertinent contributing factors include {human}.",
    ])

def render_note(style, demo, cc, tooth, finding_sentence, xray, assessment, followup, risk_clause,
                perio_desc, injury_mech, caries_risk, plaque_state, perc_state, scenario, meds, materials):
    proc_hint = {
        "restorative": f"Composite completed on tooth #{tooth} with contacts and occlusion rechecked.",
        "perio_maintenance": "Periodontal maintenance rendered with ultrasonic and hand instrumentation.",
        "endo_consult": f"Emergency pulpotomy and temporary restoration were completed for tooth #{tooth}.",
        "emergency": "Emergency management was completed and precautions were reviewed.",
        "pediatric_recall": "Preventive services were completed and diet/oral hygiene were reviewed with the parent.",
        "implant_followup": f"Implant site #{tooth} was reviewed clinically and radiographically.",
        "oral_surgery": f"Extraction planning or surgical care was completed for tooth #{tooth}.",
        "prosthodontic_restoration": f"Prosthodontic care was completed or staged for tooth #{tooth}.",
        "recall": "Periodic exam and preventive services were completed."
    }.get(scenario["visit_type"], "Treatment completed as indicated.")
    med_text = "; ".join(meds) if meds else "none"
    material_text = ", ".join(materials[:2]) if materials else "none"

    if style == "classic_soap":
        return (
            f"SUBJECTIVE: {demo.capitalize()} presents reporting {cc}. {risk_clause}\n\n"
            f"OBJECTIVE: Tooth/site #{tooth} shows {finding_sentence}. {xray} Percussion sensitivity is {perc_state}. "
            f"Additional exam notes: {perio_desc if scenario['specialty']=='periodontics' else 'soft tissues within expected limits unless otherwise noted.'}\n\n"
            f"ASSESSMENT: {assessment}, tooth/site #{tooth}.\n\n"
            f"PLAN: {proc_hint} Materials: {material_text}. Medications reviewed/prescribed: {med_text}. Follow-up: {followup.replace('_',' ')}."
        )
    if style == "abbrev_soap":
        return (
            f"S: {demo}; cc = {cc}. {risk_clause}\n"
            f"O: #{tooth} with {finding_sentence}. {xray} Percussion: {perc_state}. "
            f"{'Perio charting: ' + perio_desc + '.' if scenario['specialty']=='periodontics' else ''}\n"
            f"A: {assessment}, #{tooth}.\n"
            f"P: {proc_hint} Meds: {med_text}. FU: {followup.replace('_',' ')}."
        )
    if style == "mixed_narrative":
        return (
            f"{demo.capitalize()} was seen today for {scenario['visit_type'].replace('_',' ')} care. "
            f"Chief concern was {cc}. {risk_clause} Clinical exam of tooth/site #{tooth} revealed {finding_sentence}. "
            f"{xray} Assessment favored {assessment.lower()} involving tooth/site #{tooth}. "
            f"Treatment today: {proc_hint} Materials used included {material_text}. "
            f"Medication status: {med_text}. Follow-up recommended: {followup.replace('_',' ')}."
        )
    if style == "structured_bullets":
        parts = [
            f"CC: {cc}.",
            f"Hx: {demo}; {risk_clause}",
            f"Exam: #{tooth} with {finding_sentence}. {xray}",
            f"Assessment: {assessment}, tooth/site #{tooth}.",
            f"Tx today: {proc_hint}",
            f"Materials: {material_text}.",
            f"Meds: {med_text}.",
            f"Follow-up: {followup.replace('_',' ')}."
        ]
        return " ".join(parts)
    if style == "assessment_forward":
        return (
            f"Assessment/Impression: {assessment}, tooth/site #{tooth}. "
            f"History: {demo} reports {cc}. {risk_clause} "
            f"Exam: {finding_sentence}. {xray} Percussion response {perc_state}. "
            f"Plan: {proc_hint} Medications: {med_text}. Follow-up is {followup.replace('_',' ')}."
        )
    if style == "ehr_shortform":
        return (
            f"Pt: {demo}. Visit: {scenario['visit_type'].replace('_',' ')}. CC: {cc}. "
            f"Tooth/site #{tooth}: {finding_sentence}. {xray} Dx: {assessment}. "
            f"Tx: {proc_hint} Mat'l: {material_text}. Rx/Meds: {med_text}. "
            f"FU: {followup.replace('_',' ')}."
        )
    raise ValueError(style)

def summary_text(style, scenario, demo, cc, tooth, dx_list, procs, followup, finding_sentence):
    dx_text = "; ".join([d["summary_text"] for d in dx_list])
    proc_short = ", ".join([p["procedure_text"].split(",")[0] for p in procs[:3]])
    age_demo = demo.replace("-year-old ", " ").replace("female", "F").replace("male", "M")
    if style == "compact_soap":
        return (
            f"S: {demo.capitalize()} seen for {scenario['visit_type'].replace('_',' ')} visit with complaint of {cc}. "
            f"O: Tooth/site #{tooth} showed {finding_sentence}. "
            f"A: {dx_text}, tooth/site #{tooth}. "
            f"P: {proc_short} completed; follow-up {followup.replace('_',' ')}."
        )
    if style == "narrative_clinical":
        return (
            f"{demo.capitalize()} presented for {scenario['visit_type'].replace('_',' ')} care involving tooth/site #{tooth}. "
            f"Exam was consistent with {dx_text} and notable for {finding_sentence}. "
            f"Management today included {proc_short.lower()}. "
            f"Plan is {followup.replace('_',' ')}."
        )
    if style == "billing_oriented":
        return (
            f"Encounter summary: {scenario['visit_type'].replace('_',' ')} visit, tooth/site #{tooth}. "
            f"Diagnoses: {dx_text}. Procedures rendered: {proc_short}. "
            f"Clinical findings included {finding_sentence}. "
            f"Disposition/follow-up: {followup.replace('_',' ')}."
        )
    if style == "handoff_brief":
        return (
            f"{age_demo} dental handoff: #{tooth} with {finding_sentence}. "
            f"Dx {dx_text}. Tx today {proc_short.lower()}. "
            f"Next step: {followup.replace('_',' ')}."
        )
    raise ValueError(style)

def ensure_summary_quality(summary, tooth, dx_list, followup, procs):
    dx_terms = [d["summary_text"].lower() for d in dx_list]
    if not any(term in summary.lower() for term in dx_terms):
        summary += f" Diagnosis: {dx_list[0]['summary_text']}, tooth/site #{tooth}."
    proc_terms = [p["procedure_text"].split(",")[0].lower() for p in procs[:2]]
    if not any(term in summary.lower() for term in proc_terms):
        summary += f" Management included {procs[0]['procedure_text'].split(',')[0].lower()}."
    if "follow-up" not in summary.lower() and "next step" not in summary.lower() and "plan" not in summary.lower():
        summary += f" Follow-up: {followup.replace('_',' ')}."
    if f"#{tooth}" not in summary:
        summary += f" Tooth/site #{tooth}."
    summary = re.sub(r"\s+", " ", summary).strip()
    return summary

def primary_task_for(scenario):
    mapping = {
        "restorative":"summarization_soap",
        "perio_maintenance":"perio_metric_extraction",
        "endo_consult":"multi_label_diagnosis",
        "emergency":"summarization_soap",
        "pediatric_recall":"procedure_prediction",
        "implant_followup":"visit_type_classification",
        "oral_surgery":"procedure_prediction",
        "prosthodontic_restoration":"procedure_prediction",
        "recall":"visit_type_classification",
    }
    return mapping.get(scenario["visit_type"], "summarization_soap")

def metrics_for(scenario):
    if scenario["specialty"] == "periodontics":
        return ["ROUGE_L", "BERTScore", "F1_diag", "F1_proc", "MAE_pd_mm"]
    return ["ROUGE_L", "BERTScore", "F1_diag", "F1_proc", "accuracy_visit_type"]

def assign_annotation(split, expert_counter):
    if split == "train" and expert_counter["train"] < 1000:
        expert_counter["train"] += 1
        return "expert_dentist", "expert", 1
    if split == "validation" and expert_counter["validation"] < 500:
        expert_counter["validation"] += 1
        return "expert_dentist", "expert", 1
    if split == "test" and expert_counter["test"] < 500:
        expert_counter["test"] += 1
        return "expert_dentist", "expert", 1
    annotator = choose(["synthetic_pipeline","hybrid_dental_review","resident"])
    source = "synthetic_reviewed" if annotator != "resident" else "hybrid_review"
    round_num = choose([1,1,2])
    return annotator, source, round_num

def coverage_checks(rec):
    note = rec["input_note"].lower()
    summary = rec["summary_SOAP"].lower()
    checks = {
        "note_has_cc": bool(rec["chief_complaint"]),
        "note_has_findings": bool(rec["findings_text"]),
        "note_has_followup": rec["follow_up_recommendation"].replace("_"," ") in note or "follow-up" in note or "fu:" in note,
        "summary_has_dx": any(d["diagnosis_text"].split(",")[0].lower() in summary or d.get("summary_text","").lower() in summary for d in rec["diagnoses"]),
        "summary_has_plan": any(token in summary for token in ["follow-up", "next step", "plan", "fu"]),
        "summary_has_tooth": any(f"#{t['tooth_number']}".lower() in summary for t in rec["tooth_level"]),
    }
    return checks

def build_record(note_num, split, scenario, expert_counter):
    tooth = choose(scenario["teeth"])
    surfaces = choose(scenario["surfaces"])
    sex = choose(scenario["sexes"])
    age = choose(scenario["age_groups"])
    demo = random_demo(age, sex)
    cc = choose(scenario["chief_complaints"])
    dx_list = choose(scenario["diagnosis_options"])
    duration = choose(DURATIONS)
    injury_mech = choose(INJURY_MECHS)
    followup = choose(scenario["followups"])
    finding_sentence = findings_for_scenario(scenario, dx_list, tooth, surfaces)
    xray = xray_text(scenario, dx_list, tooth)
    assessment = "; ".join([dx["official_text"] for dx in dx_list])
    risk_candidates = [r for r in sample_some([r for r in scenario["risk_pool"] if r != "none"], max_k=2)]
    sdoh_candidates = [r for r in sample_some([r for r in scenario["sdoh_pool"] if r != "none"], max_k=1)]
    meds = list(dict.fromkeys(choose(MEDICATION_POOL).copy()))
    allergies = choose(ALLERGY_POOL)
    if any(pc[0] == "PROC_LOCAL_ANES" for pc in scenario["procedure_templates"]):
        meds = list(dict.fromkeys(meds + [choose(ANESTHETICS) + " (local anesthetic)"]))
    if scenario["id"] == "EMERG" and "Amoxicillin" not in meds and random.random() < 0.55:
        meds.append("Amoxicillin")
    if scenario["id"] == "IMPLANT" and "Chlorhexidine rinse" not in meds and random.random() < 0.50:
        meds.append("Chlorhexidine rinse")
    risk_clause = render_risk_clause(risk_candidates)
    perc_state = choose(PERC_STATES)
    note_style = choose(NOTE_STYLE_POOL)
    summary_style = choose(SUMMARY_STYLE_POOL)
    perio_desc = choose([
        "probing depths ranging from 3-6 mm with posterior bleeding on probing",
        "generalized 4-6 mm pockets and moderate plaque retention",
        "localized deeper posterior pockets with generalized inflammation"
    ])
    input_note = render_note(
        note_style, demo, cc, tooth, finding_sentence, xray, assessment,
        followup, risk_clause, perio_desc, injury_mech, choose(CARIES_RISKS),
        choose(PLAQUE_STATES), perc_state, scenario, meds, scenario["materials"]
    )
    # add small realistic variability without breaking core coverage
    if random.random() < 0.22:
        input_note += " Patient tolerated the visit well."
    if random.random() < 0.14:
        input_note += " Questions were answered and consent/understanding was documented."
    if scenario["id"] == "PEDTRAUMA" and random.random() < 0.25:
        input_note = input_note.replace("SUBJECTIVE:", "HISTORY:").replace("OBJECTIVE:", "EXAM:")
    if scenario["id"] == "RECALL" and has_tag(dx_list, "bruxism") and "nightguard" not in input_note.lower():
        input_note += " Discussed possible nightguard therapy if symptoms persist."
    if scenario["id"] == "PEDTRAUMA":
        input_note = input_note.replace("Chief concern was", f"Following {injury_mech}, the chief concern was")
    procs = proc_records(scenario, tooth)
    diagnoses = [{
        "diagnosis_code_system": DIAG_SYSTEM,
        "diagnosis_code": dx["code"],
        "diagnosis_text": f"{dx['official_text']}, tooth/site #{tooth}",
        "summary_text": dx["summary_text"],
        "diagnosis_source_label": dx["tag"],
    } for dx in dx_list]
    tooth_level = [{
        "tooth_number": tooth,
        "surfaces": normalize_surface_list(surfaces),
        "findings": make_tooth_findings(dx_list),
        "prior_restorations": make_prior_restorations(scenario),
        "procedures_here": procedures_here(procs)
    }]
    if scenario["id"] == "PERIO" and random.random() < 0.45:
        extra_tooth = choose([t for t in scenario["teeth"] if t != tooth])
        tooth_level.append({
            "tooth_number": extra_tooth,
            "surfaces": ["B","L"],
            "findings": [{"finding_code":"F_BOP","finding_text":"Bleeding on probing"}],
            "prior_restorations": [],
            "procedures_here": ["PROC_PERIO_MAINT"]
        })
    if scenario["id"] == "PEDTRAUMA" and random.random() < 0.35:
        adj = choose([t for t in ["7","8","9","10"] if t != tooth])
        tooth_level.append({
            "tooth_number": adj,
            "surfaces": ["facial"],
            "findings": [{"finding_code":"F_ADJ_TOOTH_STABLE","finding_text":"Adjacent tooth stable"}],
            "prior_restorations": [],
            "procedures_here": []
        })
    findings_text = f"{assessment} involving tooth/site #{tooth}; {finding_sentence}."
    if scenario["specialty"] == "periodontics":
        findings_text += " BOP and periodontal charting were consistent with the documented severity."
    if scenario["id"] == "IMPLANT":
        findings_text += " Implant site was assessed for tissue health, stability, and radiographic support."
    summary = summary_text(summary_style, scenario, demo, cc, tooth, dx_list, procs, followup, finding_sentence)
    summary = ensure_summary_quality(summary, tooth, dx_list, followup, procs)
    annotator_type, summary_source, annotation_round = assign_annotation(split, expert_counter)

    rec = {
        "note_id": f"DN_{note_num:06d}",
        "version": "v2.0",
        "split": split,
        "generation_method": choose(["template_rule_hybrid", "template_llm", "hybrid_prompt_pipeline"]),
        "source_prompt_id": scenario["prompt_id"] + f"_{random.randint(1,12):02d}",
        "visit_type": scenario["visit_type"],
        "specialty": scenario["specialty"],
        "patient_age_group": age,
        "patient_sex": sex,
        "emergency_flag": scenario["emergency_flag"],
        "diagnosis_code_system": DIAG_SYSTEM,
        "procedure_code_system": PROC_SYSTEM,
        "input_note": input_note,
        "chief_complaint": cc.capitalize() + "." if not cc.endswith(".") else cc,
        "findings_text": findings_text,
        "summary_SOAP": summary,
        "summary_type": "SOAP" if summary_style in ["compact_soap","narrative_clinical"] else "brief_clinical_summary",
        "summary_source": summary_source,
        "has_summary": True,
        "diagnoses": diagnoses,
        "procedures": procs,
        "tooth_level": tooth_level,
        "materials_used": list(dict.fromkeys(scenario["materials"] + ([choose(["temporary_cement","hemostatic_agent","bonding_agent"])] if random.random() < 0.22 else []))),
        "perio_metrics": make_perio_metrics(scenario, dx_list),
        "risk_factors": risk_candidates,
        "sdoh_factors": sdoh_candidates,
        "medications": meds,
        "allergies": [a["name"] + (f" ({a['reaction']})" if a.get("reaction") else "") for a in allergies],
        "follow_up_recommendation": followup,
        "task_tags": list(dict.fromkeys(
            ["summarization_soap", "multi_label_diagnosis", "procedure_prediction", "visit_type_classification", "tooth_ner"] +
            (["perio_metric_extraction"] if scenario["specialty"] == "periodontics" else []) +
            (["trauma_management"] if scenario["id"] == "PEDTRAUMA" else [])
        )),
        "primary_task": primary_task_for(scenario),
        "metrics_recommended": metrics_for(scenario),
        "annotator_type": annotator_type,
        "annotation_round": annotation_round,
        "note_style": note_style,
        "summary_style": summary_style,
    }
    rec["coverage_checks"] = coverage_checks(rec)
    return rec

def write_codebooks(records):
    diag_rows = {}
    for sc in SCENARIOS:
        for option in sc["diagnosis_options"]:
            for dx in option:
                diag_rows[dx["code"]] = {
                    "diagnosis_code": dx["code"],
                    "diagnosis_text": dx["official_text"],
                    "summary_label": dx["summary_text"],
                    "scenario_family": sc["id"],
                    "specialty": sc["specialty"],
                }
    proc_rows = []
    for code, (title, category, desc) in PROCEDURE_CODEBOOK.items():
        proc_rows.append({
            "procedure_code_system": PROC_SYSTEM,
            "procedure_code": code,
            "procedure_text": title,
            "procedure_category": category,
            "definition": desc,
        })

    # CSV files
    import csv
    with ICD_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["diagnosis_code","diagnosis_text","summary_label","scenario_family","specialty"])
        w.writeheader()
        for row in sorted(diag_rows.values(), key=lambda x: x["diagnosis_code"]):
            w.writerow(row)
    with PROC_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["procedure_code_system","procedure_code","procedure_text","procedure_category","definition"])
        w.writeheader()
        for row in sorted(proc_rows, key=lambda x: x["procedure_code"]):
            w.writerow(row)

    # Markdown tables
    icd_lines = [
        "# ICD-10-CM diagnosis mappings (v2.0)",
        "",
        "| diagnosis_code | diagnosis_text | summary_label | scenario_family | specialty |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(diag_rows.values(), key=lambda x: x["diagnosis_code"]):
        icd_lines.append(f"| {row['diagnosis_code']} | {row['diagnosis_text']} | {row['summary_label']} | {row['scenario_family']} | {row['specialty']} |")
    ICD_MD.write_text("\n".join(icd_lines), encoding="utf-8")

    proc_lines = [
        "# Internal procedure code definitions (v2.0)",
        "",
        "| procedure_code_system | procedure_code | procedure_text | procedure_category | definition |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(proc_rows, key=lambda x: x["procedure_code"]):
        proc_lines.append(f"| {row['procedure_code_system']} | {row['procedure_code']} | {row['procedure_text']} | {row['procedure_category']} | {row['definition']} |")
    PROC_MD.write_text("\n".join(proc_lines), encoding="utf-8")

def main(n=20000):
    scenario_plan = []
    per_scenario = n // len(SCENARIOS)
    remainder = n % len(SCENARIOS)
    for idx, sc in enumerate(SCENARIOS):
        scenario_plan.extend([sc] * (per_scenario + (1 if idx < remainder else 0)))
    random.shuffle(scenario_plan)

    # exact split plan: 70/15/15
    split_plan = ["train"] * int(n * 0.70) + ["validation"] * int(n * 0.15)
    split_plan += ["test"] * (n - len(split_plan))
    random.shuffle(split_plan)

    expert_counter = {"train": 0, "validation": 0, "test": 0}
    records = []
    with OUT.open("w", encoding="utf-8") as f_out:
        for i, (sc, split) in enumerate(zip(scenario_plan, split_plan), start=1):
            rec = build_record(i, split, sc, expert_counter)
            records.append(rec)
            f_out.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    write_codebooks(records)

    stats = {
        "version": "v2.0",
        "total_records": len(records),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "specialty_counts": dict(Counter(r["specialty"] for r in records)),
        "visit_type_counts": dict(Counter(r["visit_type"] for r in records)),
        "diagnosis_code_system": DIAG_SYSTEM,
        "procedure_code_system": PROC_SYSTEM,
        "unique_diagnosis_codes": sorted({d["diagnosis_code"] for r in records for d in r["diagnoses"]}),
        "diagnosis_code_count": len({d["diagnosis_code"] for r in records for d in r["diagnoses"]}),
        "unique_procedure_codes": sorted({p["procedure_code"].split("_")[0] + "_" + p["procedure_code"].split("_")[1] if p["procedure_code"].startswith("PROC_") and p["procedure_code"] not in ["PROC_PROPHY","PROC_EXAM","PROC_OHI","PROC_PERIO_MAINT","PROC_RAD_REVIEW","PROC_FLUORIDE"] else p["procedure_code"] for r in records for p in r["procedures"]}),
        "procedure_code_count": len({(p["procedure_code"] if p["procedure_code"] in ["PROC_PROPHY","PROC_EXAM","PROC_OHI","PROC_PERIO_MAINT","PROC_RAD_REVIEW","PROC_FLUORIDE"] else "_".join(p["procedure_code"].split("_")[:2])) for r in records for p in r["procedures"]}),
        "diagnosis_code_freq": dict(Counter(d["diagnosis_code"] for r in records for d in r["diagnoses"])),
        "procedure_code_freq": dict(Counter((p["procedure_code"] if p["procedure_code"] in ["PROC_PROPHY","PROC_EXAM","PROC_OHI","PROC_PERIO_MAINT","PROC_RAD_REVIEW","PROC_FLUORIDE"] else "_".join(p["procedure_code"].split("_")[:2])) for r in records for p in r["procedures"])),
        "avg_note_chars": round(statistics.mean(len(r["input_note"]) for r in records), 2),
        "avg_summary_chars": round(statistics.mean(len(r["summary_SOAP"]) for r in records), 2),
        "summary_style_counts": dict(Counter(r["summary_style"] for r in records)),
        "note_style_counts": dict(Counter(r["note_style"] for r in records)),
        "coverage_pass_counts": {key: sum(bool(r["coverage_checks"].get(key)) for r in records) for key in records[0]["coverage_checks"].keys()},
        "expert_subset_counts": dict(Counter(r["split"] for r in records if r["annotator_type"] == "expert_dentist")),
    }
    STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [OUT, STATS, SCRIPT_OUT, ICD_CSV, PROC_CSV, ICD_MD, PROC_MD]:
            zf.write(p, p.name)

    print(OUT)
    print(STATS)
    print(ICD_CSV)
    print(PROC_CSV)
    print(ZIP_OUT)

if __name__ == "__main__":
    main()
