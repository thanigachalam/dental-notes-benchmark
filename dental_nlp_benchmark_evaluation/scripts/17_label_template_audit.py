#!/usr/bin/env python3
"""Read-only structural/label-code audit of SDCN-DNLP v21.

Does not claim semantic leakage or clinical realism. Does not modify data/splits.
"""
import argparse, csv, json, re, zipfile
from collections import Counter, defaultdict
from pathlib import Path

ID_KEYS = ('record_id','note_id','sample_id','id')
TEXT_KEYS = ('input_note','clinical_note','note_text','text','note')
LABEL_HINTS = {'diagnosis':('diagnosis_codes','diagnosis_code','icd10_codes','icd10_code','icd10_cm_codes','diagnoses'),
               'procedure':('procedure_codes','procedure_code','internal_procedure_codes','procedure_labels','procedures')}
SPLIT_KEYS = ('split','partition','subset','set')


def get(rec, key):
    if key in rec: return rec[key]
    for parent in ('metadata','provenance','generation_metadata'):
        nested=rec.get(parent)
        if isinstance(nested,dict) and key in nested: return nested[key]
    return None

def pick(keys, choices, kind):
    matches=[k for k in choices if k in keys]
    if not matches: raise ValueError(f'Cannot locate {kind}; available fields: {list(keys)}')
    return matches[0]

def load_zip(path):
    with zipfile.ZipFile(path) as z:
        files=[n for n in z.namelist() if n.endswith('.jsonl') and not n.startswith('__MACOSX/') and not Path(n).name.startswith('._')]
        if len(files)!=1: raise ValueError(f'Expected one real JSONL member: {files}')
        with z.open(files[0]) as f:
            for b in f:
                if b.strip(): yield json.loads(b)

def normalize_split(s):
    s=str(s).strip().lower()
    return {'train':'train','training':'train','val':'validation','valid':'validation','dev':'validation','validation':'validation','test':'test','testing':'test'}[s]

def labels(v):
    if v is None: return []
    if isinstance(v,dict):
        # nested code lists can differ; avoid guessing which dictionary field is the label
        return []
    if isinstance(v,list):
        return [str(x).strip() for x in v if isinstance(x,(str,int,float)) and str(x).strip()]
    if isinstance(v,(str,int,float)) and str(v).strip():
        return [str(v).strip()]
    return []

def skeleton(text):
    # Conservative structural proxy, NOT an original-generator template ID.
    t=str(text).lower()
    t=re.sub(r'\b(?:[a-z]\d{2}(?:\.\d{1,4})?)\b',' <code> ',t)
    t=re.sub(r'\b\d+(?:\.\d+)?\b',' <num> ',t)
    t=re.sub(r'\s+',' ',t).strip()
    return t

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--published',required=True)
    p.add_argument('--prompt-disjoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--text-field',default=None,help='Explicit field name if auto-detection fails')
    p.add_argument('--diagnosis-field',default=None)
    p.add_argument('--procedure-field',default=None)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data=list(load_zip(a.input))
    if not data: raise SystemExit('Empty input')
    fields=set(data[0]); fields.update(k for par in ('metadata','provenance','generation_metadata') if isinstance(data[0].get(par),dict) for k in data[0][par])
    idfield=pick(fields,ID_KEYS,'record ID')
    textfield=a.text_field or pick(fields,TEXT_KEYS,'input note text')
    diafield=a.diagnosis_field or next((k for k in LABEL_HINTS['diagnosis'] if k in fields),None)
    procfield=a.procedure_field or next((k for k in LABEL_HINTS['procedure'] if k in fields),None)
    print('Discovered fields:',json.dumps({'id':idfield,'text':textfield,'diagnosis':diafield,'procedure':procfield}))
    ids={}
    for r in data:
        rid=get(r,idfield)
        if rid is None: raise ValueError('Missing record ID')
        rid=str(rid)
        if rid in ids: raise ValueError(f'Duplicate dataset ID: {rid}')
        ids[rid]=r
    if len(ids)!=20000: print('WARNING: record count differs from 20,000:',len(ids))
    manifest_stats={}; family_matrix=[]; text_stats=[]; coverage_rows=[]
    for strategy,manifest in [('published',a.published),('prompt_disjoint',a.prompt_disjoint)]:
        with open(manifest,newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            mid=pick(reader.fieldnames,ID_KEYS,'manifest record ID')
            ms=pick(reader.fieldnames,SPLIT_KEYS,'manifest split')
            assigned={}
            for row in reader:
                rid=str(row[mid]).strip()
                if rid in assigned: raise ValueError(f'Duplicate manifest ID {rid}: {manifest}')
                assigned[rid]=normalize_split(row[ms])
        if set(assigned)!=set(ids): raise ValueError(f'{strategy}: missing dataset IDs={len(set(ids)-set(assigned))}, unknown IDs={len(set(assigned)-set(ids))}')
        labelset={task:defaultdict(set) for task in ('diagnosis','procedure')}
        family_counts=defaultdict(Counter)
        skeleton_sets=defaultdict(set)
        exact_sets=defaultdict(set)
        code_match=defaultdict(Counter)
        label_totals=defaultdict(Counter)
        for rid,r in ids.items():
            part=assigned[rid]
            note=get(r,textfield)
            if not isinstance(note,str): raise ValueError(f'Input text not string for {rid}: field {textfield}')
            prompt=get(r,'source_prompt_id')
            family=re.sub(r'_\d{2}$','',str(prompt)) if prompt is not None else '(missing)'
            family_counts[family][part]+=1
            sk=skeleton(note)
            skeleton_sets[part].add(sk)
            exact_sets[part].add(note)
            for task,field in [('diagnosis',diafield),('procedure',procfield)]:
                if not field: continue
                arr=labels(get(r,field))
                labelset[task][part].update(arr)
                for code in arr:
                    label_totals[(task,part)]['total']+=1
                    # literal code string detection with alphanumeric boundaries only;
                    # not a test of semantic label mention or genuine information leakage
                    if re.search(r'(?<![A-Za-z0-9])'+re.escape(code)+r'(?![A-Za-z0-9])',note,re.I):
                        code_match[(task,part)]['matches']+=1
        for family,c in sorted(family_counts.items()):
            family_matrix.append({'strategy':strategy,'family':family,**{part:c[part] for part in ('train','validation','test')}})
        for task in ('diagnosis','procedure'):
            tr=labelset[task]['train']
            for part in ('validation','test'):
                unseen=labelset[task][part]-tr
                coverage_rows.append({'strategy':strategy,'task':task,'split':part,'n_train_labels':len(tr),'n_split_labels':len(labelset[task][part]),'n_unseen_labels':len(unseen),'unseen_labels':'|'.join(sorted(unseen))})
        partitions=('train','validation','test')
        overlaps={}
        for one,two in [('train','validation'),('train','test'),('validation','test')]:
            overlaps[f'{one}_vs_{two}']={'exact_input_text_shared_unique':len(exact_sets[one]&exact_sets[two]),'normalized_skeleton_shared_unique':len(skeleton_sets[one]&skeleton_sets[two])}
        manifest_stats[strategy]={'partition_counts':dict(Counter(assigned.values())),'source_prompt_family_counts':len(family_counts),'text_overlap':overlaps,'literal_code_mentions':{f'{task}_{part}':{'matched_label_occurrences':code_match[(task,part)]['matches'],'total_label_occurrences':label_totals[(task,part)]['total']} for task in ('diagnosis','procedure') for part in partitions}}
        print(strategy,manifest_stats[strategy]['partition_counts'],'text overlaps',overlaps)
    with open(out/'scenario_family_by_split.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['strategy','family','train','validation','test']);w.writeheader();w.writerows(family_matrix)
    with open(out/'label_coverage_by_split.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['strategy','task','split','n_train_labels','n_split_labels','n_unseen_labels','unseen_labels']);w.writeheader();w.writerows(coverage_rows)
    result={'n_records':len(ids),'fields':{'id':idfield,'text':textfield,'diagnosis':diafield,'procedure':procfield},'strategies':manifest_stats,'caveats':['Skeleton uses simple numerical/code normalization and is NOT a verified generator template.','Exact label-code appearance in text is not a sufficient test for clinical leakage; target label words/synonyms may be present without a code.','Missing label field means code-mention and label-coverage analysis is unavailable for that task.','Scenario families are inferred from source_prompt_id suffixes; validation against generator code is required.','No real-note external generalization is tested.']}
    (out/'label_template_audit.json').write_text(json.dumps(result,indent=2))
    print('Saved:',out)
    print('Review fields and caveats in label_template_audit.json before interpretation.')

if __name__=='__main__':main()
