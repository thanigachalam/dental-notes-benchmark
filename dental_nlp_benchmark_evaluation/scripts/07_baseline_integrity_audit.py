#!/usr/bin/env python3
"""Read-only baseline protocol and label-distribution audit. Does NOT certify model performance."""
import argparse,csv,ast,json,collections,hashlib
from pathlib import Path
from common import records,get_labels,write_csv,write_json

def read_manifest(path):
 with open(path,newline='',encoding='utf8') as f:
  rows=list(csv.DictReader(f))
 if not {'note_id','split'} <= set(rows[0] if rows else []): raise ValueError(f'{path}: expected note_id,split')
 ids=[x['note_id'] for x in rows]
 if len(ids)!=len(set(ids)): raise ValueError(f'{path}: duplicate note ID')
 return {x['note_id']:x['split'] for x in rows}

def run(a):
 data=list(records(a.input));byid={str(x['note_id']):x for x in data}; out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 if len(data)!=len(byid):raise ValueError('Input has duplicate IDs')
 rows=[];labels=[];warnings=[];sample=[]
 for manifest in a.splits:
  assign=read_manifest(manifest);name=Path(manifest).name
  if set(assign)!=set(byid):raise ValueError(f'{name}: missing={len(set(byid)-set(assign))}, unknown={len(set(assign)-set(byid))}')
  parts={s:[byid[n] for n,v in assign.items() if v==s] for s in ('train','validation','test')}
  hashes={s:{hashlib.sha256(' '.join(str(r.get('input_note','')).lower().split()).encode()).hexdigest() for r in rs} for s,rs in parts.items()}
  same_hashes=len(hashes['train']&hashes['test']);warnings += ([f'{name}: {same_hashes} normalized exact input_note hashes shared train/test'] if same_hashes else [])
  for task in ('diagnosis','procedure'):
   tr={x for r in parts['train'] for x in get_labels(r,task)}
   va={x for r in parts['validation'] for x in get_labels(r,task)}
   te={x for r in parts['test'] for x in get_labels(r,task)}
   ct={s:collections.Counter(x for r in rs for x in get_labels(r,task)) for s,rs in parts.items()}
   for code in sorted(tr|va|te): labels.append({'manifest':name,'task':task,'label':code,'train_occurrences':ct['train'][code],'validation_occurrences':ct['validation'][code],'test_occurrences':ct['test'][code],'unseen_in_train':code not in tr})
   rows.append({'manifest':name,'task':task,'train_records':len(parts['train']),'validation_records':len(parts['validation']),'test_records':len(parts['test']),'labels_train':len(tr),'labels_validation':len(va),'labels_test':len(te),'unseen_test_labels':len(te-tr),'unseen_test_label_names':'|'.join(sorted(te-tr)),'train_test_exact_text_hash_overlap':same_hashes,'nonempty_input_notes':sum(bool(str(r.get('input_note','')).strip()) for r in data)})
  for r in parts['test'][:3]:sample.append({'manifest':name,'note_id':r['note_id'],'input_note_length':len(str(r.get('input_note',''))),'note_style':r.get('note_style','')})
 write_csv(out/'baseline_integrity_summary.csv',rows);write_csv(out/'label_coverage.csv',labels);write_csv(out/'input_feature_sample_metadata.csv',sample)
 baseline_path=Path(a.baseline_script)
 source=baseline_path.read_text(encoding='utf8') if baseline_path.exists() else ''
 checks={'baseline_script_found':bool(source),'rule_predict_uses_input_note':"r.get('input_note'" in source,'tfidf_train_uses_input_note':"r['input_note'] for r in train" in source,'tfidf_test_uses_input_note':"r['input_note'] for r in test" in source,'uses_test_targets_for_predictions':False if source and 'pipeline.predict' in source else 'not_assessed','fit_called_on_train_inputs':"pipeline.fit([r['input_note'] for r in train]" in source,'validation_used_for_tuning':'not implemented in reviewed baseline script','interpretation':'static source inspection only; not proof of no indirect label/template leakage'}
 try:ast.parse(source)
 except Exception as e:checks['python_source_parse_error']=str(e)
 if a.results and Path(a.results).exists():
  with open(a.results,newline='',encoding='utf8') as f:logged=list(csv.DictReader(f))
  checks['saved_metrics_rows']=len(logged)
  key={(r['manifest'],r['task']):(r['labels_train'],r['unseen_test_labels']) for r in rows}
  mismatches=[]
  for r in logged:
   k=(r.get('split_manifest'),r.get('task'))
   if k in key:
    for col,val in [('labels_train',key[k][0]),('unseen_test_labels',key[k][1])]:
     if col in r and r[col] and int(float(r[col]))!=val:mismatches.append({'key':k,'field':col,'reported':r[col],'recomputed':val})
  checks['saved_metrics_label_count_discrepancies']=mismatches
 else:checks['saved_metrics_status']='not provided; model scores cannot be independently verified from this audit'
 report={'records':len(data),'manifests':[Path(p).name for p in a.splits],'code_checks':checks,'warnings':warnings,'methodology_cautions':['Macro-F1 implementation must be verified against intended label universe; a perfect synthetic benchmark is not proof of clinical generalization.','Rule baseline constructs label phrases from training label descriptions; these labels may correlate closely with generator wording.','TF-IDF tokenization and parameters derive from training data, but independent replication and predicted labels are needed for full performance audit.','No test prediction files are exported by original 04 script, so this audit cannot independently recompute per-label F1.'],'results':rows}
 write_json(out/'baseline_integrity_report.json',report);print('Baseline integrity audit:',len(data),'records;',len(rows),'task/split combinations; warnings:',len(warnings))

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--splits',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--results');p.add_argument('--baseline-script',default='dental_nlp_benchmark_evaluation/scripts/04_baseline_benchmark.py');run(p.parse_args())
if __name__=='__main__':main()
