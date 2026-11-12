#!/usr/bin/env python3
"""Local artifact/metadata inventory. No network or DataPort modifications."""
import argparse,hashlib,json,csv,re
from pathlib import Path
from common import records,write_json,write_csv,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--repo-root',required=True);p.add_argument('--evaluation-root',default='dental_nlp_benchmark_evaluation');p.add_argument('--out',required=True);a=p.parse_args()
 root=Path(a.repo_root).resolve();er=(root/a.evaluation_root).resolve();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 targets=[('root','README.md'),('root','LICENSE'),('root','CITATION.cff'),('root','codebooks/icd10_cm_diagnosis_mappings_v21.csv'),('root','codebooks/internal_procedure_code_definitions_v21.csv'),('root','generators/generate_dentalnote_bench_20000_icd10_internalproc_v21.py')]
 targets += [('eval','requirements.txt'),('eval','README.md')]
 targets += [('eval',f'scripts/{i:02d}_{name}.py') for i,name in [(1,'dataset_audit'),(2,'build_splits'),(3,'duplicate_analysis'),(4,'baseline_benchmark'),(5,'check_split_integrity'),(6,'generate_paper_figures_tables'),(7,'baseline_integrity_audit'),(8,'cross_split_duplicate_audit'),(9,'template_provenance_audit'),(10,'clinicalbert_benchmark'),(11,'generation_provenance_audit'),(12,'release_integrity_audit')]]
 entries=[]
 for scope,relative in targets:
  path=(root if scope=='root' else er)/relative
  entries.append({'scope':scope,'artifact':relative,'exists':path.is_file(),'bytes':path.stat().st_size if path.is_file() else '', 'sha256':digest(path) if path.is_file() else ''})
 write_csv(out/'artifact_manifest.csv',entries)
 license_file=root/'LICENSE';license_text=license_file.read_text(errors='replace') if license_file.exists() else ''
 readme_text=(root/'README.md').read_text(errors='replace') if (root/'README.md').exists() else ''
 citation_text=(root/'CITATION.cff').read_text(errors='replace') if (root/'CITATION.cff').exists() else ''
 counts=0;unique=set();splits={}
 for r in records(a.input):
  counts+=1;unique.add(str(r.get('note_id')));s=str(r.get('split','<ABSENT>'));splits[s]=splits.get(s,0)+1
 verdict={'record_count':counts,'unique_note_ids':len(unique),'record_split_counts':splits,'local_artifacts_present':sum(x['exists'] for x in entries),'local_artifacts_checked':len(entries),'missing_local_artifacts':[x['artifact'] for x in entries if not x['exists']],'license_check':{'contains_cc_by_4_0':bool(re.search(r'CC\s*BY\s*4\.0|Attribution\s*4\.0',license_text,re.I)),'readme_mentions_cc_by':bool(re.search(r'CC\s*BY\s*4\.0',readme_text,re.I)),'citation_mentions_cc_by':bool(re.search(r'CC\s*BY\s*4\.0',citation_text,re.I))},'limitations':['Local file inventory cannot verify publicly accessible GitHub or DataPort metadata.','A license applied to software may differ from the data license; verify rights for each artifact.','This script does not update any files or repositories.']}
 write_json(out/'release_integrity_report.json',verdict)
 (out/'reproducibility_checklist.md').write_text('# Release verification checklist\n\n- [ ] Review missing local files in artifact_manifest.csv\n- [ ] Check DataPort public DOI metadata and exact license manually\n- [ ] Check GitHub README, LICENSE and CITATION.cff\n- [ ] Validate v21 codebooks against the *historical generation-era* official ICD-10-CM release\n- [ ] Record Git commit and original generation environment where known\n- [ ] Confirm no clinical notes or data ZIPs staged for GitHub\n- [ ] Confirm saved benchmark CSVs and split integrity results\n',encoding='utf8')
 print('Release integrity:',counts,'records,',len(unique),'unique IDs;',verdict['local_artifacts_present'],'/',len(entries),'local artifacts found')
if __name__=='__main__':main()
