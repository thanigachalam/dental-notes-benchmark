#!/usr/bin/env python3
"""Evidence inventory; distinguishes observed dataset metadata from source-code defaults and runbook examples."""
import argparse,collections,csv,hashlib,re,platform,subprocess
from pathlib import Path
from common import records,write_csv,write_json,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--generator',required=True);p.add_argument('--summary-script',required=True);p.add_argument('--run-log');p.add_argument('--out',required=True);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 count=0;counts={k:collections.Counter() for k in ('source_prompt_id','llm_summary_model','llm_summary_source','llm_summary_prompt_version','llm_summary_validation_status')}
 for r in records(a.input):
  count+=1
  for k,c in counts.items():c[str(r.get(k,'<ABSENT>'))]+=1
 rows=[];paths={}
 for role,location in [('structured_generator',a.generator),('summary_augmentation',a.summary_script),('execution_log',a.run_log)]:
  if not location:paths[role]={'status':'not_provided'};continue
  path=Path(location)
  if not path.exists():paths[role]={'status':'missing','path':str(path)};continue
  body=path.read_text(errors='replace');paths[role]={'status':'present','path':str(path),'sha256':digest(path),'bytes':path.stat().st_size}
  for pattern,label in [(r'20260313','seed_20260313'),(r'0\.35','temperature_0_35'),(r'0\.9\b','top_p_0_9'),(r'1\.08','repeat_penalty_1_08'),(r'--num-predict\s+380','num_predict_380'),(r'--num-ctx\s+4096','num_ctx_4096'),(r'llama3\.1:8b-instruct-q4_K_M','model_tag')]:
   rows.append({'source':role,'setting':label,'text_occurrence_found':bool(re.search(pattern,body)),'evidence_level':'SOURCE_TEXT_ONLY (not proof of executed setting)'})
 write_csv(out/'generation_config_inventory.csv',rows)
 distro=[]
 for key,counter in counts.items():
  for v,n in counter.most_common():distro.append({'field':key,'value':v,'count':n})
 write_csv(out/'record_provenance_distributions.csv',distro)
 report={'records':count,'distinct_source_prompt_ids':len(counts['source_prompt_id']),'recorded_metadata':{k:dict(v) for k,v in counts.items()},'file_inventory':paths,'python_version_current_machine':platform.python_version(),'warnings':['The current machine is not necessarily the original generation machine.','Source-code settings may be defaults rather than run-time arguments.','Prompt identifiers are not verified underlying template-family counts.','Do not infer original Ollama version or model digest without execution evidence.']}
 write_json(out/'generation_provenance_report.json',report)
 print('Provenance inventory:',count,'records;',len(counts['source_prompt_id']),'source prompt identifiers')
if __name__=='__main__':main()
