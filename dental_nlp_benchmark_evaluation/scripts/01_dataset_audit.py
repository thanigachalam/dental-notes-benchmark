#!/usr/bin/env python3
"""Read-only audit: IDs, splits, empty values, domains, code coverage, provenance."""
import argparse,collections,sys
from pathlib import Path
from common import records,write_json,write_csv,get_labels,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',default='dental_nlp_benchmark_evaluation/outputs/audit');a=p.parse_args();out=Path(a.out)
 counts=collections.Counter(); fields=collections.defaultdict(lambda:collections.Counter()); labels=collections.Counter(); procs=collections.Counter(); ids=set(); duplicate_ids=[]; prompts=collections.Counter(); specialties=collections.Counter(); status=collections.Counter(); splits=collections.Counter(); annotators=collections.Counter()
 for r in records(a.input):
  counts['rows']+=1;nid=r.get('note_id');
  if nid in ids:duplicate_ids.append(nid)
  ids.add(nid);splits[str(r.get('split','MISSING'))]+=1;specialties[str(r.get('specialty','MISSING'))]+=1;prompts[str(r.get('source_prompt_id','MISSING'))]+=1;status[str(r.get('llm_summary_validation_status','MISSING'))]+=1;annotators[str(r.get('annotator_type','MISSING'))]+=1
  labels.update(get_labels(r,'diagnosis'));procs.update(get_labels(r,'procedure'))
  for k,v in r.items():
   fields[k]['key_present']+=1
   if v is None:fields[k]['null']+=1
   elif v=='':fields[k]['empty_string']+=1
   elif isinstance(v,(list,dict)) and not v:fields[k]['empty_collection']+=1
   else:fields[k]['populated']+=1
  for k in ('allergies','sdoh_factors'):
   if k not in r:fields[k]['absent']+=1
 report={'dataset_rows':counts['rows'],'unique_note_ids':len(ids),'duplicate_note_ids':len(duplicate_ids),'split_counts':splits,'specialty_counts':specialties,'prompt_id_count':len(prompts),'prompt_counts':prompts,'diagnosis_unique':len(labels),'procedure_base_unique':len(procs),'llm_status':status,'annotator_metadata':annotators,'note':'Recorded annotator_type is metadata, not independent evidence of clinician review.'}
 write_json(out/'audit_report.json',report)
 write_csv(out/'field_states.csv',[{'field':k,'total_records':counts['rows'],'absent':counts['rows']-v['key_present'],'null':v['null'],'empty_string':v['empty_string'],'empty_collection':v['empty_collection'],'populated':v['populated']} for k,v in sorted(fields.items())])
 for name,c in [('specialties',specialties),('splits',splits),('diagnoses',labels),('procedures',procs),('source_prompts',prompts)]:write_csv(out/f'{name}.csv',[{'value':k,'count':v} for k,v in c.most_common()],['value','count'])
 write_csv(out/'duplicate_note_ids.csv',[{'note_id':x} for x in duplicate_ids],['note_id']);print('Audit complete:',counts['rows'],'rows,',len(ids),'unique IDs ->',out)
if __name__=='__main__':main()
