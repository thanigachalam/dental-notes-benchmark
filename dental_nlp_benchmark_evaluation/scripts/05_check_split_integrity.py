#!/usr/bin/env python3
import argparse,csv,collections
from pathlib import Path
from common import records,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--splits',nargs='+',required=True);p.add_argument('--out',default='dental_nlp_benchmark_evaluation/outputs/split_integrity.json');p.add_argument('--template-map');a=p.parse_args()
 data=list(records(a.input));ref={r['note_id']:r for r in data};report={}
 for path in a.splits:
  with open(path,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
  ids=[r['note_id'] for r in rows];assign={r['note_id']:r['split'] for r in rows}
  item={'row_count':len(rows),'missing_ids':len(set(ref)-set(ids)),'unknown_ids':len(set(ids)-set(ref)),'duplicate_ids':len(ids)-len(set(ids)),'split_counts':dict(collections.Counter(assign.values()))}
  prompt=collections.defaultdict(set)
  for nid,split in assign.items():
   if nid in ref:prompt[ref[nid].get('source_prompt_id')].add(split)
  item['prompt_groups_crossing_splits']=sum(len(v)>1 for v in prompt.values())
  item['valid']=not (item['missing_ids'] or item['unknown_ids'] or item['duplicate_ids'])
  report[Path(path).name]=item
 write_json(a.out,report);print(report)
if __name__=='__main__':main()
