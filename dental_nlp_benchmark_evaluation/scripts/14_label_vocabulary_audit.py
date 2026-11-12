#!/usr/bin/env python3
"""Read-only label coverage audit for the frozen SDCN-DNLP v21 split manifests.

Produces one summary per task/split and one row per held-out label. Never writes
back to the dataset. Uses common.get_labels, same extraction as scripts 04/10.
"""
import argparse,csv,json
from collections import Counter
from pathlib import Path
from common import records,get_labels

def export_csv(path,rows,fields):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',required=True,help='Frozen DataPort JSONL or ZIP with one JSONL')
 p.add_argument('--splits',required=True,nargs='+',help='Published/prompt-disjoint split CSVs')
 p.add_argument('--out',required=True,help='Output directory for audit reports')
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 source=list(records(a.input));byid={r['note_id']:r for r in source}
 if len(byid)!=len(source):raise SystemExit('Duplicate note_id in frozen dataset')
 summaries=[]; details=[]
 for manifest in a.splits:
  with open(manifest,encoding='utf-8',newline='') as f:assign=list(csv.DictReader(f))
  ids=[r['note_id'] for r in assign]
  if len(assign)!=len(source) or len(set(ids))!=len(source) or set(ids)!=set(byid):
   raise SystemExit(f'{manifest}: missing, duplicated or unknown IDs')
  partitions={s:[byid[r['note_id']] for r in assign if r['split']==s] for s in ('train','validation','test')}
  if sum(map(len,partitions.values()))!=len(source):raise SystemExit(f'{manifest}: unexpected partition name')
  for task in ('diagnosis','procedure'):
   labelsets={s:set(label for r in group for label in get_labels(r,task)) for s,group in partitions.items()}
   train=labelsets['train']
   for name,group in partitions.items():
    freq=Counter(label for r in group for label in get_labels(r,task))
    unseen=set(freq)-train if name!='train' else set()
    records_with_unseen=sum(bool(set(get_labels(r,task))-train) for r in group) if name!='train' else 0
    assignments=sum(freq[x] for x in unseen)
    summaries.append(dict(manifest=Path(manifest).name,task=task,partition=name,n_records=len(group),n_unique_labels=len(freq),n_train_labels=len(train),n_unseen_labels=len(unseen),records_with_unseen_labels=records_with_unseen,unseen_label_assignments=assignments,unseen_codes=';'.join(sorted(unseen))))
    for code in sorted(freq):
     details.append(dict(manifest=Path(manifest).name,task=task,partition=name,label=code,support=freq[code],seen_in_train=code in train))
   print(f'{Path(manifest).name} / {task}: training={len(train)} labels, validation unseen={len(labelsets["validation"]-train)}, test unseen={len(labelsets["test"]-train)}')
 export_csv(out/'label_coverage_summary.csv',summaries,['manifest','task','partition','n_records','n_unique_labels','n_train_labels','n_unseen_labels','records_with_unseen_labels','unseen_label_assignments','unseen_codes'])
 export_csv(out/'label_frequency_by_partition.csv',details,['manifest','task','partition','label','support','seen_in_train'])
 (out/'label_coverage_report.json').write_text(json.dumps({'dataset_records':len(source),'label_source':'get_labels() shared by baseline scripts 04 and 10','results':summaries,'interpretation':'Unseen labels cannot be predicted by a train-only model head and must count as false negatives in held-out evaluation. This is not evidence of an implementation failure.'},indent=2)+'\n',encoding='utf-8')
 print('Completed read-only audit:',out)
if __name__=='__main__':main()
