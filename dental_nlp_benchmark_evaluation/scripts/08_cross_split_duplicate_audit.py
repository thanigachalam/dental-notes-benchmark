#!/usr/bin/env python3
"""Join saved near-duplicate candidate pairs to split assignments; no note text required."""
import argparse,csv,json,collections
from pathlib import Path
from common import write_csv,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--duplicates-root',required=True,help='Path containing duplicates, duplicates_0.80 etc');p.add_argument('--splits',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 root=Path(a.duplicates_root);out=Path(a.out);manifests={}
 for filename in a.splits:
  with open(filename,newline='',encoding='utf8') as f: rows=list(csv.DictReader(f))
  if len({r['note_id'] for r in rows})!=len(rows):raise ValueError('Duplicate manifest IDs: '+filename)
  manifests[Path(filename).name]={r['note_id']:r['split'] for r in rows}
 paths=list(root.glob('duplicates*/**/*_near_duplicate_pairs.csv'))+list(root.glob('*_near_duplicate_pairs.csv'))
 paths=sorted(set(paths));details=[];summary=[]
 for file in paths:
  report=file.parent/'duplicate_report.json';threshold=None
  if report.exists():
   reports=json.loads(report.read_text(encoding='utf8'))
   if isinstance(reports,list):threshold=reports[0].get('threshold') if reports else None
  if threshold is None:
   if file.parent.name.startswith('duplicates_'):threshold=file.parent.name.split('duplicates_',1)[1]
   elif file.parent.name=='duplicates':threshold='0.85'
  field=file.name.replace('_near_duplicate_pairs.csv','')
  with open(file,newline='',encoding='utf8') as f: pairs=list(csv.DictReader(f))
  for name,assign in manifests.items():
   c=collections.Counter()
   for pair in pairs:
    id1,id2=pair['note_id_1'],pair['note_id_2']
    if id1 not in assign or id2 not in assign:raise ValueError(f'{file}: pair ID missing from {name}')
    s1,s2=assign[id1],assign[id2];bucket='cross_split' if s1!=s2 else f'within_{s1}';c[bucket]+=1
    details.append({'threshold':threshold,'field':field,'manifest':name,'note_id_1':id1,'note_id_2':id2,'similarity':pair.get('similarity',''),'split_1':s1,'split_2':s2,'pair_category':bucket})
   summary.append({'threshold':threshold,'field':field,'manifest':name,'verified_candidate_pairs':len(pairs),'cross_split_pairs':c['cross_split'],'within_train':c['within_train'],'within_validation':c['within_validation'],'within_test':c['within_test']})
 if not paths:raise ValueError('No duplicate pair files found. Pass --duplicates-root dental_nlp_benchmark_evaluation/outputs')
 summary.sort(key=lambda x:(float(x['threshold']),x['field'],x['manifest']))
 write_csv(out/'cross_split_duplicate_summary.csv',summary);write_csv(out/'cross_split_duplicate_details.csv',details,['threshold','field','manifest','note_id_1','note_id_2','similarity','split_1','split_2','pair_category'])
 write_json(out/'cross_split_duplicate_report.json',{'pair_files':[str(x) for x in paths],'summary':summary,'caution':'Results are limited to candidate pairs retrieved using the original approximate MinHash-LSH algorithm; not exhaustive.'})
 print('Cross-split duplicate audit:',len(summary),'summary rows,',len(details),'pair-to-manifest assignments')
if __name__=='__main__':main()
