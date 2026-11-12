#!/usr/bin/env python3
"""Create external ID-only partitions; fail rather than invent template provenance."""
import argparse,collections,random,csv
from pathlib import Path
from common import records,write_json,write_csv

def allocate(groups, seed):
 rng=random.Random(seed); order=list(groups);rng.shuffle(order);order.sort(key=lambda g:-len(groups[g])); totals=collections.Counter(); assignment={}
 total=sum(map(len,groups.values()));targets={'train':.7*total,'validation':.15*total,'test':.15*total}
 for g in order:
  choice=min(targets,key=lambda k:(totals[k]+len(groups[g])-targets[k])**2 - (totals[k]-targets[k])**2)
  # Prevent greedy overpacking by preferring greatest remaining target capacity.
  choice=max(targets,key=lambda k:(targets[k]-totals[k]) / targets[k])
  for id in groups[g]: assignment[id]=choice
  totals[choice]+=len(groups[g])
 return assignment

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',default='dental_nlp_benchmark_evaluation/outputs/splits');p.add_argument('--seed',type=int,default=20260313);p.add_argument('--template-map',help='CSV: source_prompt_id,template_family_id; only use verified mapping');a=p.parse_args()
 out=Path(a.out);data=list(records(a.input)); ids=[r['note_id'] for r in data]; assert len(ids)==len(set(ids)),'Duplicate note IDs'
 published={r['note_id']:r['split'] for r in data};assert set(published.values()) <= {'train','validation','test'}
 write_csv(out/'published_70_15_15.csv',[{'note_id':nid,'split':published[nid]} for nid in sorted(ids)])
 groups=collections.defaultdict(list)
 for r in data:
  if not r.get('source_prompt_id'):raise ValueError('Missing source_prompt_id')
  groups[r['source_prompt_id']].append(r['note_id'])
 prompt=allocate(groups,a.seed);write_csv(out/'prompt_disjoint.csv',[{'note_id':x,'split':prompt[x]} for x in sorted(ids)])
 summary={'seed':a.seed,'published':collections.Counter(published.values()),'prompt_disjoint':collections.Counter(prompt.values()),'prompt_groups':len(groups),'prompt_overlap':0,'template_status':'not_generated_without_verified_template_mapping'}
 if a.template_map:
  with open(a.template_map,newline='',encoding='utf-8') as f: mapping={r['source_prompt_id']:r['template_family_id'] for r in csv.DictReader(f)}
  if any(not mapping.get(g) for g in groups):raise ValueError('Incomplete template mapping: refusing to invent family IDs')
  families=collections.defaultdict(list)
  for promptid,members in groups.items():families[mapping[promptid]].extend(members)
  family_split=allocate(families,a.seed);write_csv(out/'template_disjoint.csv',[{'note_id':x,'split':family_split[x]} for x in sorted(ids)])
  summary.update(template_status='verified_mapping_provided',template_families=len(families),template_disjoint=collections.Counter(family_split.values()))
 write_json(out/'split_report.json',summary)
 print('Wrote original/prompt splits. Template:',summary['template_status']);print('Counts:',dict(summary['prompt_disjoint']))
if __name__=='__main__':main()
