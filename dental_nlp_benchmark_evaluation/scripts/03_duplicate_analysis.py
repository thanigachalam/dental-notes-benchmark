#!/usr/bin/env python3
"""Exact duplicates plus approximate MinHash-LSH candidate pairs and verified Jaccard clusters."""
import argparse,collections,hashlib,re
from pathlib import Path
from common import records,write_csv,write_json

def shingles(s,k=4):
 words=re.findall(r'[a-z0-9#]+',(s or '').lower());return {' '.join(words[i:i+k]) for i in range(max(0,len(words)-k+1))} or set(words)
class DSU:
 def __init__(self,n):self.p=list(range(n))
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,x,y):self.p[self.find(x)]=self.find(y)

def analyze(data,field,out,threshold,num_perm):
 from datasketch import MinHash,MinHashLSH
 index=MinHashLSH(threshold=max(.1,threshold-.08),num_perm=num_perm)
 signatures=[]; sets=[]; candidates=set(); exact=collections.defaultdict(list)
 for i,r in enumerate(data):
  value=r.get(field) or '';exact[hashlib.sha256(' '.join(value.lower().split()).encode()).hexdigest()].append(i)
  tokens=shingles(value);sets.append(tokens);mh=MinHash(num_perm=num_perm)
  for token in tokens:mh.update(token.encode())
  for j in index.query(mh):candidates.add((int(j),i))
  index.insert(str(i),mh);signatures.append(mh)
 dsu=DSU(len(data));matches=[]
 for j,i in sorted(candidates):
  s1,s2=sets[j],sets[i];sim=len(s1&s2)/len(s1|s2) if s1|s2 else 1.
  if sim>=threshold:
   dsu.union(i,j);matches.append({'note_id_1':data[j]['note_id'],'note_id_2':data[i]['note_id'],'similarity':round(sim,5),'split_1':data[j].get('split'),'split_2':data[i].get('split'),'cross_split':data[j].get('split')!=data[i].get('split')})
 clusters=collections.defaultdict(list)
 for i,r in enumerate(data):clusters[dsu.find(i)].append(r['note_id'])
 write_csv(out/f'{field}_near_duplicate_pairs.csv',matches,['note_id_1','note_id_2','similarity','split_1','split_2','cross_split'])
 write_csv(out/f'{field}_clusters.csv',[{'cluster_id':j,'size':len(m),'note_ids':'|'.join(m)} for j,m in enumerate(clusters.values(),1) if len(m)>1],['cluster_id','size','note_ids'])
 exact_groups=[g for g in exact.values() if len(g)>1]
 return {'field':field,'threshold':threshold,'metric':'word_4gram_jaccard','candidate_method':'MinHashLSH','approximate_retrieval':True,'candidate_pairs':len(candidates),'near_duplicate_pairs':len(matches),'cross_original_split_pairs':sum(bool(x['cross_split']) for x in matches),'multi_record_clusters':sum(len(x)>1 for x in clusters.values()),'largest_cluster':max(map(len,clusters.values()),default=0),'exact_duplicate_groups':len(exact_groups),'exact_duplicate_excess_records':sum(len(x)-1 for x in exact_groups),'caveat':'LSH approximate retrieval may miss pairs; not an exhaustive all-pairs certificate.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',default='dental_nlp_benchmark_evaluation/outputs/duplicates');p.add_argument('--threshold',type=float,default=.85);p.add_argument('--num-perm',type=int,default=128);p.add_argument('--fields',nargs='+',default=['input_note','llm_summary_note']);a=p.parse_args()
 data=list(records(a.input));out=Path(a.out);report=[analyze(data,f,out,a.threshold,a.num_perm) for f in a.fields];write_json(out/'duplicate_report.json',report)
 for r in report:print(r['field'],r['near_duplicate_pairs'],'verified candidate pairs;',r['multi_record_clusters'],'clusters')
if __name__=='__main__':main()
