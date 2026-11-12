#!/usr/bin/env python3
"""Leakage-aware diagnosis/procedure multilabel baselines; ClinicalBERT optional separately."""
import argparse,csv,json,collections,random,re
from pathlib import Path
from common import records,get_labels,write_csv,write_json

def evaluate(ytrue,ypred,labels):
 from sklearn.preprocessing import MultiLabelBinarizer
 from sklearn.metrics import precision_recall_fscore_support
 mlb=MultiLabelBinarizer(classes=sorted(labels));mlb.fit([[]]);y=mlb.transform(ytrue);p=mlb.transform(ypred)
 out={}
 for avg in ['micro','macro','weighted','samples']:
  pr,re_,f1,_=precision_recall_fscore_support(y,p,average=avg,zero_division=0)
  out.update({f'{avg}_precision':round(float(pr),5),f'{avg}_recall':round(float(re_),5),f'{avg}_f1':round(float(f1),5)})
 return out

def rule_predict(train, test, task):
 # Build rule dictionary exclusively from training set, using label descriptions; no evaluation-set label access.
 key='diagnoses' if task=='diagnosis' else 'procedures';code='diagnosis_code' if task=='diagnosis' else 'procedure_code_base'; term='diagnosis_text' if task=='diagnosis' else 'procedure_text'
 lex=collections.defaultdict(collections.Counter)
 for r in train:
  for d in r.get(key,[]):
   if isinstance(d,dict) and d.get(code):
    txt=re.split(r'[,;:]|\btooth/site\b',str(d.get(term,'')),maxsplit=1)[0].strip().lower()
    if len(txt)>=7:lex[d[code]][txt]+=1
 phrases={label:[t for t,_ in cnt.most_common(5)] for label,cnt in lex.items()}
 preds=[]
 for r in test:
  text=str(r.get('input_note','')).lower();preds.append([label for label,patterns in phrases.items() if any(t in text for t in patterns)])
 return preds

def run(a):
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.multiclass import OneVsRestClassifier
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import MultiLabelBinarizer
 from sklearn.pipeline import make_pipeline
 data=list(records(a.input));byid={r['note_id']:r for r in data};out=Path(a.out);results=[]
 for name in a.splits:
  with open(name,newline='',encoding='utf-8') as f:assign={r['note_id']:r['split'] for r in csv.DictReader(f)}
  if set(assign)!=set(byid):raise ValueError(f'{name}: ID mismatch with frozen dataset')
  train=[byid[n] for n,s in assign.items() if s=='train'];val=[byid[n] for n,s in assign.items() if s=='validation'];test=[byid[n] for n,s in assign.items() if s=='test']
  if not train or not test:raise ValueError(f'{name}: empty train/test group')
  for task in a.tasks:
   true=[get_labels(r,task) for r in test];labelset=sorted(set(x for r in train for x in get_labels(r,task))|set(x for ys in true for x in ys))
   for modelname in a.models:
    if modelname=='rule':pred=rule_predict(train,test,task)
    elif modelname=='tfidf':
     mlb=MultiLabelBinarizer(classes=labelset);mlb.fit([[]]);ytr=mlb.transform([get_labels(r,task) for r in train]);
     pipeline=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=a.max_features,sublinear_tf=True),OneVsRestClassifier(LogisticRegression(max_iter=500,random_state=a.seed)))
     pipeline.fit([r['input_note'] for r in train],ytr)
     pred=mlb.inverse_transform(pipeline.predict([r['input_note'] for r in test]));pred=[list(x) for x in pred]
    metrics=evaluate(true,pred,labelset)
    row={'split_manifest':Path(name).name,'task':task,'model':modelname,'n_train':len(train),'n_validation':len(val),'n_test':len(test),'labels_train':len(set(x for r in train for x in get_labels(r,task))),'unseen_test_labels':len(set(x for ys in true for x in ys)-set(x for r in train for x in get_labels(r,task))),**metrics}
    results.append(row);print(row['split_manifest'],task,modelname,metrics['macro_f1'])
 write_csv(out/'baseline_results.csv',results)
 write_json(out/'baseline_config.json',vars(a))

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--splits',nargs='+',required=True);p.add_argument('--out',default='dental_nlp_benchmark_evaluation/outputs/benchmarks');p.add_argument('--tasks',nargs='+',choices=['diagnosis','procedure'],default=['diagnosis','procedure']);p.add_argument('--models',nargs='+',choices=['rule','tfidf'],default=['rule','tfidf']);p.add_argument('--max-features',type=int,default=30000);p.add_argument('--seed',type=int,default=20260313);a=p.parse_args();run(a)
if __name__=='__main__':main()
