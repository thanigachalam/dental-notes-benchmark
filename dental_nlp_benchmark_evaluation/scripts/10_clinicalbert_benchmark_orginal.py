#!/usr/bin/env python3
"""ClinicalBERT multilabel evaluation. Read-only dataset, one task+split per run.
Install extra requirements; pretrained weights must be downloaded unless cached.
"""
import argparse,csv,json,random,time,sys,platform
from pathlib import Path
import numpy as np
from common import records,get_labels,write_csv,write_json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',required=True);p.add_argument('--split',required=True)
 p.add_argument('--task',required=True,choices=['diagnosis','procedure']);p.add_argument('--out',required=True)
 p.add_argument('--model',default='emilyalsentzer/Bio_ClinicalBERT');p.add_argument('--seed',type=int,default=20260313)
 p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=8)
 p.add_argument('--eval-batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=2e-5)
 p.add_argument('--max-length',type=int,default=256);p.add_argument('--weight-decay',type=float,default=0.01)
 p.add_argument('--thresholds',type=float,nargs='+',default=[0.3,0.4,0.5,0.6,0.7])
 p.add_argument('--max-train',type=int,default=0,help='Smoke test only; nonzero results not publication-ready')
 p.add_argument('--max-validation',type=int,default=0);p.add_argument('--max-test',type=int,default=0)
 p.add_argument('--local-files-only',action='store_true',help='Require pretrained model to be cached locally')
 a=p.parse_args()
 try:
  import torch,transformers
  from torch.utils.data import Dataset,DataLoader
  from transformers import AutoTokenizer,AutoModelForSequenceClassification, get_linear_schedule_with_warmup
  from sklearn.preprocessing import MultiLabelBinarizer
  from sklearn.metrics import precision_recall_fscore_support
 except ImportError as e:raise SystemExit('Install: python -m pip install -r dental_nlp_benchmark_evaluation/requirements_clinicalbert.txt\n'+str(e))
 if a.epochs<1 or a.batch_size<1 or a.max_length<16:raise SystemExit('Invalid hyperparameters')
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 if hasattr(torch.backends,'mps') and torch.backends.mps.is_available():device=torch.device('mps')
 elif torch.cuda.is_available():device=torch.device('cuda')
 else:device=torch.device('cpu')
 source=list(records(a.input));byid={r['note_id']:r for r in source}
 with open(a.split,newline='',encoding='utf8') as f:
  reader=list(csv.DictReader(f))
 if len(reader)!=len(source) or len(set(r['note_id'] for r in reader))!=len(source) or set(r['note_id'] for r in reader)!=set(byid):
  raise SystemExit('Split manifest does not contain each frozen dataset ID exactly once')
 groups={s:[byid[r['note_id']] for r in reader if r['split']==s] for s in ['train','validation','test']}
 for s,cap in [('train',a.max_train),('validation',a.max_validation),('test',a.max_test)]:
  if cap:groups[s]=groups[s][:cap]
 if any(not x for x in groups.values()):raise SystemExit('All three partitions must contain records')
 labels=sorted({lab for r in groups['train'] for lab in get_labels(r,a.task)})
 if not labels:raise SystemExit('No training labels')
 unseen=sorted({lab for s in ['validation','test'] for r in groups[s] for lab in get_labels(r,a.task)}-set(labels))
 # Train-only output dimension; unseen labels are reported as false negatives, not silently excluded.
 mlb=MultiLabelBinarizer(classes=labels);mlb.fit([[]]);tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=a.local_files_only)
 class Notes(Dataset):
  def __init__(self,items):self.items=items;self.Y=mlb.transform([get_labels(x,a.task) for x in items]).astype('float32')
  def __len__(self):return len(self.items)
  def __getitem__(self,i):return i
 def loader(items,batch,shuffle):
  obj=Notes(items)
  def collate(ids):
   texts=[str(obj.items[j].get('input_note','')) for j in ids]
   enc=tokenizer(texts,padding=True,truncation=True,max_length=a.max_length,return_tensors='pt')
   enc['labels']=torch.tensor(obj.Y[ids],dtype=torch.float32);return enc
  return DataLoader(obj,batch_size=batch,shuffle=shuffle,collate_fn=collate)
 train_loader=loader(groups['train'],a.batch_size,True)
 val_loader=loader(groups['validation'],a.eval_batch_size,False)
 test_loader=loader(groups['test'],a.eval_batch_size,False)
 model = AutoModelForSequenceClassification.from_pretrained(
      a.model,
      num_labels=len(labels),
      problem_type="multi_label_classification",
      ignore_mismatched_sizes=True,
      use_safetensors=True,
      local_files_only=a.local_files_only
  ).to(device)
 optim=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
 steps=a.epochs*len(train_loader);scheduler=get_linear_schedule_with_warmup(optim,num_warmup_steps=max(1,steps//10),num_training_steps=steps)
 def predict(ld):
  model.eval();allp=[];ally=[]
  with torch.no_grad():
   for batch in ld:
    y=batch.pop('labels');batch={k:v.to(device) for k,v in batch.items()}
    logits=model(**batch).logits;allp.append(torch.sigmoid(logits).cpu().numpy());ally.append(y.numpy())
  return np.concatenate(allp),np.concatenate(ally)
 def metric(y,pred):
  res={}
  for avg in ('micro','macro','weighted','samples'):
   pr,rec,f,_=precision_recall_fscore_support(y,pred,average=avg,zero_division=0)
   res.update({avg+'_precision':float(pr),avg+'_recall':float(rec),avg+'_f1':float(f)})
  return res
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 audit={'model':a.model,'task':a.task,'split_manifest':Path(a.split).name,'seed':a.seed,'device':str(device),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'n_train':len(groups['train']),'n_validation':len(groups['validation']),'n_test':len(groups['test']),'label_count_train':len(labels),'unseen_eval_labels':unseen,'hyperparameters':{k:getattr(a,k) for k in ['epochs','batch_size','eval_batch_size','lr','max_length','weight_decay','thresholds']},'smoke_test':any((a.max_train,a.max_validation,a.max_test)),'truncation_warning':'Texts longer than max_length tokenizer tokens are truncated; assess truncation coverage before publication.'}
 write_json(out/'clinicalbert_run_config.json',audit);(out/'label_vocabulary.json').write_text(json.dumps(labels,indent=2)+'\n')
 best_score=-1;best_state=None;best_threshold=None;history=[]
 for epoch in range(1,a.epochs+1):
  model.train();total=0.0
  for batch in train_loader:
   batch={k:v.to(device) for k,v in batch.items()};optim.zero_grad(set_to_none=True)
   loss=model(**batch).loss;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step();scheduler.step();total+=loss.item()
  vprobs,vy=predict(val_loader)
  choices=[(metric(vy,(vprobs>=t).astype(int))['micro_f1'],t) for t in a.thresholds]
  score,t=max(choices,key=lambda q:(q[0],-abs(q[1]-0.5)))
  history.append({'epoch':epoch,'avg_train_loss':total/len(train_loader),'validation_micro_f1':score,'validation_threshold':t})
  print(f'epoch={epoch} train_loss={total/len(train_loader):.4f} validation_micro_f1={score:.5f} threshold={t}',flush=True)
  if score>best_score:
   best_score=score;best_threshold=t;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 write_csv(out/'training_history.csv',history)
 model.load_state_dict(best_state);test_probs,yt=predict(test_loader)
 pred=(test_probs>=best_threshold).astype(int)
 scores=metric(yt,pred)
 # Evaluate over ALL observed test labels: unseen labels contribute zero true positives and FNs.
 test_truth=[get_labels(r,a.task) for r in groups['test']]
 all_eval_labels=sorted(set(labels)|{c for ys in test_truth for c in ys})
 if len(all_eval_labels)>len(labels):
  eval_mlb=MultiLabelBinarizer(classes=all_eval_labels);eval_mlb.fit([[]]);yfull=eval_mlb.transform(test_truth)
  pfull=np.zeros_like(yfull)
  loc={c:i for i,c in enumerate(all_eval_labels)}
  for i,c in enumerate(labels):pfull[:,loc[c]]=pred[:,i]
  scores=metric(yfull,pfull)
 else:yfull=yt;pfull=pred
 per=[]
 for j,c in enumerate(all_eval_labels):
  pr,rec,f,support=precision_recall_fscore_support(yfull[:,j],pfull[:,j],average='binary',zero_division=0)
  per.append({'label':c,'test_support':int(yfull[:,j].sum()),'train_seen':int(c in labels),'precision':float(pr),'recall':float(rec),'f1':float(f)})
 write_csv(out/'clinicalbert_per_label_metrics.csv',per)
 result={'split_manifest':Path(a.split).name,'task':a.task,'model':'ClinicalBERT','n_train':len(groups['train']),'n_validation':len(groups['validation']),'n_test':len(groups['test']),'labels_train':len(labels),'unseen_test_labels':len(set(c for ys in test_truth for c in ys)-set(labels)),'selected_threshold':best_threshold,'best_validation_micro_f1':best_score,'smoke_test':audit['smoke_test'],**scores}
 write_csv(out/'clinicalbert_results.csv',[result]);write_json(out/'clinicalbert_results.json',result)
 print('COMPLETE:',out,'test_macro_f1=',round(scores['macro_f1'],5),'smoke_test=',audit['smoke_test'])
if __name__=='__main__':main()
