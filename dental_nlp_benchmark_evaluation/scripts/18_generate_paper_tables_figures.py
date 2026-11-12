#!/usr/bin/env python3
"""Generate editorial DRAFT tables/figures from reported benchmark scores.
Replace scores CSV with confirmed artifact-based values before manuscript submission.
"""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REQUIRED=('task','split','model','test_macro_f1','evidence_status')
MODELS=['Rule-based','TF-IDF','ClinicalBERT']
TASKS=['Diagnosis','Procedure']
SPLITS=['Published','Prompt-ID-disjoint']

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--scores',required=True)
    pa.add_argument('--out',required=True)
    pa.add_argument('--family-matrix',default='')
    a=pa.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with open(a.scores,newline='',encoding='utf-8-sig') as f:
        r=csv.DictReader(f)
        if not set(REQUIRED).issubset(r.fieldnames):raise ValueError(f'Required columns {REQUIRED}; found {r.fieldnames}')
        rows=list(r)
    key={};statuses=set()
    for row in rows:
        k=(row['task'],row['split'],row['model'])
        if k in key:raise ValueError(f'Duplicate row {k}')
        score=float(row['test_macro_f1'])
        if not 0<=score<=1:raise ValueError(f'Out of range score {k}: {score}')
        key[k]=score;statuses.add(row['evidence_status'])
    if len(key)!=12 or any((t,s,m) not in key for t in TASKS for s in SPLITS for m in MODELS):raise ValueError('Expected all 12 task/split/model combinations')
    with open(out/'table_benchmark_macro_f1.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['Task','Split','Rule-based','TF-IDF','ClinicalBERT'])
        for task in TASKS:
            for split in SPLITS:
                w.writerow([task,split]+[f'{key[(task,split,m)]:.5f}' for m in MODELS])
    with open(out/'table_split_difference_pp.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['Task','Model','Published macro-F1','Prompt-ID-disjoint macro-F1','Decline (percentage points)'])
        for task in TASKS:
            for model in MODELS:
                a1=key[(task,'Published',model)];a2=key[(task,'Prompt-ID-disjoint',model)]
                w.writerow([task,model,f'{a1:.5f}',f'{a2:.5f}',f'{(a1-a2)*100:.3f}'])
    # Grouped bar chart, consistent with common manuscript figure styles.
    labels=[f'{t}\n{s}' for t in TASKS for s in SPLITS]
    x=np.arange(4);width=.25
    fig,ax=plt.subplots(figsize=(10.2,5.0),layout='constrained')
    for i,model in enumerate(MODELS):
        vals=[key[(t,s,model)] for t in TASKS for s in SPLITS]
        ax.bar(x+(i-1)*width,vals,width,label=model)
    ax.set_ylabel('Test macro-F1');ax.set_ylim(0,1.08)
    ax.set_xticks(x,labels);ax.legend(loc='lower left',ncol=3,fontsize=9)
    ax.grid(axis='y',alpha=.18)
    fig.savefig(out/'figure_benchmark_macro_f1.png',dpi=300,bbox_inches='tight')
    fig.savefig(out/'figure_benchmark_macro_f1.svg',bbox_inches='tight')
    plt.close(fig)
    # Paired split degradation by model (percentage point changes).
    fig,ax=plt.subplots(figsize=(8.6,4.6),layout='constrained')
    xx=np.arange(len(MODELS));width=.35
    for j,task in enumerate(TASKS):
        changes=[(key[(task,'Published',m)]-key[(task,'Prompt-ID-disjoint',m)])*100 for m in MODELS]
        ax.bar(xx+(j-.5)*width,changes,width,label=task)
    ax.set_xticks(xx,MODELS);ax.set_ylabel('Macro-F1 decline (percentage points)');ax.legend()
    ax.grid(axis='y',alpha=.18)
    fig.savefig(out/'figure_split_degradation.png',dpi=300,bbox_inches='tight')
    fig.savefig(out/'figure_split_degradation.svg',bbox_inches='tight');plt.close(fig)
    if a.family_matrix:
        with open(a.family_matrix,newline='',encoding='utf-8-sig') as f:
            frows=list(csv.DictReader(f))
        subset=[r for r in frows if r.get('split_strategy',r.get('strategy')) in ('prompt_disjoint','prompt-disjoint')]
        if subset:
            subset.sort(key=lambda r:r.get('scenario_family',r.get('family','')))
            names=[r.get('scenario_family',r.get('family')) for r in subset]
            vals=np.array([[int(r[p]) for p in ['train','validation','test']] for r in subset])
            fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
            left=np.zeros(len(names))
            for idx,part in enumerate(('train','validation','test')):
                ax.barh(names,vals[:,idx],left=left,label=part.capitalize());left+=vals[:,idx]
            ax.set_xlabel('Records');ax.set_ylabel('Inferred scenario family');ax.legend()
            fig.savefig(out/'figure_scenario_family_split.png',dpi=300,bbox_inches='tight')
            fig.savefig(out/'figure_scenario_family_split.svg',bbox_inches='tight');plt.close(fig)
    status='; '.join(sorted(statuses))
    note=(
        'DRAFT FIGURES AND TABLES\n'
        'Source: ' + str(a.scores) + '\n'
        'Evidence status: ' + status + '\n'
        'IMPORTANT: Verify every metric against final machine-readable baseline and ClinicalBERT results before submitting.\n'
        'The prompt-ID-disjoint split is NOT a verified independent-template or scenario-disjoint split.\n'
        'Perfect performance on synthetic notes does NOT establish clinical generalization.\n'
        'Recommended manuscript caption (pending final audit): Test macro-F1 for rule-based, TF-IDF, and Bio_ClinicalBERT classifiers under the original record-level and source-prompt-ID-disjoint partition strategies.\n'
        'No empirical confidence intervals were estimated in these files.\n'
    )
    (out/'FIGURE_TABLE_README.txt').write_text(note)
    print('Generated draft tables/figures under',out)
    print(note)
if __name__=='__main__':main()
