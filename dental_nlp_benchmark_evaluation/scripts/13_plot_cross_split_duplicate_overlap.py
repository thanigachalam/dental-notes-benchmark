#!/usr/bin/env python3
"""Read the output of 08_cross_split_duplicate_audit.py and export a paper figure/table.
Does not access or modify the underlying SDCN-DNLP clinical dataset.
"""
import argparse, csv, pathlib

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='cross_split_duplicate_summary.csv')
    p.add_argument('--out',default='dental_nlp_benchmark_evaluation/paper_outputs/generated/cross_split_duplicates')
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=pathlib.Path(a.out);output.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(open(a.input,encoding='utf-8-sig',newline='')))
    needed={'threshold','field','manifest','verified_candidate_pairs','cross_split_pairs'}
    if not rows or not needed.issubset(rows[0]):raise ValueError('Missing expected CSV columns: '+str(needed-set(rows[0] if rows else [])))
    results={}
    for r in rows:
        if r['field']!='input_note':continue
        t=float(r['threshold']); m=r['manifest']
        results[(t,m)]={k:int(r[k]) for k in ('verified_candidate_pairs','cross_split_pairs','within_train','within_validation','within_test')}
    thresholds=sorted(set(t for t,_ in results))
    names=('published_70_15_15.csv','prompt_disjoint.csv')
    if not all((t,n) in results for t in thresholds for n in names):raise ValueError('Both split manifests required at every threshold')
    data=[]
    for t in thresholds:
        pub=results[(t,names[0])]; prm=results[(t,names[1])]
        if pub['verified_candidate_pairs']!=prm['verified_candidate_pairs']:raise ValueError('Pair totals differ across split manifests')
        data.append({'Threshold':f'{t:.2f}','Detected input-note pairs':pub['verified_candidate_pairs'], 'Published cross-split':pub['cross_split_pairs'], 'Prompt-disjoint cross-split':prm['cross_split_pairs']})
    columns=list(data[0]); csvfile=output/'table_cross_split_duplicate_overlap.csv'
    with open(csvfile,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(data)
    with open(output/'table_cross_split_duplicate_overlap.md','w',encoding='utf-8') as f:
        f.write('| '+' | '.join(columns)+' |\n| '+' | '.join(['---']+['---:']*(len(columns)-1))+' |\n')
        for d in data:f.write('| '+' | '.join(str(d[c]) for c in columns)+' |\n')
    xs=list(range(len(data)));width=.34
    fig,ax=plt.subplots(figsize=(7.4,4.4))
    pubs=[d['Published cross-split'] for d in data];prms=[d['Prompt-disjoint cross-split'] for d in data]
    b1=ax.bar([x-width/2 for x in xs],pubs,width,label='Published 70/15/15')
    b2=ax.bar([x+width/2 for x in xs],prms,width,label='Prompt-disjoint')
    ax.bar_label(b1,padding=3,fontsize=9);ax.bar_label(b2,padding=3,fontsize=9)
    ax.set_xticks(xs,[d['Threshold'] for d in data]);ax.set_xlabel('Word 4-gram Jaccard verification threshold')
    ax.set_ylabel('Detected cross-partition input-note pairs')
    ax.set_ylim(0,max(pubs+prms)+3);ax.legend(frameon=False,ncol=2,loc='upper right')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(output/'figure_cross_split_near_duplicate_pairs.png',dpi=300,bbox_inches='tight')
    fig.savefig(output/'figure_cross_split_near_duplicate_pairs.pdf',bbox_inches='tight')
    plt.close(fig)
    print(f'Generated cross-split overlap figure and table for {len(thresholds)} thresholds in {output}')
    print('Reminder: counts are approximate-retrieval-conditional, not exhaustive all-pairs evidence.')
if __name__=='__main__':main()
