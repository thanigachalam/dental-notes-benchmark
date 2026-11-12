#!/usr/bin/env python3
"""Create paper figures and tables from existing SDCN-DNLP v21 ANALYSIS OUTPUTS.

Read-only with respect to dataset and experimental outputs. This script does not
load the underlying dental-note JSONL or alter source files. No invented results.

Usage:
  python dental_nlp_benchmark_evaluation/scripts/06_generate_paper_figures_tables.py \
      --outputs dental_nlp_benchmark_evaluation/outputs \
      --paper-out dental_nlp_benchmark_evaluation/paper_outputs/generated

Requires: matplotlib. Uses only stdlib otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_json(path):
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def load_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open('r', newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def to_num(value, fallback=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return fallback


def save_table(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    md = path.with_suffix('.md')
    with md.open('w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(columns) + ' |\n')
        f.write('| ' + ' | '.join('---' for _ in columns) + ' |\n')
        for row in rows:
            f.write('| ' + ' | '.join(str(row.get(c, '')).replace('|', '\\|').replace('\n', ' ') for c in columns) + ' |\n')
    return path.name


def save_figure(fig, path, dpi=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path.name


def tidy(ax, ylabel=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.16)
    ax.set_axisbelow(True)


def get_actual_manifest(key):
    return {'published_70_15_15.csv': 'Published 70/15/15',
            'prompt_disjoint.csv': 'Prompt-disjoint',
            'template_disjoint.csv': 'Template-disjoint'}.get(key, key)


def build_baselines(base, tables, figs, inventory):
    rows = load_csv(base / 'benchmarks' / 'baseline_results.csv')
    if rows is None:
        inventory['missing'].append('benchmarks/baseline_results.csv')
        return
    if not rows:
        inventory['warnings'].append('Baseline CSV has no rows')
        return
    columns = ['split_manifest', 'task', 'model', 'n_train', 'n_validation', 'n_test',
               'labels_train', 'unseen_test_labels', 'micro_precision', 'micro_recall', 'micro_f1',
               'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1', 'samples_f1']
    inventory['tables'].append(save_table(tables / 'table_baseline_metrics.csv', rows, columns))
    for r in rows:
        try:
            float(r['macro_f1'])
        except (KeyError, ValueError, TypeError):
            raise ValueError('Saved baseline data must include numeric macro_f1')
    tasks = sorted(set(r['task'] for r in rows))
    models = sorted(set(r['model'] for r in rows))
    split_names = [s for s in ('published_70_15_15.csv', 'prompt_disjoint.csv', 'template_disjoint.csv')
                   if s in {r['split_manifest'] for r in rows}]
    if not split_names:
        inventory['warnings'].append('No recognized manifests for baseline grouped chart')
        return
    keys = [(t, m) for t in tasks for m in models]
    bykey = {(r['split_manifest'], r['task'], r['model']): r for r in rows}
    x = list(range(len(keys)))
    fig, ax = plt.subplots(figsize=(max(8.2, len(keys)*1.8), 4.8))
    width = min(.75/len(split_names), .35)
    for s_i, split in enumerate(split_names):
        positions = [i + (s_i-(len(split_names)-1)/2)*width for i in x]
        ys = [float(bykey[(split, task, model)]['macro_f1']) if (split,task,model) in bykey else 0 for task,model in keys]
        bars = ax.bar(positions, ys, width=width*.95, label=get_actual_manifest(split))
        for bar, y, key in zip(bars, ys, keys):
            if (split, *key) in bykey:
                ax.text(bar.get_x()+bar.get_width()/2, y+.013, f'{y:.3f}', ha='center', fontsize=8, rotation=0)
    ax.set_xticks(x, [f'{t.title()}\n{m.upper() if m=="tfidf" else m.title()}' for t,m in keys])
    ax.set_ylim(0, 1.13)
    ax.set_title('Baseline performance by evaluation split')
    ax.legend(frameon=False, fontsize=9)
    tidy(ax, 'Test macro-F1')
    fig.tight_layout()
    inventory['figures'].append(save_figure(fig, figs / 'figure_baseline_macro_f1'))
    if len(split_names)>1 and 'published_70_15_15.csv' in split_names and 'prompt_disjoint.csv' in split_names:
        changes = []
        for t,m in keys:
            a = bykey.get(('published_70_15_15.csv',t,m)); b = bykey.get(('prompt_disjoint.csv',t,m))
            if a and b:
                changes.append({'task':t,'model':m,'published_macro_f1':a['macro_f1'],
                                'prompt_disjoint_macro_f1':b['macro_f1'],
                                'decrease_percentage_points':round((float(a['macro_f1'])-float(b['macro_f1']))*100,3)})
        if changes:
            inventory['tables'].append(save_table(tables/'table_prompt_disjoint_changes.csv',changes,
                ['task','model','published_macro_f1','prompt_disjoint_macro_f1','decrease_percentage_points']))
            fig, ax = plt.subplots(figsize=(8, max(3, len(changes)*.7+1)))
            y = list(range(len(changes)))
            vals = [float(r['decrease_percentage_points']) for r in changes]
            bars = ax.barh(y, vals)
            ax.set_yticks(y,[r['task'].title()+' / '+r['model'].upper() for r in changes])
            ax.invert_yaxis()
            for bar, v in zip(bars, vals):
                ax.text(v+(0.15 if v>=0 else -0.15),bar.get_y()+bar.get_height()/2,
                        f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
            ax.set_xlim(min(0,min(vals)-1),max(1,max(vals)+2))
            ax.set_xlabel('Macro-F1 decrease (percentage points; published minus prompt-disjoint)')
            ax.set_title('Sensitivity to prompt-disjoint evaluation')
            tidy(ax)
            fig.tight_layout()
            inventory['figures'].append(save_figure(fig,figs/'figure_prompt_disjoint_change'))


def build_splits(base, tables, figs, inventory):
    report = load_json(base / 'split_integrity.json')
    if report is None:
        inventory['missing'].append('split_integrity.json')
        return
    rows=[]
    for name, r in sorted(report.items()):
        for part in ('train','validation','test'):
            rows.append({'manifest':name,'partition':part,'records':to_num(r.get('split_counts',{}).get(part)),
                         'all_rows':to_num(r.get('row_count')),
                         'missing_ids':to_num(r.get('missing_ids')),
                         'unknown_ids':to_num(r.get('unknown_ids')),
                         'duplicate_ids':to_num(r.get('duplicate_ids')),
                         'prompt_groups_crossing_splits':to_num(r.get('prompt_groups_crossing_splits')),
                         'manifest_valid':r.get('valid')})
    inventory['tables'].append(save_table(tables/'table_split_integrity.csv',rows,
        ['manifest','partition','records','all_rows','missing_ids','unknown_ids','duplicate_ids','prompt_groups_crossing_splits','manifest_valid']))
    names = list(report.keys()); x=list(range(3));width=.72/max(1,len(names))
    fig, ax=plt.subplots(figsize=(8,4.6))
    for j, name in enumerate(names):
        vals=[to_num(report[name].get('split_counts',{}).get(k)) for k in ('train','validation','test')]
        bars=ax.bar([i+(j-(len(names)-1)/2)*width for i in x],vals,width*.9,label=get_actual_manifest(name))
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+max(vals)*.01,f'{v:,}',ha='center',fontsize=8)
    ax.set_xticks(x,['Train','Validation','Test']);ax.set_ylim(0,max(to_num(report[n].get('split_counts',{}).get('train')) for n in names)*1.16)
    ax.legend(frameon=False);ax.set_title('Records per evaluation partition');tidy(ax,'Records')
    fig.tight_layout();inventory['figures'].append(save_figure(fig,figs/'figure_split_sizes'))
    fig,ax=plt.subplots(figsize=(7,4))
    ov=[to_num(report[n].get('prompt_groups_crossing_splits')) for n in names]
    bars=ax.bar(range(len(names)),ov)
    ax.set_xticks(range(len(names)),[get_actual_manifest(n) for n in names],rotation=12)
    for b,v in zip(bars,ov):ax.text(b.get_x()+b.get_width()/2,v+max(ov+[1])*.015,str(v),ha='center')
    ax.set_ylim(0,max(ov+[1])*1.2);ax.set_title('Source prompt IDs present in multiple partitions')
    tidy(ax,'Cross-partition prompt groups');fig.tight_layout()
    inventory['figures'].append(save_figure(fig,figs/'figure_prompt_group_overlap'))


def build_duplicates(base, tables, figs, inventory):
    all_reports = []
    paths = sorted(set([base/'duplicates'/'duplicate_report.json'] + list(base.glob('duplicates_*/duplicate_report.json'))))
    for path in paths:
        report = load_json(path)
        if not isinstance(report,list):continue
        for r in report:
            all_reports.append({'source_directory':path.parent.name,'field':r.get('field',''),
                                'threshold':r.get('threshold',''),
                                'metric':r.get('metric',''),
                                'candidate_method':r.get('candidate_method',''),
                                'candidate_pairs':r.get('candidate_pairs',''),
                                'verified_near_duplicate_pairs':r.get('near_duplicate_pairs',''),
                                'multi_record_clusters':r.get('multi_record_clusters',''),
                                'cross_original_split_pairs':r.get('cross_original_split_pairs',''),
                                'exact_duplicate_groups':r.get('exact_duplicate_groups',''),
                                'approximate_retrieval':r.get('approximate_retrieval','')})
    if not all_reports:
        inventory['missing'].append('duplicates/duplicate_report.json')
        return
    inventory['tables'].append(save_table(tables/'table_duplicates_by_threshold.csv',all_reports,
        ['source_directory','field','threshold','metric','candidate_method','candidate_pairs','verified_near_duplicate_pairs','multi_record_clusters','cross_original_split_pairs','exact_duplicate_groups','approximate_retrieval']))
    groups=defaultdict(dict)
    for r in all_reports:
        groups[str(r['threshold'])][r['field']]=to_num(r['verified_near_duplicate_pairs'])
    thresholds=sorted(groups,key=float)
    fields=sorted({f for g in groups.values() for f in g})
    fig,ax=plt.subplots(figsize=(max(7, len(thresholds)*1.7),4.3))
    w=.72/max(1,len(fields))
    for j,field in enumerate(fields):
        vals=[groups[t].get(field,0) for t in thresholds]
        bars=ax.bar([i+(j-(len(fields)-1)/2)*w for i in range(len(thresholds))], vals, w*.9,label=field)
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.15,str(v),ha='center',fontsize=9)
    ax.set_xticks(range(len(thresholds)),thresholds)
    ax.set_xlabel('Word 4-gram Jaccard confirmation threshold')
    ax.set_ylabel('Detected candidate pairs');ax.set_ylim(0,max([to_num(r['verified_near_duplicate_pairs']) for r in all_reports]+[1])*1.2+1)
    ax.legend(frameon=False);ax.set_title('MinHash-LSH candidate near-duplicates (approximate retrieval)')
    tidy(ax);fig.tight_layout()
    inventory['figures'].append(save_figure(fig,figs/'figure_detected_near_duplicates'))


def build_audit(base, tables, figs, inventory):
    report=load_json(base/'audit'/'audit_report.json')
    field_rows=load_csv(base/'audit'/'field_states.csv')
    if report is None:
        inventory['missing'].append('audit/audit_report.json')
    else:
        summary=[{'measure':'dataset_rows','value':report.get('dataset_rows')},
                 {'measure':'unique_note_ids','value':report.get('unique_note_ids')},
                 {'measure':'duplicate_note_ids','value':report.get('duplicate_note_ids')},
                 {'measure':'prompt_id_count','value':report.get('prompt_id_count')},
                 {'measure':'diagnosis_unique','value':report.get('diagnosis_unique')},
                 {'measure':'procedure_base_unique','value':report.get('procedure_base_unique')}]
        inventory['tables'].append(save_table(tables/'table_dataset_summary.csv',summary,['measure','value']))
    if field_rows is None:
        inventory['missing'].append('audit/field_states.csv')
    else:
        inventory['tables'].append(save_table(tables/'table_field_states.csv',field_rows,
            ['field','total_records','absent','null','empty_string','empty_collection','populated']))
        chosen=[r for r in field_rows if r.get('field') in ('allergies','sdoh_factors')]
        if chosen:
            fig,ax=plt.subplots(figsize=(7,4.3))
            states=['absent','null','empty_string','empty_collection','populated']
            for j,state in enumerate(states):
                vals=[to_num(r.get(state)) for r in chosen]
                left=[sum(to_num(r.get(s)) for s in states[:j]) for r in chosen]
                ax.barh(range(len(chosen)),vals,left=left,label=state.replace('_',' ').title())
            ax.set_yticks(range(len(chosen)),[r['field'] for r in chosen]);ax.invert_yaxis()
            ax.set_xlabel('Records');ax.set_title('Allergy and SDOH field-state distribution')
            ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12))
            ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
            fig.tight_layout();inventory['figures'].append(save_figure(fig,figs/'figure_allergies_sdoh_field_states'))
    specialties=load_csv(base/'audit'/'specialties.csv')
    if specialties:
        inventory['tables'].append(save_table(tables/'table_specialties.csv',specialties,['value','count']))
        vals=sorted(specialties,key=lambda r:to_num(r.get('count')))
        fig,ax=plt.subplots(figsize=(8,max(4,len(vals)*.42+1)))
        b=ax.barh([r['value'] for r in vals],[to_num(r['count']) for r in vals])
        for bar,r in zip(b,vals):ax.text(bar.get_width()+25,bar.get_y()+bar.get_height()/2,f"{to_num(r['count']):,}",va='center',fontsize=8)
        ax.set_xlim(0,max(to_num(r['count']) for r in vals)*1.18);ax.set_title('Records by specialty')
        ax.set_xlabel('Records');ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        fig.tight_layout();inventory['figures'].append(save_figure(fig,figs/'figure_specialty_distribution'))
    else:
        inventory['missing'].append('audit/specialties.csv')
    for fname in ('diagnoses','procedures','splits'):
        rows=load_csv(base/'audit'/f'{fname}.csv')
        if rows:
            inventory['tables'].append(save_table(tables/f'table_{fname}_distribution.csv', rows,['value','count']))


def build_validation(base, tables, inventory, valpath=None):
    if not valpath:
        return
    report=load_json(Path(valpath))
    if report is None:
        inventory['missing'].append(str(valpath))
        return
    rows=[]
    for label,key in [('Records evaluated','row_count'),('Runtime failures','generation_runtime_failure_count')]:
        if key in report:rows.append({'measure':label,'value':report[key]})
    for key,val in report.get('status_counts',{}).items():rows.append({'measure':'Final automated status: '+key,'value':val})
    for key,val in report.get('error_counts',{}).items():rows.append({'measure':'Final error: '+key,'value':val})
    if rows:
        inventory['tables'].append(save_table(tables/'table_llm_final_validation.csv',rows,['measure','value']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs',type=Path,default=Path(__file__).resolve().parents[1]/'outputs')
    p.add_argument('--paper-out',type=Path,default=Path(__file__).resolve().parents[1]/'paper_outputs'/'generated')
    p.add_argument('--llm-report',type=Path,help='Optional final LLM summary validation JSON report')
    a=p.parse_args()
    out=a.paper_out; tables=out/'tables';figs=out/'figures'
    inventory={'generated_utc':datetime.now(timezone.utc).isoformat(),
               'source_outputs_directory':str(a.outputs.resolve()),
               'figures':[],'tables':[],'missing':[],'warnings':[],
               'methodology_warning':'MinHash-LSH candidate retrieval is approximate. Automated support checks do not imply clinical correctness. Results are based on synthetic labels.'}
    build_audit(a.outputs,tables,figs,inventory)
    build_splits(a.outputs,tables,figs,inventory)
    build_duplicates(a.outputs,tables,figs,inventory)
    build_baselines(a.outputs,tables,figs,inventory)
    build_validation(a.outputs,tables,inventory,a.llm_report)
    out.mkdir(parents=True,exist_ok=True)
    (out/'REPORT_GENERATION_MANIFEST.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    print('Paper reporting complete:')
    print(f'  Figures: {len(inventory["figures"])} (PNG + PDF)')
    print(f'  Tables : {len(inventory["tables"])} (CSV + Markdown)')
    print(f'  Folder : {out}')
    if inventory['missing']:
        print('  Missing inputs skipped:',', '.join(inventory['missing']))
    if inventory['warnings']:
        print('  Warnings:', '; '.join(inventory['warnings']))

if __name__=='__main__':
    main()
