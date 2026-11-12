#!/usr/bin/env python3
"""Read-only prompt provenance inventory. Never invent template family assignments."""
import argparse,collections,csv,hashlib,re
from pathlib import Path
from common import records,write_csv,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--generator',help='Original v21 generator source for evidence inventory');p.add_argument('--out',required=True);p.add_argument('--verified-template-map',help='Optional externally verified mapping to audit, never inferred');a=p.parse_args()
 data=list(records(a.input)); groups=collections.defaultdict(list);styles=collections.defaultdict(set);scenarios=collections.Counter();versions=collections.Counter()
 for r in data:
  pid=str(r.get('source_prompt_id','')).strip();groups[pid].append(r['note_id']);styles[pid].add(str(r.get('note_style','')));scenarios[str(r.get('specialty',''))]+=1;versions[str(r.get('llm_summary_prompt_version',''))]+=1
 generator=Path(a.generator) if a.generator else None
 text=generator.read_text(encoding='utf8') if generator and generator.exists() else ''
 source_details={'provided':bool(generator),'found':bool(text),'path':str(generator) if generator else None,'sha256':hashlib.sha256(text.encode()).hexdigest() if text else None,'mentions_source_prompt_id':'source_prompt_id' in text,'mentions_template_family_id':'template_family_id' in text,'mentions_template_id':bool(re.search(r'\btemplate_id\b',text)),'note':'String searches only, not verified lineage from source_prompt_id to template families.'}
 mapping=None;mapping_errors=[]
 if a.verified_template_map:
  mapping={};path=Path(a.verified_template_map)
  with path.open(newline='',encoding='utf8') as f:
   for row in csv.DictReader(f):
    p_id=row.get('source_prompt_id','');fam=row.get('template_family_id','')
    if not p_id or not fam:mapping_errors.append('Empty prompt/family in mapping')
    if p_id in mapping and mapping[p_id]!=fam:mapping_errors.append('Conflicting family mapping for '+p_id)
    mapping[p_id]=fam
  mapping_errors += [f'{len(set(groups)-set(mapping))} missing prompt IDs'] if set(groups)-set(mapping) else []
  mapping_errors += [f'{len(set(mapping)-set(groups))} unknown prompt IDs'] if set(mapping)-set(groups) else []
 rows=[{'source_prompt_id':p,'record_count':len(m),'observed_note_styles':'|'.join(sorted(styles[p])),'verified_template_family_id':mapping.get(p,'') if mapping else ''} for p,m in sorted(groups.items())]
 out=Path(a.out);write_csv(out/'source_prompt_inventory.csv',rows)
 valid=bool(mapping) and not mapping_errors
 report={'dataset_rows':len(data),'unique_source_prompt_ids':len(groups),'specialty_distribution':scenarios,'summary_prompt_versions':versions,'generator_source_checks':source_details,'verified_mapping_provided':bool(mapping),'verified_mapping_valid':valid,'verified_template_family_count':len(set(mapping.values())) if valid else None,'mapping_errors':mapping_errors,'template_disjoint_status':'verified_mapping_available' if valid else 'NOT_ESTABLISHED; do not generate template_disjoint.csv','important':'120 source_prompt_id values do not prove 120 independent underlying template families. This script never creates a mapping without supplied documentary verification.'}
 write_json(out/'template_provenance_report.json',report);print('Template provenance:',len(groups),'source prompts; verified family map:',valid)
if __name__=='__main__':main()
