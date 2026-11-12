import csv, hashlib, json, zipfile
from collections import Counter
from pathlib import Path

def records(path):
    path=Path(path)
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as z:
            names=[x for x in z.namelist() if x.endswith('.jsonl') and not x.startswith('__MACOSX/')]
            if len(names)!=1: raise ValueError(f'Expected one JSONL in ZIP, found {names}')
            with z.open(names[0]) as f:
                for i,b in enumerate(f,1):
                    if b.strip(): yield json.loads(b)
    else:
        with path.open(encoding='utf-8') as f:
            for i,line in enumerate(f,1):
                if line.strip(): yield json.loads(line)

def write_csv(path, rows, fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    fields=fields or (list(rows[0]) if rows else [])
    if not fields: return path.write_text('',encoding='utf-8')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def get_labels(record, task):
    key='diagnoses' if task=='diagnosis' else 'procedures'
    code='diagnosis_code' if task=='diagnosis' else 'procedure_code_base'
    return sorted({str(d.get(code,'')) for d in record.get(key,[]) if isinstance(d,dict) and d.get(code)})
