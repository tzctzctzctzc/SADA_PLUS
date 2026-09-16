import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
for prefix,folders,tag in [('P004',['P004_mantis_3b_baseline','P004_mantis_3b_scale'],'category'),
                           ('P005',['P005_muir_dev_baseline','P005_muir_dev_scale'],'task')]:
 data=[]
 for folder in folders:
    p=Path('runs')/folder
    assert json.loads((p/'summary.json').read_text())['complete']
    data.append({r['id']:r for r in map(json.loads,(p/'predictions.jsonl').read_text().splitlines())})
 a,b=data;assert a.keys()==b.keys()
 ids=list(a);gain=sum(not a[i]['correct'] and b[i]['correct'] for i in ids);loss=sum(a[i]['correct'] and not b[i]['correct'] for i in ids)
 report={'n':len(ids),'baseline_correct':sum(r['correct'] for r in a.values()),'scale_correct':sum(r['correct'] for r in b.values()),
  'gains':gain,'losses':loss,'descriptive_question_level_mcnemar_p':float(binomtest(gain,gain+loss).pvalue) if gain+loss else 1,
  'limitations':['One sampling seed; descriptive question-level test ignores shared-image dependence','Fixed public 3B example parameters; not a tuned 7B replication' if prefix=='P005' else 'Official public 3B example parameters; paper scaled score not reproduced'],
  'groups':{},'changed_ids':[i for i in ids if a[i]['correct']!=b[i]['correct']]}
 groups=defaultdict(list)
 for i in ids:groups[a[i][tag]].append(i)
 for k,v in groups.items(): report['groups'][k]={'n':len(v),'baseline_correct':sum(a[i]['correct'] for i in v),'scale_correct':sum(b[i]['correct'] for i in v)}
 if prefix=='P005':
    pairs={(i,a[i]['counterpart_idx']) for i in ids if a[i]['counterpart_idx'] in a}
    pairs={tuple(sorted(pair)) for pair in pairs}
    report['both_counterpart_questions_correct']={m:sum(all(dd[i]['correct'] for i in pair) for pair in pairs) for m,dd in zip(['baseline','scale'],data)}
    report['n_counterpart_pairs']=len(pairs)
 Path('results/'+prefix+'_paired_analysis.json').write_text(json.dumps(report,indent=2))
 print(prefix,json.dumps(report))
