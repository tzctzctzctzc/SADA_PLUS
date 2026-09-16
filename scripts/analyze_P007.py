import sys,json
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lad_bridge import official_lad_functions
F=official_lad_functions();cfg=json.loads(Path('configs/P007_lad_bridge.json').read_text())
data={}
for mode in cfg['modes']:
    root=Path('runs/P007_lad_'+mode)
    assert json.loads((root/'summary.json').read_text())['complete']
    data[mode]=[json.loads(l) for l in (root/'predictions.jsonl').read_text().splitlines()]
cal={}
for mode,rows in data.items():
    c=[r for r in rows if r['split']=='calibration']
    stats={i:[np.array(r['scores']) for r in c if r['gold']==i] for i in range(4)}
    assert all(len(v)==5 for v in stats.values())
    B,I,L=F['compute_bias_matrix'](stats,4,verbose=False)
    cal[mode]={'B':B,'prior':np.mean([r['layer_attention'] for r in c],axis=0),
        'patch_prior':np.mean([r['parts']['patch'] for r in c],axis=0)}
# Shared-image connected components for exploratory cluster bootstrap.
ev={r['id']:r for r in cfg['evaluation']}
parent={i:i for i in ev}
def find(i):
    while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
    return i
for i,a in ev.items():
    for j,b in ev.items():
        if set(a['image_hashes'])&set(b['image_hashes']):parent[find(i)]=find(j)
clusters={i:find(i) for i in ev}
records=[]
for mode,rows in data.items():
    for r in rows:
        if r['split']!='evaluation':continue
        a=np.array(r['layer_attention']); z=np.array(r['scores']); c=cal[mode]
        raw=F['select_attention_scores'](a,None,cfg['topk'])
        adj=F['select_attention_scores'](a,c['prior'],cfg['topk'])
        pa=F['sharpen_distribution'](adj,cfg['sharpen_power'])
        patch=F['select_attention_scores'](np.array(r['parts']['patch']),c['patch_prior'],cfg['topk'])
        other=cal['baseline']
        pa_base=F['sharpen_distribution'](F['select_attention_scores'](a,other['prior'],cfg['topk']),cfg['sharpen_power'])
        pred={'logits':int(z.argmax()),'attention_raw':int(raw.argmax()),'attention_calibrated':int(adj.argmax()),
            'agd_official':int((z-pa@c['B']).argmax()),'patch_attention_calibrated':int(patch.argmax()),
            'agd_baseline_calibration':int((z-pa_base@other['B']).argmax())}
        top=np.argsort(a.mean(1))[-cfg['topk']:]
        boundary=(np.array(r['parts']['start'])+np.array(r['parts']['end']))[top].sum()
        total=a[top].sum()
        records.append({'id':r['id'],'base_id':r['base_id'],'cluster':clusters[r['base_id']],
            'mode':mode,'gold':r['gold'],'predictions':pred,'top_layers':[cfg['layer_range'][0]+int(i) for i in top],
            'selected_boundary_mass_fraction':float(boundary/total),'attention_probs':pa.tolist()})
summary={'purpose':cfg['purpose'],'n_base_questions':len(ev),'n_image_clusters':len(set(clusters.values())),
 'calibration_base_questions':len(cfg['calibration']),'calibration_eval_exact_image_overlap':False,
 'models':{},'limitations':['Exploratory reused Mantis questions; no independent confirmation','Calibration and evaluation byte-disjoint, not perceptual near-duplicate checked',
 'LAD transfer uses original code aggregation and 336px preprocessing, not original paper benchmark',
 'Do not treat 4 permutations of a question as 4 independent examples'],
 'calibrations':{m:{k:v.tolist() for k,v in c.items()} for m,c in cal.items()}}
for mode in cfg['modes']:
    rr=[r for r in records if r['mode']==mode]
    report={'n_permutations':len(rr),'methods':{},'mean_selected_boundary_mass_fraction':float(np.mean([r['selected_boundary_mass_fraction'] for r in rr]))}
    for method in rr[0]['predictions']:
        correct=np.array([r['predictions'][method]==r['gold'] for r in rr])
        report['methods'][method]={'correct':int(correct.sum()),'accuracy':float(correct.mean()),
          'all_permutations_correct_questions':sum(all(r['predictions'][method]==r['gold'] for r in rr if r['base_id']==i) for i in ev),
          'prediction_counts':dict(Counter(r['predictions'][method] for r in rr))}
    delta=np.array([int(r['predictions']['agd_official']==r['gold'])-int(r['predictions']['logits']==r['gold']) for r in rr])
    groups=sorted(set(clusters.values()))
    totals=np.array([[sum(delta[i] for i,r in enumerate(rr) if r['cluster']==g),sum(r['cluster']==g for r in rr)] for g in groups])
    rng=np.random.default_rng(20260917)
    boot=totals[rng.integers(0,len(groups),size=(10000,len(groups)))].sum(1)
    report['agd_minus_logits_cluster_bootstrap_95ci_pp']=(100*np.quantile(boot[:,0]/boot[:,1],[.025,.975])).tolist()
    report['attention_correct_logits_wrong']=sum(r['predictions']['attention_calibrated']==r['gold'] and r['predictions']['logits']!=r['gold'] for r in rr)
    report['logits_correct_attention_wrong']=sum(r['predictions']['attention_calibrated']!=r['gold'] and r['predictions']['logits']==r['gold'] for r in rr)
    summary['models'][mode]=report
Path('results/P007_lad_bridge.json').write_text(json.dumps(summary,indent=2))
Path('runs/P007_lad_predictions.json').write_text(json.dumps(records,indent=2))
print(json.dumps(summary,indent=2))
