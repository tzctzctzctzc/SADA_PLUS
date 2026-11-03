"""只读诊断终审：独立负样本排序计数AUC和固定预算/风险统计。"""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def independent_auc(bg, scores):
    bg=np.asarray(bg,dtype=bool);scores=np.asarray(scores)
    negatives=np.sort(scores[~bg]);positive=scores[bg]
    if not len(negatives) or not len(positive):return None
    lower=np.searchsorted(negatives,positive,side='left');upper=np.searchsorted(negatives,positive,side='right')
    return float(np.mean((lower+upper)*.5/len(negatives)))


def main(root):
    parent=Path('/vepfs/vlms/background_ap_cost_source6_bd16507_1103')
    final=read(root/'final.json');assert set(final['results'])=={f'S{i}' for i in range(1,7)}
    result={}
    for shift,done in final['results'].items():
        entry=root/shift
        assert sha(entry/'complete.json')==final['complete_sha256'][shift]
        assert done==read(entry/'complete.json') and done['passed'] and done['parameters_unchanged'] and done['RNG_unchanged'] and done['all_validation_predictions_exact']
        assert sha(entry/'readout.json')==done['rows_sha256']
        assert sha(parent/shift/'fixed20_heads.pt')==done['origin_checkpoint_sha256']
        assert sha(parent/shift/'costs.json')==done['origin_costs_sha256']
        assert sha(parent/'independent_final_audit.json')==done['origin_final_audit_sha256']
        assert done['head_digest']==read(parent/shift/'gpu_complete.json')['final_digests']
        assert sha(done['guard'])==done['guard_sha256'];guard=read(done['guard']);lease=read(guard['lease'])
        assert sha(guard['lease'])==guard['lease_sha256'] and guard['owner_pid']==lease['owner_pid']==2728829
        assert lease['uuid']==guard['observed']['uuid'] and lease['index']==guard['observed']['index'] and guard['observed']['free_mib']>=12288
        rows=read(entry/'readout.json');cost=read(parent/shift/'costs.json')
        assert [r['video'] for r in rows['fit']]==[r['record']['video'] for r in cost['rows']]
        assert not {r['video'] for r in rows['fit']}&{r['video'] for r in rows['validation']}
        for row,truth in zip(rows['fit'],cost['rows']):
            assert row['bg']==truth['bg'] and row['delta']==truth['delta']
            data=torch.load(truth['record']['path'],map_location='cpu');ids=truth['hard_indices']
            assert row['scores']==data['scores'][ids].tolist() and row['labels']==data['labels'][ids].tolist()
        old=torch.load(parent/shift/'source_val_predictions.pt',map_location='cpu')
        for row,native in zip(rows['validation'],old['control']):
            assert row['video']==native['video'];mask=native['scores']>=.3
            assert row['scores']==native['scores'][mask].tolist() and row['labels']==native['labels'][mask].tolist()
            for mode in ('control','ap_cost'):
                other=next(r for r in old[mode] if r['video']==row['video'])
                assert row['pbg'][mode]==other['pbg'][mask].tolist()
        summary={}
        for role,items in rows.items():
            summary[role]={}
            for mode in ('control','ap_cost','confidence'):
                reported=done['results'][role][mode];values=[];scores=[];labels=[];bg=[];delta=[];harm_scores=[];benefit_scores=[]
                for row,v in zip(items,reported['videos']):
                    assert row['video']==v['video']
                    s=np.asarray(row['pbg'][mode]) if mode!='confidence' else 1-np.asarray(row['scores'])
                    b=np.asarray(row['bg'],dtype=bool);a=independent_auc(b,s)
                    if a is None:assert v['AUC'] is None
                    else:assert abs(a-v['AUC'])<1e-12;values.append(a)
                    scores.extend(s);labels.extend(row['labels']);bg.extend(b)
                    if role=='fit':
                        d=np.asarray(row['delta']);delta.extend(d);harm=np.maximum(-d,0)*(~b);benefit=np.maximum(d,0)*b
                        for weights,key,collector in [(harm,'harm_weighted_BG_score',harm_scores),(benefit,'benefit_weighted_BG_score',benefit_scores)]:
                            val=float(np.average(s,weights=weights)) if weights.sum()>0 else None
                            if val is None:assert v[key] is None
                            else:assert abs(val-v[key])<1e-12;collector.append(val)
                assert len(values)==reported['AUC_valid_videos'] and len(items)==reported['total_videos']
                assert abs(np.mean(values)-reported['video_macro_AUC'])<1e-12
                bg=np.asarray(bg,dtype=bool);scores=np.asarray(scores);labels=np.asarray(labels)
                for rate,report in zip((.005,.01,.02,.05),reported['all_fixed_budgets']):
                    chosen=[]
                    for c in range(10):
                        ids=np.flatnonzero(labels==c);order=np.lexsort((ids,-scores[ids]));chosen.extend(ids[order[:int(np.floor(rate*len(ids)))]].tolist())
                    chosen=sorted(chosen);assert chosen==report['selected_indices']
                    assert int(bg[chosen].sum())==report['BG_removed'] and len(chosen)-int(bg[chosen].sum())==report['nonBG_removed']
                    if role=='fit':
                        harm=np.maximum(-np.asarray(delta),0)*(~bg);proxy=float(harm[chosen].sum()/harm.sum())
                        assert abs(proxy-report['deleted_harm_cost_share_proxy_not_AP'])<1e-12
                if role=='fit':
                    assert abs(np.mean(harm_scores)-reported['harm_weighted_BG_score_macro'])<1e-12
                    assert abs(np.mean(benefit_scores)-reported['benefit_weighted_BG_score_macro'])<1e-12
                summary[role][mode]={k:v for k,v in reported.items() if k not in ('videos','all_fixed_budgets')}
                summary[role][mode]['budgets']=[{k:v for k,v in b.items() if k!='selected_indices'} for b in reported['all_fixed_budgets']]
        result[shift]=dict(complete_sha256=sha(entry/'complete.json'),summary=summary)
    out=dict(passed=True,source_only=True,no_AP_reestimate=True,final_sha256=sha(root/'final.json'),results=result)
    (root/'independent_final_audit.json').write_text(json.dumps(out,sort_keys=True))
    for role in ('fit','validation'):
        print(role,{m:float(np.mean([result[s]['summary'][role][m]['video_macro_AUC'] for s in result])) for m in ('control','ap_cost','confidence')},flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
