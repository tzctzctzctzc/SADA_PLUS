"""仅审计本轮原Prototype基线配对与原生AP，不混入Trident/调参读出。"""
from pathlib import Path
import json


def audit(entry):
    import torch,numpy as np,pandas as pd
    from tools.baseline_protocol import read,save,sha,termination
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    from tools.audit_background_ap_cost import IndependentAP
    entry=Path(entry);cfg=read(entry/'config.json');binding=read(entry/'binding.json')
    from tools.run_class_path_formal import specification
    original=read(Path(binding['reference'])/'config.json');route=cfg['uda_route']['kind']
    assert cfg['uda_route']==specification()
    assert read(entry/'runtime_config.json')==cfg
    assert cfg['seed']==original['seed']==123 and not cfg['model']['use_trident_head'] and not cfg.get('head_replacement')
    for key in ('opt','loader','train_cfg','test_cfg','prototype_capacity','src_train_split','tgt_train_split','src_val_split','tgt_val_split','dataset'):
        assert cfg[key]==original[key],key
    expected=dict(original['model'])
    assert cfg['model']==expected and read(entry/'route_initialization.json')['native_sha']==binding['initial_state_sha']
    for path,expected_sha in binding['inputs'].items():assert sha(path)==expected_sha
    source=read(cfg['dataset']['json_file'])['database'];target=read(cfg['baseline']['target_manifest'])['database']
    assert not set(source)&set(target) and all('annotations' not in r for r in target.values())
    best=torch.load(entry/'best.pt',map_location='cpu');latest=torch.load(entry/'latest.pt',map_location='cpu')
    stop=read(entry/'termination.json');assert termination(stop['history'])==stop['decision']
    assert best['epoch']==stop['decision']['best_epoch'] and latest['epoch']==stop['decision']['epoch']
    for saved in (best,latest):
        assert {'model_controls','ema_controls','state_dict','ema_state_dict','optimizers','schedulers','rng'}<=set(saved)
        assert len(saved['optimizers'])==len(saved['schedulers'])==3
        assert {'python','numpy','cpu','cuda','loaders'}<=set(saved['rng'])
        assert 'adapt_route.calls' in saved['state_dict']
    rows=[json.loads(line) for line in (entry/'metrics.jsonl').read_text().splitlines()]
    assert all(r['target_gt_cls_loss']==r['target_gt_reg_loss']==r['target_gt_downstream_loss']==0 for r in rows)
    result=dict(status='passed',route=route,shift=cfg['baseline']['shift'],no_trident=True,target_train_GT=False,
                best_epoch=best['epoch'],latest_epoch=latest['epoch'],best_sha256=sha(entry/'best.pt'),independent_AP={})
    result['baseline_shared_discriminator_changed_to_per_scale']=False
    active=[r for r in rows if r['epoch']>=cfg['uda_route']['start_epoch']]
    assert active
    result['active_group_epoch_mean']=sum(r.get('uda_route',{}).get('aligned_groups',0) for r in active)/len(active)
    assert result['active_group_epoch_mean']>0
    assert all(r['uda_route']['reconstruction']==0 for r in active)
    assert 'adapt_route.projection' in best['state_dict'] and 'adapt_route.projection' in latest['ema_state_dict']
    result['source_anchor']=True;result['original_discriminator_retained']=True
    torch.set_num_threads(2)
    for domain,splits in [('source',cfg['src_val_split']),('target',cfg['tgt_val_split'])]:
        prediction=pd.DataFrame(torch.load(entry/'evaluation'/(domain+'_predictions.pt'),map_location='cpu'))
        truth=load_gt_seg_from_json(cfg['baseline']['annotation_for_offline_evaluation'],split=splits,action_subsampling='10_majority_classes')
        truth=truth[truth['t-start']!=truth['t-end']].reset_index(drop=True)
        ap=np.asarray([IndependentAP(truth[truth.label==c].reset_index(drop=True),prediction[prediction.label==c].reset_index(drop=True)).ap(
            np.ones(int((prediction.label==c).sum()),dtype=bool)) for c in sorted(truth.label.unique())]).mean(0)
        official=read(entry/'evaluation'/(domain+'_val.json'))
        assert np.max(np.abs(ap-np.asarray(official['map_by_tiou'])))<1e-7
        assert abs(ap.mean()-official['map_0.1_to_0.5'])<1e-7 and official['checkpoint_sha256']==result['best_sha256']
        result['independent_AP'][domain]=dict(map_by_tiou=ap.tolist(),mean_map=float(ap.mean()))
    result['delta_vs_prototype_pp']=100*result['independent_AP']['target']['mean_map']-binding['baseline_target_percent']
    save(entry/'state_audit.json',result);return result
