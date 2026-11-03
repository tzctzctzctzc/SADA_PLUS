"""审计原训练器配对协议、完整状态与历史on结果；不宣称独立重写AP。"""
import argparse
from pathlib import Path
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,termination,decision,code_identity
from tools.run_canonical_sada_ablation import TASKS,assert_pair


def run(root):
    import torch
    from tools.baseline_engine import tensor_digest
    while not (root/'final.json').exists():
        if (root/'stop.json').exists():raise RuntimeError('user stopped; no continuation')
        time.sleep(5)
    final=read(root/'final.json');state=read(root/'state.json');reports={}
    for shift in ('S1','S2'):assert_pair(read(root/(shift+'_on/config.json')),read(root/(shift+'_off/config.json')))
    for task in final['completed']:
        e=root/task;cfg=read(e/'config.json');binding=read(e/'origin_binding.json');complete=read(e/'complete.json')
        for path,digest in complete['artifacts'].items():assert sha(e/path)==digest
        for path,digest in binding['inputs'].items():assert sha(path)==digest
        for stage in ('train','evaluate'):
            proof=read(e/(stage+'_complete.json'));guard=read(proof['guard']);lease=read(guard['lease'])
            assert sha(proof['guard'])==proof['guard_sha256'] and sha(guard['lease'])==guard['lease_sha256']
            assert lease['owner_pid']==guard['owner_pid']==state['owner_pid']
            assert guard['observed']['uuid']==lease['observed']['uuid'] and guard['observed']['free_mib']>=16384
        probe=read(e/'probe.json');assert probe['initial_state_sha']==binding['initial_state_sha'] and probe['rng_roundtrip']['status']=='passed'
        reuse=read(e/'probe_reuse.json');assert sha(Path(reuse['origin'])/'probe.json')==reuse['probe_sha256']==sha(e/'probe.json')
        assert sha(reuse['guard'])==reuse['guard_sha256']
        old_cfg=read(Path(reuse['origin'])/'config.json');expected=__import__('copy').deepcopy(cfg)
        for name in ('code','tmpdir'):expected['baseline'][name]=old_cfg['baseline'][name]
        assert expected==old_cfg
        initial=read(e/'formal_initialization.json');assert initial['passed'] and initial['initial_state_sha']==binding['initial_state_sha'] and not initial['loaded_trained_checkpoint']
        term=read(e/'termination.json');chosen=termination(term['history']);assert term['initial_state_sha']==binding['initial_state_sha']
        for i in range(1,len(term['history'])):assert decision(term['history'][:i])['reason']=='running'
        rows=[__import__('json').loads(line) for line in (e/'metrics.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in rows]==list(range(1,chosen['epoch']+1))
        assert all(r['target_gt_cls_loss']==r['target_gt_reg_loss']==r['target_gt_downstream_loss']==0 for r in rows)
        best=torch.load(e/'best.pt',map_location='cpu');last=torch.load(e/'latest.pt',map_location='cpu')
        assert best['epoch']==chosen['best_epoch'] and last['epoch']==chosen['epoch']
        for saved in (best,last):
            assert saved['config']==cfg and len(saved['optimizers'])==len(saved['schedulers'])==3
            assert set(saved['rng'])=={'python','numpy','cpu','cuda','loaders'} and len(saved['rng']['cuda'])==1
        target=read(e/'evaluation/target_val.json');source=read(e/'evaluation/source_val.json')
        assert target['checkpoint_sha256']==source['checkpoint_sha256']==sha(e/'best.pt')
        assert target['selected_epoch']==source['selected_epoch']==chosen['best_epoch']
        delta=100*(target['map_0.1_to_0.5']-binding['old_target_map'])
        reports[task]=dict(target_map=target['map_0.1_to_0.5'],source_map=source['map_0.1_to_0.5'],
            delta_vs_original_pp=delta,stop=chosen,complete_sha256=sha(e/'complete.json'),original_initialization_verified=True)
        if task.endswith('_on'):reports[task]['historical_target_AP_reproduction_within_005pp']=abs(delta)<=.05
        print('CANONICAL_AUDITED',task,delta,flush=True)
    pairs={}
    for shift in ('S1','S2'):
        on,off=shift+'_on',shift+'_off'
        if on in reports and off in reports:
            pairs[shift]=dict(off_minus_on_pp=100*(reports[off]['target_map']-reports[on]['target_map']),
                historical_on_reproduced=reports[on]['historical_target_AP_reproduction_within_005pp'],
                caveat='same-seed paired development evidence; original official AP evaluator reused')
            a=read(root/on/'probe.json');b=read(root/off/'probe.json')
            assert a['initial_state_sha']==b['initial_state_sha']
            assert a['phases']['1']['loss']==b['phases']['1']['loss']
    save(root/'independent_protocol_audit.json',dict(passed=len(reports)==4,code=code_identity(),
        producer_final_sha256=sha(root/'final.json'),reports=reports,pairs=pairs,failed=final['failed'],paused=final['paused'],
        independent_AP_reimplementation=False,scope='S1/S2 development; original training from original seed, only SADA adversarial weight intervention'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
