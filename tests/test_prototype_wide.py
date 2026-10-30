"""宽度、配对配置、真实验收拒绝条件及停止语义；不镜像训练循环。"""
from copy import deepcopy
import json
import os
import subprocess
import sys

import pytest
import torch

from libs.models.SADA.prototype_capacity import specification,build_capacity,VLMProposalCoAdaptationBranch
from tools.run_prototype_wide import (assert_control_pair,validate_probe,memory_policy,
                                     finish_process,assert_alignment_weight,SHIFTS,VARIANT)


def test_only_width_changes_and_all_parameter_shapes_match(tmp_path):
    row=dict(proposal_start_sec=16.,proposal_end_sec=36.,source_fpn_levels=[0],candidate_id='a',
        semantic_label='TAKE',class_probabilities=[.9]+[.1/9]*9,participation_selected=True)
    path=tmp_path/'cache.json';path.write_text(json.dumps(dict(by_video={'t':[row]})))
    import hashlib
    cfg=dict(variant=VARIANT,prepared_cache=str(path),prepared_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    wide=build_capacity(cfg,512,97,[1],[[0,10000]],'cpu').branch
    assert len(wide.encoder.layers)==1 and wide.input_projection.weight.shape==(256,512)
    assert wide.position_embedding.shape==(7,256) and wide.output_norm.weight.shape==(256,)
    assert wide.classifier.weight.shape==(97,256) and wide.source_prototypes.shape==(97,256)
    assert wide.encoder.layers[0].linear1.weight.shape==(512,256)
    a=specification('control');b=specification(VARIANT)
    assert {k for k in a if a[k]!=b[k]}=={'variant','embedding_dim','hidden_dim'}
    with torch.random.fork_rng():
        torch.manual_seed(0);native=VLMProposalCoAdaptationBranch(512,97,hidden_dim=512,embedding_dim=256,confidence_threshold=0.)
    assert all(torch.equal(v,native.state_dict()[k]) for k,v in wide.state_dict().items())
    assert SHIFTS==('S1','S2','S3','S4','S5','S6')


def pair():
    old=dict(seed=123,init_rand_seed=123,loader=dict(batch_size=2),opt=dict(learning_rate=.0001,epochs=50),
        baseline=dict(code={'commit':'old'},tmpdir='/old',policy={'patience_epochs':15}),
        prototype_capacity=dict(variant='control',specification=specification('control'),
            prepared_cache='/same.json',prepared_cache_sha256='same'))
    new=deepcopy(old);new['baseline'].update(code={'commit':'new'},tmpdir='/new')
    new['prototype_capacity'].update(variant=VARIANT,specification=specification(VARIANT))
    return new,old


def test_control_pair_accepts_only_declared_width_and_run_identity():
    new,old=pair();assert_control_pair(new,old)


@pytest.mark.parametrize('change',['weight','depth','cache','seed','batch','lr','patience'])
def test_pair_rejects_other_scientific_changes(change):
    new,old=pair()
    if change=='weight':new['prototype_capacity']['specification']['target_alignment_weight']=.2
    elif change=='depth':new['prototype_capacity']['specification']['layers']=2
    elif change=='cache':new['prototype_capacity']['prepared_cache_sha256']='changed'
    elif change=='seed':new['seed']=6
    elif change=='batch':new['loader']['batch_size']=4
    elif change=='lr':new['opt']['learning_rate']=.0002
    else:new['baseline']['policy']['patience_epochs']=10
    with pytest.raises(ValueError):assert_control_pair(new,old)


def report():
    phase=dict(branch_parameters_updated=True,model_updated=True,target_gt_losses=[0.,0.,0.],
        source_gradient_norm=.2,peak_reserved_mib=4500.,auxiliary=dict(rq3_target_gradient_norm=.001,
        rq3_alignment_loss=.1,rq3_source_loss=.5,rq3_loss=.51,rq3_selected=1.))
    return dict(status='passed',batch=2,initial_detector_sha='same',rng_roundtrip={'status':'passed'},
        checked_target_batches=10,phases={'1':deepcopy(phase),'5':deepcopy(phase)},peak_reserved_mib=6000.)


def test_real_acceptance_and_memory_keep_training_and_full_video_budgets_separate():
    r=report();validate_probe(r,'same');policy=memory_policy(r)
    assert policy['training_minimum_free_mib']==9216
    assert policy['evaluation_minimum_free_mib']==12288


@pytest.mark.parametrize('change',['initial','target_gt','rng','gradient','branch','weight','selected'])
def test_acceptance_rejects_unreal_or_mismatched_evidence(change):
    r=report()
    if change=='initial':r['initial_detector_sha']='other'
    elif change=='target_gt':r['phases']['1']['target_gt_losses'][0]=.1
    elif change=='rng':r['rng_roundtrip']['status']='failed'
    elif change=='gradient':r['phases']['5']['auxiliary']['rq3_target_gradient_norm']=0.
    elif change=='branch':r['phases']['5']['branch_parameters_updated']=False
    elif change=='weight':r['phases']['5']['auxiliary']['rq3_loss']=.52
    else:r['phases']['5']['auxiliary']['rq3_selected']=0.
    with pytest.raises(ValueError):validate_probe(r,'same')


def test_actual_control_style_log_has_no_explicit_weight_but_loss_is_checked():
    aux=dict(rq3_loss=.398184301437072,rq3_source_loss=.3904998915375404,
             rq3_alignment_loss=.07684411530224781)
    assert_alignment_weight(aux)
    aux['rq3_loss']=aux['rq3_source_loss']+.2*aux['rq3_alignment_loss']
    with pytest.raises(ValueError):assert_alignment_weight(aux)
    aux['rq3_loss']=aux['rq3_source_loss']+.1*aux['rq3_alignment_loss']
    aux['rq3_target_alignment_weight']=.2
    with pytest.raises(ValueError):assert_alignment_weight(aux)


@pytest.mark.skipif(os.name=='nt',reason='真实Linux进程组停止')
def test_stage_child_is_reaped_when_supervision_stops():
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    try:
        finish_process(child);assert child.poll() is not None
    finally:
        if child.poll() is None:child.kill();child.wait()
