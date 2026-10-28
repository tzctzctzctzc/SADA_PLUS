import copy
import importlib.util
from pathlib import Path
import sys
import torch
import pytest

from test_rq3_revised import MODULE as OLD, prototype_data
from tools.rq3_enhancement import ORDER, specification, validate, assert_reference_pair

spec = importlib.util.spec_from_file_location('rq3_revised_cpu.rq3_enhancement',
    Path(__file__).resolve().parents[1]/'libs/models/SADA/rq3_enhancement.py')
NEW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NEW)


def build(name):
    from test_rq3_revised import entry
    return NEW.EnhancedPrototype(4,10,{'v':[dict(entry(),participation_selected=True)]},[1],[(0,99)],0.,specification(name))


@pytest.mark.parametrize('name',ORDER)
def test_real_branch_backward_target_gt_isolation_and_strict_restore(name):
    torch.set_num_threads(1)
    torch.manual_seed(6)
    before=torch.random.get_rng_state().clone()
    model=build(name)
    assert torch.equal(before,torch.random.get_rng_state())
    data=prototype_data()
    data[1][0]['labels']=object()
    data[1][0]['segments']=object()
    result=model(*data,epoch=5)
    assert torch.isfinite(result['loss']) and result['target_gradient_norm']>0
    result['loss'].backward()
    assert data[2][0].grad.abs().sum()>0 and data[3][0].grad.abs().sum()>0
    other=build(name)
    other.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),other.state_dict().values()))


def test_control_exact_old_state_and_loss():
    from test_rq3_revised import entry
    old=OLD.RQ3ProposalPrototype(4,10,{'v':[dict(entry(),participation_selected=True)]},[1],[(0,99)],0.)
    new=build('control')
    assert all(torch.equal(old.state_dict()[k],v) for k,v in new.state_dict().items())
    torch.manual_seed(6)
    data=prototype_data()
    a=old(*data,epoch=5)
    b=new(*data,epoch=5)
    for k in a:
        assert torch.equal(torch.as_tensor(a[k]),torch.as_tensor(b[k]))


def test_parameter_counts_and_pool_scope():
    for name, expected in [('control',200576),('wide',663296),('deep',1190400)]:
        model=NEW.EnhancedPrototype(512,10,{},[1],[(0,99)],0.,specification(name))
        assert sum(p.numel() for p in model.parameters())==expected
    assert build('pool').branch.sample_ratios == build('control').branch.sample_ratios


def test_source_supcon_correct_pairs_and_no_positive_case():
    x=torch.nn.functional.normalize(torch.randn(6,12),dim=-1).requires_grad_()
    target=torch.nn.functional.one_hot(torch.tensor([0,0,1,1,2,3]),10).float()
    loss,n=NEW.source_supcon(x,target,.1)
    assert n==4 and torch.isfinite(loss)
    loss.backward()
    assert x.grad.abs().sum()>0
    loss,n=NEW.source_supcon(x[:2],torch.eye(10)[:2],.1)
    assert n==0 and loss==0


def test_multiple_prototypes_source_only_and_unit_length():
    model=build('multiprototype')
    data=prototype_data()
    model(*data,epoch=5)
    branch=model.branch
    assert branch.subprototype_ready[:2].any()
    norms=branch.subprototypes.norm(dim=-1)[branch.subprototype_ready]
    assert torch.allclose(norms,torch.ones_like(norms),atol=1e-6)
    assert not branch.subprototypes.requires_grad


def test_warmup_reaches_frozen_weight_and_no_target_gradient_before_start():
    model=build('warmup')
    for epoch,expected in [(0,0.),(1,.02),(5,.1),(50,.1)]:
        out=model(*prototype_data(),epoch=epoch)
        assert abs(out['alignment_weight']-expected)<1e-8
        assert (out['target_gradient_norm']>0)==(epoch>0)


def test_spec_tampering_and_nonbranch_changes_rejected():
    cfg={'rq3_enhancement':specification('deep')}
    validate(cfg)
    cfg['rq3_enhancement']['embedding_dim']=512
    with pytest.raises(ValueError):
        validate(cfg)
    reference={'seed':6,'loader':{'batch_size':2},'rq3_auxiliary':{'selected_coordinates':[['v',2]]}}
    cfg=copy.deepcopy(reference)
    cfg['rq3_enhancement']=specification('wide')
    assert_reference_pair(cfg,reference)
    cfg['rq3_auxiliary']['selected_coordinates']=[['v',3]]
    with pytest.raises(ValueError):
        assert_reference_pair(cfg,reference)


def test_numpy2_namespace_checkpoint_is_read_without_changing_array(tmp_path):
    import numpy as np
    import zipfile
    from tools.rq2_evaluate import load_checkpoint
    source=tmp_path/'source.pt'
    moved=tmp_path/'numpy2.pt'
    values=np.arange(6,dtype=np.float64)
    torch.save({'array':values,'weight':torch.ones(3)},source)
    with zipfile.ZipFile(source) as inp, zipfile.ZipFile(moved,'w') as out:
        for name in inp.namelist():
            data=inp.read(name)
            if name.endswith('data.pkl'):
                assert b'numpy.core.multiarray' in data
                data=data.replace(b'numpy.core.multiarray',b'numpy._core.multiarray')
            out.writestr(name,data)
    restored=load_checkpoint(moved)
    assert np.array_equal(restored['array'],values)
    assert torch.equal(restored['weight'],torch.ones(3))


def test_registry_annotation_symlink_preserves_logical_provenance(tmp_path):
    import json,yaml
    from tools.extract_rq2_dense_vlm_features import video_selection_from_registry
    from tools.rq2_dense_protocol import sha256_normalized_text
    repo=tmp_path/'repo'
    (repo/'configs').mkdir(parents=True)
    content=tmp_path/'content.json'
    content.write_text(json.dumps({'database':{'v':{'subset':'train'}}}))
    (repo/'annotation.json').symlink_to(content)
    cfg={'dataset':{'json_file':'annotation.json'},'src_train_split':['train'],
         'src_val_split':['val'],'tgt_train_split':['target'],'tgt_val_split':['target_val']}
    path=repo/'configs/shift.yaml'
    path.write_text(yaml.safe_dump(cfg))
    registry=repo/'configs/registry.yaml'
    registry.write_text(yaml.safe_dump({'repository_root':'..','shifts':{
        'S1':{'config':'configs/shift.yaml','config_sha256':sha256_normalized_text(path)}}}))
    ids,proof=video_selection_from_registry(registry)
    assert ids==['v'] and proof['annotation']['path']=='annotation.json'


@pytest.mark.parametrize('name',ORDER)
def test_empty_source_crop_keeps_legal_zero_alignment(name):
    model=build(name)
    data=prototype_data()
    data[0][0]['segments']=torch.empty(0,2)
    data[0][0]['labels']=torch.empty(0,dtype=torch.long)
    out=model(*data,epoch=5)
    assert torch.isfinite(out['loss']) and out['target_gradient_norm']==0


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-group cancellation')
def test_signal_cancellation_reaps_own_child_without_wait_deadlock(tmp_path):
    import subprocess,os,signal,time
    marker=tmp_path/'child.pid'
    child_script="import os,time,pathlib;pathlib.Path(%r).write_text(str(os.getpid()));time.sleep(60)"%str(marker)
    script="""import signal,sys
from pathlib import Path
from tools import run_rq3_enhancement as m
signal.signal(signal.SIGTERM,m.stop)
m.run([sys.executable,'-c',%r],Path(%r),Path(%r))
"""%(child_script,str(tmp_path/'child.log'),str(Path.cwd()))
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]),CUDA_VISIBLE_DEVICES='')
    proc=subprocess.Popen([sys.executable,'-c',script],env=env,start_new_session=True)
    child_pid=None
    try:
        deadline=time.monotonic()+20
        while not marker.exists() and time.monotonic()<deadline and proc.poll() is None:
            time.sleep(.05)
        assert marker.exists()
        child_pid=int(marker.read_text())
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=8)==143
        assert not Path('/proc',str(child_pid)).exists()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
        if child_pid and Path('/proc',str(child_pid)).exists():
            os.kill(child_pid,signal.SIGKILL)
