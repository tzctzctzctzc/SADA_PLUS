"""第三轮：合成夹具只检查证据链拒绝逻辑，不冒充真实源拟合或GPU验收。"""
from copy import deepcopy
from pathlib import Path
import pytest
import torch

from libs.models.SADA.endpoint_risk import SCHEMA,POLICY
from tools.baseline_protocol import read,save,sha
from tools.baseline_engine import tensor_digest
from tools.endpoint_risk_protocol import (calibrate,audit_certificate,promote_checkpoint,
    validate_installed_certificate,assert_formal_ready,gate_audit)


def evidence(tmp_path):
    rows=[]
    for video in ('fit1','fit2'):
        rows.append(dict(video_id=video,candidate_id='fit',level=0,duration=10.,
            student_sigma=[1.,1.],teacher_sigma=[1.,1.],student_error=[1.,1.],teacher_error=[1.,1.]))
    for video in ('audit1','audit2','audit3'):
        for name,ss,ts,se,te in [('good',2.,.5,4.,1.),('bad',.5,2.,1.,3.)]:
            rows.append(dict(video_id=video,candidate_id=name,level=0,duration=10.,
                student_sigma=[ss,ss],teacher_sigma=[ts,ts],student_error=[se,se],teacher_error=[te,te]))
    code=dict(commit='test-fixture-only',clean=True)
    cfg=dict(baseline=dict(code=code),endpoint_risk=dict(mode='relative'))
    state=dict(schema=SCHEMA,config=cfg,epoch=5,history=[[5,.5]],state_dict={'x':torch.tensor([3.])},
        ema_state_dict={'x':torch.tensor([2.])},optimizers=[{'state':torch.tensor([1.])}],
        rng={'test_only':torch.get_rng_state()},endpoint_risk=dict(calibration=None,
            optimizer={'state':{'x':torch.tensor([.1])}},scheduler={'step':10},generator=torch.get_rng_state()))
    snapshot=tmp_path/'source_checkpoint.pt';torch.save(state,snapshot)
    cert=calibrate(rows,['fit1','fit2'],['audit1','audit2','audit3'],['train'],0.)
    save(tmp_path/'source_rows.json',rows)
    cert.update(code=code,actual_student_and_ema=True,models_unchanged=True,
        source_rows_sha256=sha(tmp_path/'source_rows.json'),checkpoint=str(snapshot),checkpoint_sha256=sha(snapshot))
    for key in ('source_manifest','source_pool','requests'):
        p=tmp_path/(key+'.json');save(p,{'fixture':key});cert[key]=str(p);cert[key+'_sha256']=sha(p)
    feature=tmp_path/'feature.bin';feature.write_bytes(b'unit-test-feature')
    cert['feature_inputs']={str(feature):sha(feature)}
    path=tmp_path/'calibration.json';save(path,cert)
    return cfg,state,cert,path


def installed(tmp_path):
    cfg,state,cert,path=evidence(tmp_path)
    output=tmp_path/'promoted.pt';promote_checkpoint(cert['checkpoint'],path,output)
    return cfg,torch.load(output,map_location='cpu'),path,output


def synthetic_acceptance(tmp_path,cfg,saved,checkpoint):
    # 不执行CUDA；只给validate函数提供文件夹具，绝不输出到实验目录作为真实凭证。
    lease=tmp_path/'lease.json';guard=tmp_path/'guard.json'
    save(lease,dict(commit=cfg['baseline']['code']['commit'],observed={'uuid':'fixture-gpu'},minimum_free_mib=12288))
    save(guard,dict(lease=str(lease),lease_sha256=sha(lease),observed={'uuid':'fixture-gpu','free_mib':16384}))
    report=dict(status='passed',device='cuda',code=cfg['baseline']['code'],config_sha256=tensor_digest(cfg),
        calibration_sha256=saved['endpoint_risk']['calibration']['calibration_sha256'],
        target_gt_zero=True,risk_head_updated=True,teacher_detached=True,rng_roundtrip=True,
        pseudo_gradient_nonzero=True,checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        gpu_guard=str(guard),gpu_guard_sha256=sha(guard))
    report_path=tmp_path/'synthetic_gpu_report.json';save(report_path,report)
    return dict(report,reports={str(report_path):sha(report_path)})


def test_promote_only_adds_evidence_without_changing_any_training_state(tmp_path):
    cfg,original,cert,path=evidence(tmp_path);before=sha(cert['checkpoint'])
    output=tmp_path/'promoted.pt';proof=promote_checkpoint(cert['checkpoint'],path,output)
    promoted=torch.load(output,map_location='cpu');extra=promoted['endpoint_risk']['calibration']
    promoted['endpoint_risk']['calibration']=None
    assert extra['status']=='passed' and proof['all_training_state_unchanged']
    assert tensor_digest(promoted)==tensor_digest(original) and sha(cert['checkpoint'])==before
    validate_installed_certificate(extra)
    with pytest.raises(ValueError):promote_checkpoint(cert['checkpoint'],path,output)
    with pytest.raises(ValueError):promote_checkpoint(cert['checkpoint'],path,cert['checkpoint'])


def test_source_certificate_recomputed_not_trusted_status(tmp_path):
    cfg,state,cert,path=evidence(tmp_path);assert audit_certificate(path)['status']=='passed'
    cert['factors']['0']['student'][0]=.001;save(path,cert)
    with pytest.raises(ValueError):audit_certificate(path)


@pytest.mark.parametrize('change',['rows','feature','checkpoint','source_manifest','requests'])
def test_changed_source_inputs_rejected(tmp_path,change):
    cfg,state,cert,path=evidence(tmp_path)
    p=tmp_path/'source_rows.json' if change=='rows' else Path(next(iter(cert['feature_inputs']))) if change=='feature' else Path(cert[change])
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):audit_certificate(path)


def test_installed_factors_cannot_differ_from_bound_file(tmp_path):
    cfg,saved,path,checkpoint=installed(tmp_path)
    saved['endpoint_risk']['calibration']['factors']['0']['teacher'][0]=99.
    with pytest.raises(ValueError):validate_installed_certificate(saved['endpoint_risk']['calibration'])


def test_mutable_origin_latest_not_used_as_source_certificate(tmp_path):
    cfg,state,cert,path=evidence(tmp_path)
    origin=tmp_path/'mutable_latest.pt';torch.save(state,origin)
    cert['origin_checkpoint']=str(origin);save(path,cert)
    origin.write_bytes(b'replaced by later epoch')
    assert audit_certificate(path)['checkpoint']==str(tmp_path/'source_checkpoint.pt')


def test_cpu_report_never_formally_launches(tmp_path):
    cfg,saved,path,checkpoint=installed(tmp_path)
    acceptance=synthetic_acceptance(tmp_path,cfg,saved,checkpoint);acceptance['device']='cpu'
    with pytest.raises(ValueError,match='真实GPU'):assert_formal_ready(cfg,saved,acceptance)


@pytest.mark.parametrize('flag',['target_gt_zero','risk_head_updated','teacher_detached','rng_roundtrip','pseudo_gradient_nonzero'])
def test_missing_acceptance_check_rejected(tmp_path,flag):
    cfg,saved,path,checkpoint=installed(tmp_path)
    acceptance=synthetic_acceptance(tmp_path,cfg,saved,checkpoint)
    assert_formal_ready(cfg,saved,acceptance)  # 仅字段验证的正夹具，并非GPU通过。
    acceptance[flag]=False
    with pytest.raises(ValueError):assert_formal_ready(cfg,saved,acceptance)


def test_acceptance_cannot_rename_report_flags_or_omit_report(tmp_path):
    cfg,saved,path,checkpoint=installed(tmp_path)
    acceptance=synthetic_acceptance(tmp_path,cfg,saved,checkpoint)
    acceptance['invented_field']=True
    with pytest.raises(ValueError):assert_formal_ready(cfg,saved,acceptance)
    acceptance=synthetic_acceptance(tmp_path,cfg,saved,checkpoint);acceptance['reports']={}
    with pytest.raises(ValueError):assert_formal_ready(cfg,saved,acceptance)


@pytest.mark.parametrize('change',['report','checkpoint','guard','lease','config'])
def test_gpu_acceptance_binding_rejects_changed_artifact(tmp_path,change):
    cfg,saved,path,checkpoint=installed(tmp_path)
    acceptance=synthetic_acceptance(tmp_path,cfg,saved,checkpoint)
    if change=='config':cfg['endpoint_risk']['changed']=True;saved['config']=cfg
    else:
        p={'report':next(iter(acceptance['reports'])),'checkpoint':checkpoint,
           'guard':acceptance['gpu_guard'],'lease':tmp_path/'lease.json'}[change]
        p=Path(p);p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):assert_formal_ready(cfg,saved,acceptance)


def test_source_gate_uses_same_semantic_and_per_side_visibility_support():
    rows=[dict(video_id='v',candidate_id='a',level=0,duration=10.,student_sigma=[2.,2.],
        teacher_sigma=[1.,1.],student_error=[4.,4.],teacher_error=[1.,1.],allowed=[True,False])]
    factors={'0':dict(available=[True,True],student=[1.,1.],teacher=[1.,1.])}
    report=gate_audit(rows,factors,0.)
    assert report['available_endpoints']==1 and report['selected_endpoints']==1
    rows[0]['allowed']=[False,False]
    assert gate_audit(rows,factors,0.)['selected_endpoints']==0


def test_old_checkpoint_rejected_before_new_source_output(tmp_path):
    from tools.audit_endpoint_risk_source import collect
    old=tmp_path/'old.pt';torch.save(dict(schema='old',config={}),old)
    with pytest.raises(ValueError,match='真实训练过'):collect(old,tmp_path/'pool',tmp_path/'output',0.,'cpu',{})
    assert not (tmp_path/'output').exists()


def test_random_control_requires_actual_nonidentity_exchange(tmp_path):
    cfg,saved,path,checkpoint=installed(tmp_path)
    cfg['endpoint_risk']['mode']='random';saved['config']=cfg
    torch.save(saved,checkpoint)
    acceptance=synthetic_acceptance(tmp_path,cfg,saved,checkpoint)
    with pytest.raises(ValueError,match='random_gate_actually_changed'):assert_formal_ready(cfg,saved,acceptance)
