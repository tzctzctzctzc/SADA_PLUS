from copy import deepcopy
import ast
import json
from pathlib import Path
import subprocess

import pytest
import torch

from tools import s2_development as dev
from tools.baseline_protocol import CODE, save, sha
from tools.run_s2_development import finalize_search
from libs.models.SADA.prototype_capacity import specification


def config(method):
    cfg=dict(seed=123,init_rand_seed=123,loader=dict(batch_size=2),
        model=dict(pl_threshold=.5,loss_weights=dict(weight_adv=2.),
                   **{'weight_level_%d'%i:v for i,v in enumerate(dev.LEVELS['repository'])}),
        baseline=dict(shift='S2',regime='uda',code={},tmpdir='/old'),opt=dict(lr=.0001,epochs=50))
    if method=='prototype':
        cfg['prototype_capacity']=dict(variant='control',specification=specification('control'),
            prepared_cache='/same',prepared_cache_sha256='frozen')
    return cfg


def test_exact_grid_and_only_declared_changes():
    assert len(dev.GRID)==6
    assert len([(n,m) for n in dev.GRID if n!=dev.REUSED for m in ('uda','prototype')])==10
    for method in ('uda','prototype'):
        old=config(method);before=deepcopy(old)
        for name in dev.GRID:
            cfg=dev.changed_config(old,name,{'commit':'new'},'/new')
            dev.assert_config(cfg,old,name,method)
            assert cfg['model']['loss_weights']['weight_adv']==2.
            assert old==before


@pytest.mark.parametrize('change',['seed','batch','lr','total_adv','width','cache','weight','extra_gt'])
def test_config_rejects_extra_changes(change):
    old=config('prototype');cfg=dev.changed_config(old,'paper_t07',{},'/new')
    if change=='seed':cfg['seed']=6
    elif change=='batch':cfg['loader']['batch_size']=4
    elif change=='lr':cfg['opt']['lr']=.0002
    elif change=='total_adv':cfg['model']['loss_weights']['weight_adv']=3.
    elif change=='width':cfg['prototype_capacity']['specification']['embedding_dim']=256
    elif change=='cache':cfg['prototype_capacity']['prepared_cache_sha256']='other'
    elif change=='weight':cfg['prototype_capacity']['specification']['target_alignment_weight']=.2
    else:cfg['target_dev_labels']='/gt'
    with pytest.raises(ValueError):dev.assert_config(cfg,old,'paper_t07','prototype')


def test_video_split_fixed_disjoint_and_no_training_gt(tmp_path):
    annotations=[dict(label_id=c,segment=[c,c+.5]) for c in range(10)]
    db={str(i):dict(subset='target_val',annotations=annotations) for i in range(50)}
    db['train']=dict(subset='target_train',annotations=annotations)
    path=tmp_path/'gt.json';save(path,dict(database=db))
    cfg=dict(baseline=dict(annotation_for_offline_evaluation=str(path)),
             tgt_val_split=['target_val'],tgt_train_split=['target_train'],src_train_split=['source_train'],src_val_split=['source_val'])
    split=dev.freeze_split(cfg)
    assert len(split['development'])==len(split['confirmation'])==25
    assert not set(split['development'])&set(split['confirmation'])
    assert 'train' not in split['development']+split['confirmation']
    assert split==dev.freeze_split(cfg)
    cfg['src_train_split']=['target_val']
    with pytest.raises(ValueError):dev.freeze_split(cfg)


def test_point_counts_real_threshold_tie_multilabel_and_background():
    p=torch.tensor([[.5,.2],[.1,.8],[.9,.1],[.1,.1],[.1,.9]])
    g=torch.tensor([[1,0],[1,1],[0,0],[1,0],[1,0]],dtype=torch.bool)
    r=dev.point_counts(p,g,.5)
    assert r==dict(valid=5,gt_foreground=4,predicted_foreground=4,correct_foreground=2,
        foreground_predicted_background=1,background_predicted_foreground=1,wrong_foreground_class=1)


def test_selection_never_prefers_weak_uda_to_get_a_big_prototype_gain():
    rows={n:dict(uda=.49,prototype=.59) for n in dev.GRID}
    rows['paper_t03']=dict(uda=.55,prototype=.56)
    rows['paper_t07']=dict(uda=.58,prototype=.57)
    r=dev.select_candidate(rows,.5)
    assert r['selected']=='paper_t03' and r['best_uda']=='paper_t07' and r['ordering_found']
    r=dev.select_candidate(rows,.60)
    assert not r['ordering_found'] and r['selected']=='paper_t07'
    rows.pop('paper_t05')
    with pytest.raises(ValueError):dev.select_candidate(rows,.5)


def test_confirmation_is_evaluated_after_immutable_selection(tmp_path,monkeypatch):
    from tools import run_s2_development as driver
    exports={m:tmp_path/'exports'/m for m in ('so','uda','prototype')}
    entries={(n,m):tmp_path/n/m for n in dev.GRID for m in ('uda','prototype')}
    save(tmp_path/'split.json',dict(development=['dev'],confirmation=['confirm']))
    calls=[]
    def score(path,split,videos):
        confirmation=videos==['confirm'];selection=tmp_path/'selection.json'
        assert selection.exists()==confirmation
        if confirmation:calls.append(sha(selection))
        return dict(map=.5 if 'so' in Path(path).parts else .6 if 'uda' in Path(path).parts else .7)
    monkeypatch.setattr(driver,'subset_score',score)
    driver.finalize_search(tmp_path,dict(development=['dev'],confirmation=['confirm']),exports,entries)
    assert len(set(calls))==1
    assert read_json(tmp_path/'confirmation.json')['no_reselection']


def read_json(path):return json.loads(path.read_text())


def test_training_functions_and_model_data_files_are_unmodified():
    base='23d553e26b80f2263bc019caa5b9100020732903'
    previous=subprocess.check_output(['git','show',base+':tools/baseline_engine.py'],cwd=CODE,text=True)
    def functions(source):
        return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(source).body
                if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name!='evaluate'}
    assert functions(previous)==functions((CODE/'tools/baseline_engine.py').read_text(encoding='utf-8'))
    files=subprocess.check_output(['git','diff',base,'--name-only','--','libs/','configs/','train_da.py'],cwd=CODE,text=True)
    assert not files.strip()


def test_observer_uses_second_head_and_original_assignment(tmp_path):
    class Model:
        def label_points(self,points,segments,labels):
            assert len(points)==1 and labels[0].tolist()==[0]
            return [torch.tensor([[1,0],[0,0]],dtype=torch.float)],[]
    observer=dev.PointObserver(['video']);observer.model=Model()
    video=[dict(video_id='video',segments=torch.tensor([[0.,1.]]),labels=torch.tensor([0]))]
    observer.before('target',video)
    mask=[torch.ones(1,1,2,dtype=torch.bool)]
    source=[torch.zeros(1,2,2)]
    target=[torch.tensor([[[10.,10.],[-10.,-10.]]])]
    pl=[torch.tensor([[[-10.,-10.],[10.,10.]]])]
    observer.heads=[(source,mask),(target,mask),(pl,mask)]
    observer.points=[[torch.zeros(2,4)]]
    observer.after('target',video);observer.finish(tmp_path/'diagnostic.json')
    row=read_json(tmp_path/'diagnostic.json')['per_level_threshold']['0/0.5']
    assert row['correct_foreground']==1 and row['background_predicted_foreground']==1
    observer.before('target',[dict(video_id='confirmation')])
    assert not observer.enabled


def test_real_sada_observer_hooks_do_not_change_predictions_or_rng(monkeypatch,tmp_path):
    from runpy import run_path
    from libs.models.SADA.meta_archs import SADA
    small=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']
    native=torch.tensor
    def cpu_copy(*args,**kwargs):
        value=native(*args,**kwargs)
        return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    cfg=small();cfg['test_cfg']['min_score']=.0
    torch.manual_seed(123);model=SADA(cfg,'cpu');model.eval()
    video=[dict(feats=torch.randn(8,64),segments=torch.tensor([[10.,35.]]),labels=torch.tensor([0]),
        video_id='real',metadata=dict(video_id='real',feat_stride=16,feat_num_frames=32,fps=30,duration=60))]
    with torch.no_grad():
        before=model(video_list_src=video,video_list_tgt=video,mode='test',domain='target',curr_epoch=20)
        rng=torch.get_rng_state().clone();state={k:v.clone() for k,v in model.state_dict().items()}
        observer=dev.PointObserver(['real']);observer.attach(model);observer.before('target',video)
        after=model(video_list_src=video,video_list_tgt=video,mode='test',domain='target',curr_epoch=20)
        observer.after('target',video);observer.finish(tmp_path/'real.json')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    for a,b in zip(before[1],after[1]):
        for key in ('segments','labels','scores'):assert torch.equal(a[key],b[key])
    assert len(read_json(tmp_path/'real.json')['per_level_threshold'])==18
