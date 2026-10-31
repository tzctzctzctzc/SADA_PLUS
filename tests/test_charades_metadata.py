"""验证视角、词表、切分、连续时间及目标标注隔离，不静默清洗坏区间。"""
import csv
import io
import json
from pathlib import Path
from runpy import run_path
import zipfile
import pytest

M=run_path(str(Path(__file__).resolve().parents[1]/'tools/prepare_charades_inputs.py'))


def fixture(path,target_action='c000 1.25 3.75',source_action='c000 1.25 3.75'):
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('CharadesEgo/Charades_v1_mapping.txt','c000 o000 v023\n')
        z.writestr('CharadesEgo/Charades_v1_verbclasses.txt','\n'.join('v%03d %s'%(i,label.lower()) for i,label in zip(M['VERB_IDS'],M['LABEL_NAMES'])))
        for name,rows in [('train',[('s','No',source_action),('tEGO','Yes',target_action)]),
                          ('test',[('sv','No','c000 2.5 4.25'),('tvEGO','Yes','c000 3 4')])]:
            text=io.StringIO();w=csv.DictWriter(text,fieldnames=['id','egocentric','actions','length']);w.writeheader()
            for vid,view,action in rows:w.writerow(dict(id=vid,egocentric=view,actions=action,length=10))
            z.writestr('CharadesEgo/CharadesEgo_v1_'+name+'.csv',text.getvalue())


def test_native_view_splits_and_continuous_labels(tmp_path):
    path=tmp_path/'a.zip';fixture(path);out=tmp_path/'out';p=M['convert'](path,out)
    source=json.loads((out/'source.json').read_text())['database']
    target=json.loads((out/'target_unlabeled.json').read_text())['database']
    assert set(source)=={'s','sv'} and set(target)=={'tEGO'}
    assert source['s']['annotations'][0]==dict(label='TAKE',label_id=0,segment=[1.25,3.75])
    assert set(target['tEGO'])=={'subset','fps','duration'}
    assert p['label_names']==M['LABEL_NAMES'] and p['target_gt_for_cropping'] is False


def test_target_label_changes_do_not_change_training_inputs(tmp_path):
    for key,action in [('a','c000 1 2'),('b','c000 6 8')]:
        fixture(tmp_path/(key+'.zip'),target_action=action)
        M['convert'](tmp_path/(key+'.zip'),tmp_path/key)
    for filename in ['source.json','target_unlabeled.json']:
        assert (tmp_path/'a'/filename).read_bytes()==(tmp_path/'b'/filename).read_bytes()


def test_bad_source_interval_requires_explicit_policy(tmp_path):
    path=tmp_path/'a.zip';fixture(path,source_action='c000 3 2')
    with pytest.raises(ValueError,match='区间'):M['convert'](path,tmp_path/'out')


def test_explicit_clean_policy_preserves_original_for_second_evaluation(tmp_path):
    path=tmp_path/'a.zip';fixture(path,target_action='c000 8 12',source_action='c000 3 2')
    p=M['convert'](path,tmp_path/'out','clean')
    assert len(p['corrections'])==2
    source=json.loads((tmp_path/'out/source.json').read_text())['database']
    clean=json.loads((tmp_path/'out/offline_evaluation.json').read_text())['database']
    raw=json.loads((tmp_path/'out/offline_evaluation_raw.json').read_text())['database']
    assert source['s']['annotations']==[]
    assert clean['tEGO']['annotations'][0]['segment']==[8.,10.]
    assert raw['tEGO']['annotations'][0]['segment']==[8.,12.]


def test_explicit_author_policy_does_not_silently_clean(tmp_path):
    path=tmp_path/'a.zip';fixture(path,source_action='c000 3 2')
    p=M['convert'](path,tmp_path/'out','author')
    source=json.loads((tmp_path/'out/source.json').read_text())['database']
    assert source['s']['annotations'][0]['segment']==[3.,2.] and p['corrections']==[]
