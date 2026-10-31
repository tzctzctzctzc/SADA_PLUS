"""把官方Charades-Ego标注转换为统一TAL接口；训练目标清单没有annotations。"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

VERB_IDS=(23,16,18,29,8,22,12,4,19,9)
LABEL_NAMES=('TAKE','PUT','SIT','WALK','HOLD','STAND','OPEN','DRINK','SMILE','LAUGH')


def convert(archive,output,interval_policy='strict'):
    if interval_policy not in ('strict','author','clean'):raise ValueError('未知标注协议')
    output.mkdir(parents=True,exist_ok=False)
    corrections=[];raw_database={}
    with zipfile.ZipFile(archive) as z:
        prefix=next(n.rsplit('/',1)[0]+'/' for n in z.namelist() if n.endswith('/Charades_v1_mapping.txt'))
        mapping={line.split()[0]:int(line.split()[2][1:]) for line in z.read(prefix+'Charades_v1_mapping.txt').decode().splitlines() if line.strip()}
        verbs={int(line.split()[0][1:]):line.split()[1].upper() for line in z.read(prefix+'Charades_v1_verbclasses.txt').decode().splitlines() if line.strip()}
        if tuple(verbs[i] for i in VERB_IDS)!=LABEL_NAMES:raise ValueError('官方verb映射与冻结类别顺序不符')
        database={}
        for split,filename in [('training','CharadesEgo_v1_train.csv'),('validation','CharadesEgo_v1_test.csv')]:
            for row in csv.DictReader(io.StringIO(z.read(prefix+filename).decode())):
                vid=row['id'];ego='EGO' in vid
                if ego != (row['egocentric'].lower()=='yes'):raise ValueError('第一/第三人称身份不符')
                if vid in database:raise ValueError('官方train/test视频交叉')
                annotations=[];raw_annotations=[]
                for action in row['actions'].split(';'):
                    if not action.strip():continue
                    cls,start,end=action.split();verb=mapping[cls]
                    if verb not in VERB_IDS:continue
                    start,end=float(start),float(end)
                    label=VERB_IDS.index(verb)
                    raw_annotations.append(dict(label=LABEL_NAMES[label],label_id=label,segment=[start,end]))
                    if interval_policy=='strict' and not 0<=start<end<=float(row['length'])+.1:
                        raise ValueError('官方时间区间越界')
                    if interval_policy=='clean':
                        a,b=max(0.,start),min(float(row['length']),end)
                        if b<=a:
                            corrections.append(dict(video=vid,split=split,original=[start,end],result=None,reason='nonpositive_or_outside'))
                            continue
                        if (a,b)!=(start,end):corrections.append(dict(video=vid,split=split,original=[start,end],result=[a,b],reason='clipped_to_video'))
                        start,end=a,b
                    annotations.append(dict(label=LABEL_NAMES[label],label_id=label,segment=[start,end]))
                annotations.sort(key=lambda x:x['segment'][0])
                database[vid]=dict(subset=('ego_' if ego else 'third_')+split,
                    fps=24.,duration=float(row['length']),annotations=annotations)
                raw_database[vid]=dict(database[vid],annotations=sorted(raw_annotations,key=lambda x:x['segment'][0]))
    source={v:x for v,x in database.items() if x['subset'].startswith('third_')}
    target={v:{k:x[k] for k in ('subset','fps','duration')} for v,x in database.items() if x['subset']=='ego_training'}
    assert not set(source)&set(target)
    for name,db in [('source.json',source),('target_unlabeled.json',target),('offline_evaluation.json',database),
                    ('offline_evaluation_raw.json',raw_database)]:
        (output/name).write_text(json.dumps(dict(database=db),sort_keys=False))
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.json')}
    proof=dict(status='passed',schema='charades-ego-third-to-first-inputs-v1',label_names=LABEL_NAMES,
        interval_policy=interval_policy,corrections=corrections,
        evaluation_protocol='clean_only' if interval_policy=='clean' else interval_policy,
        raw_annotations_role='provenance_only_not_additional_evaluation',
        official_verb_ids=VERB_IDS,annotation_archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        counts={s:sum(x['subset']==s for x in database.values()) for s in
            ('third_training','third_validation','ego_training','ego_validation')},
        target_annotations_present=False,target_gt_for_cropping=False,feat_stride=8,num_frames=32,fps=24,
        files=files,video_pairs_not_used_as_target_supervision=True)
    (output/'identity.json').write_text(json.dumps(proof,sort_keys=True));return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval-policy',choices=['strict','author','clean'],default='strict')
    a=p.parse_args();print(json.dumps(convert(a.annotations,a.output,a.interval_policy)))
