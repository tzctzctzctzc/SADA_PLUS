"""发布审计完成的12项基线：只导出文本结果和证据，不复制视频/特征/权重。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TRAINING_COMMIT = 'b9d4429848bd679372dd968868abac5309839b85'
MAPPING = {'S1':'S3','S2':'S1','S3':'S4','S4':'S2','S5':'S5','S6':'S6'}
EXPECTED = {(s,r) for s in MAPPING for r in ('so','uda')}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def export(root, training_code, destination):
    final=read(root/'final_audit.json')
    if final.get('status')!='passed' or final.get('count')!=12:
        raise ValueError('仅发布12项完整终审结果，不把部分结果冒充完成')
    if {(x['shift'],x['regime']) for x in final['results']}!=EXPECTED:
        raise ValueError('最终身份不是精确六方向SO/UDA')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=training_code,text=True).strip()
    if actual!=TRAINING_COMMIT:raise ValueError('训练代码提交不符')
    sys.path.insert(0,str(training_code))
    from tools.baseline_protocol import audit_entry,audit_pair,sha
    os.chdir('/vepfs/vlms/SADA_PLUS')
    # 所有前置核验完成后才创建新发布目录；失败不覆盖旧结果。
    if destination.exists():raise FileExistsError(destination)
    rows=[];evidence=[]
    for internal in MAPPING:
        for regime in ('so','uda'):
            entry=root/'formal'/internal/regime
            complete=read(entry/'complete.json')
            if complete['code']['commit']!=TRAINING_COMMIT:raise ValueError('混入其他代码的结果')
            record=root/'monitor/audits'/(internal+'_'+regime+'_b9d4429.json')
            if not record.exists() or read(record)['complete_sha256']!=digest(entry/'complete.json'):
                verified=audit_entry(entry)
                verified['audited_at']=datetime.now(timezone.utc).isoformat()
                write(record,verified)
            else:
                verified=read(record)
            target=read(entry/'evaluation/target_val.json')
            source=read(entry/'evaluation/source_val.json')
            if verified['map']!=target['map_0.1_to_0.5']:
                raise ValueError('审计数值与目标评估文件不同')
            for path,expected_hash in complete['artifacts'].items():
                if Path(path).suffix in ('.json','.jsonl') and sha(path)!=expected_hash:
                    raise ValueError('文本评估/配置证据改变: '+path)
            stop=read(entry/'termination.json')['decision']
            cfg=read(entry/'config.json')
            rows.append(dict(internal_id=internal,paper_scenario=MAPPING[internal],regime=regime,
                target_map=target['map_0.1_to_0.5'],target_map_by_tiou=target['map_by_tiou'],
                source_map=source['map_0.1_to_0.5'],best_epoch=stop['best_epoch'],
                stop_epoch=stop['epoch'],stop_reason=stop['reason'],
                batch_per_domain=cfg['loader']['batch_size'],seed=cfg['seed']))
            evidence.append(dict(internal_id=internal,regime=regime,
                complete_sha256=digest(entry/'complete.json'),
                best_checkpoint_sha256=complete['artifacts'][str(entry/'best.pt')],
                latest_checkpoint_sha256=complete['artifacts'][str(entry/'latest.pt')],
                target_eval_sha256=digest(entry/'evaluation/target_val.json'),
                independent_audit=verified))
        audit_pair(root/'formal'/internal/'so',root/'formal'/internal/'uda')
    destination.mkdir(parents=True)
    selected=('config.json','runtime_config.json','metrics.jsonl','termination.json','complete.json',
              'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')
    for internal in MAPPING:
        for regime in ('so','uda'):
            entry=root/'formal'/internal/regime
            for name in selected:
                out=destination/'entries'/internal/regime/name;out.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(entry/name,out)
            record=root/'monitor/audits'/(internal+'_'+regime+'_b9d4429.json')
            shutil.copyfile(record,destination/'entries'/internal/regime/'independent_audit.json')
        write(destination/'entries'/internal/'pair_audit.json',
              audit_pair(root/'formal'/internal/'so',root/'formal'/internal/'uda'))
    for name in ('final_audit.json','plan.json','acceptance.json','inputs/identity.json','inputs/feature_inventory.json'):
        out=destination/'provenance'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,out)
    for name in ('scenario_mapping_audit.json','membership_correction.json','paper_baseline_review.json'):
        path=root/'monitor'/name
        if path.exists():shutil.copyfile(path,destination/'provenance'/name)
    pairs=[]
    for internal in MAPPING:
        so=next(x for x in rows if x['internal_id']==internal and x['regime']=='so')
        uda=next(x for x in rows if x['internal_id']==internal and x['regime']=='uda')
        pairs.append(dict(internal_id=internal,paper_scenario=MAPPING[internal],
                          so_percent=so['target_map']*100,uda_percent=uda['target_map']*100,
                          delta_percentage_points=(uda['target_map']-so['target_map'])*100))
    means={regime:sum(x['target_map'] for x in rows if x['regime']==regime)/6 for regime in ('so','uda')}
    summary=dict(schema='sada-baseline-results-release-v1',status='complete',count=12,
        exported_at_utc=datetime.now(timezone.utc).isoformat(),training_code_commit=TRAINING_COMMIT,
        metric='target mAP averaged over tIoU 0.1,0.2,0.3,0.4,0.5',raw_map_unit='0_to_1',
        results=rows,pairs=pairs,means=means,delta_percentage_points=(means['uda']-means['so'])*100,
        source_experiment_root=str(root),evidence=evidence)
    write(destination/'results.json',summary)
    with (destination/'results.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['internal_id','paper_scenario','regime','mAP_0.1_percent','mAP_0.2_percent',
                'mAP_0.3_percent','mAP_0.4_percent','mAP_0.5_percent','mAP_average_percent',
                'source_mAP_average_percent','best_epoch','stop_epoch','stop_reason','batch_per_domain','seed']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            item={k:row[k] for k in ('internal_id','paper_scenario','regime','best_epoch','stop_epoch',
                                     'stop_reason','batch_per_domain','seed')}
            item.update({'mAP_%.1f_percent'%(i/10):value*100
                         for i,value in enumerate(row['target_map_by_tiou'],1)})
            item.update(mAP_average_percent=row['target_map']*100,
                        source_mAP_average_percent=row['source_map']*100)
            writer.writerow(item)
    with (destination/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    cfg=read(root/'formal/S1/so/config.json')
    write(destination/'protocol.json',dict(schema='sada-controlled-baseline-publication-v1',
        training_code_commit=TRAINING_COMMIT,official_code_commit=cfg['baseline']['code']['official_commit'],
        training_code_branch='exp/sada-baseline-earlystop-1029',results_branch='results/sada-baseline-1029',
        seed=6,batch_per_domain=4,uda_joint_batch=8,source_validation_earlystop=cfg['baseline']['policy'],
        data_identity_sha256=digest(root/'inputs/identity.json'),
        annotation_sha256=cfg['baseline']['annotation_sha256'],scenario_mapping=MAPPING,
        dependencies={p:importlib.metadata.version(p) for p in ('torch','numpy','pandas','einops')},
        notes=['受控基线，非原论文原始划分/训练协议的逐项复现',
               '内部S1-S4编号与论文不同；对外对照使用paper_scenario映射',
               '当前冻结划分版本已由用户确认，后续方法须使用相同划分与协议',
               'S1-S5保留发布配置，S6按论文补全并明确标注受控重建',
               '不与旧batch2或混合停止预算的结果直接计算严格配对增益',
               '审计清单target_train包装键错误的修正仅保存在provenance/membership_correction.json；训练输入未改变',
               '未上传视频、预提取特征、VLM缓存、模型权重、SSH凭据或外部进程明细'],
        reproduction=dict(checkout=TRAINING_COMMIT,
            command='python -B tools/run_sada_baseline.py --data-root /path/to/SADA_PLUS --gpus 0,1,2,3 --output-root /new/unique/result/root',
            requirements='准备相同SHA的数据/SlowFast特征、官方NMS扩展和对应Python环境；从代码目录启动')))
    # 保留Linux原始证据字节，避免Windows Git换行转换破坏SHA。
    (destination/'.gitattributes').write_bytes(b'* -text\n*.csv whitespace=cr-at-eol\n')
    files={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()}
    write(destination/'file_manifest.json',dict(hash='sha256',files=files))
    return dict(destination=str(destination),count=12,files=len(files)+1,means=means,
                delta_percentage_points=summary['delta_percentage_points'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training-code',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.root.resolve(),args.training_code.resolve(),args.destination.resolve())))
