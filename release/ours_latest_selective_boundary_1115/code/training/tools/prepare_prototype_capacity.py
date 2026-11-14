"""只读复审旧Qwen工件，输出原固定40/200名单的独立绑定，不重查教师。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def main(args):
    code=args.legacy_code.resolve();root=args.output_root.resolve()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    if commit!='6da910b78c17953924821ed49606dad7dce30524':raise ValueError('旧证据审计代码身份不符')
    sys.path.insert(0,str(code))
    from tools import rq3_protocol as old
    from tools.rq3_confidence import select_budget
    os.chdir('/vepfs/vlms/SADA_PLUS')
    # 同一只读文件跨shift复用精确SHA；大小/mtime变化即重新读取。
    memo={}
    def checked_hash(path):
        p=Path(path);s=p.stat();key=(str(p.resolve()),s.st_size,s.st_mtime_ns)
        if key not in memo:memo[key]=sha(p)
        return memo[key]
    old.sha256_file=checked_hash
    references=json.loads(args.references.read_text())['references']
    root.mkdir(parents=True,exist_ok=False);records={}
    for row in references:
        shift=row['shift'];entry=Path(row['entry']);preflight=entry/'preflight.json'
        cfg=json.loads(preflight.read_text());complete=json.loads((entry/'complete.json').read_text())
        old.validate_config(cfg,'uda')
        aux=cfg['rq3_auxiliary']
        # 旧complete没有preflight SHA字段；其cache SHA与原教师接口输出共同绑定证据。
        if (complete['cache_sha256']!=checked_hash(aux['cache_path']) or complete['teacher']!='qwen3_vl_8b'
                or complete['shift']!=shift or complete['strategy']!='prototype'):
            raise ValueError('旧完成记录与教师缓存身份不一致')
        if aux['strategy']!='prototype' or aux['teacher_model']!='qwen3_vl_8b' or aux['participation_fraction']!=.2:
            raise ValueError('必须复用原Qwen Prototype固定20%')
        payload,_=old.load_cache(aux['cache_path'],'qwen3_vl_8b',shift)
        if aux['selected_coordinates']!=select_budget(payload['entries']) or len(aux['selected_coordinates'])!=40:
            raise ValueError('原40/200名单不一致')
        target=json.loads((args.baseline_root/'inputs'/shift/'target_unlabeled.json').read_text())['database']
        if set(target)!=set(payload['target_video_ids']):raise ValueError('新基线目标训练成员与教师缓存不匹配')
        by_video=old.proposal_evidence(payload,aux['manifest_path'],aux['selected_coordinates'])
        selected=sum(bool(x['participation_selected']) for rows in by_video.values() for x in rows)
        if selected!=40:raise ValueError('准备后的实际参与数量不是40')
        proof=dict(schema='prototype-capacity-evidence-v1',shift=shift,teacher='qwen3_vl_8b',
            student_seed=123,candidate_cache_seed=6,participation_fraction=.2,selected_count=40,candidate_count=len(payload['entries']),
            target_gt_used=False,by_video=by_video,legacy_audit_commit=commit,
            source_artifacts={str(p):checked_hash(p) for p in (preflight,entry/'complete.json',Path(aux['cache_path']),
                Path(aux['manifest_path']),Path(aux['calibration_path']),Path(aux['input_protocol']['interface_audit_path']))})
        out=root/(shift+'.json');out.write_text(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n')
        records[shift]=dict(path=str(out),sha256=sha(out),selected_count=selected)
        print('PREPARED',shift,selected,flush=True)
    if set(records)!={'S'+str(i) for i in range(1,7)}:raise ValueError('不是精确六方向')
    (root/'audit.json').write_text(json.dumps(dict(status='passed',count=6,entries=records,
        legacy_audit_commit=commit,unique_hashed_files=len(memo)),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-code',type=Path,required=True);p.add_argument('--references',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    main(p.parse_args())
