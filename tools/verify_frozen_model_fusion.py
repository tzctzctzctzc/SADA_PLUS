"""校验冻结融合版本，不训练、不推理。

python -B tools/verify_frozen_model_fusion.py
提供 --run-root EVALUATION_ROOT 时，额外核验评估工件和原模型权重。
权重、数据与预测文件在外部实验存储中，仓库仅固定其SHA256。
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
RELEASE=REPO/'tools/releases/proto_trident_fusion_1105.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def verify(run_root=None):
    release=read(RELEASE)
    assert release['training_runs_for_fusion']==0 and release['seed']==123
    assert release['fusion']['score_weights']==[.5,.5] and not release['fusion']['parameter_search']
    assert sorted(release['shifts'])==['S1','S2','S3','S4','S5','S6']
    for name,digest in release['code_sha256_lf'].items():
        # Git源码以LF绑定；Windows工作区的CRLF不改变这份源码身份。
        actual=hashlib.sha256((REPO/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        assert actual==digest,'冻结源码变化: '+name
    means={name:100*sum(s['target_maps'][name] for s in release['shifts'].values())/6 for name in ('proto','trident','fusion')}
    for name,value in means.items():assert abs(value-release['results']['means_percent'][name])<1e-10
    assert abs(means['fusion']-means['proto']-release['results']['delta_vs_proto_pp'])<1e-10
    assert abs(means['fusion']-means['trident']-release['results']['delta_vs_trident_pp'])<1e-10
    assert release['results']['delta_vs_proto_pp']>=1.
    if run_root is not None:
        root=Path(run_root)
        assert sha(root/'summary.json')==release['evidence']['summary_sha256']
        assert sha(root/'final_fusion_audit.json')==release['evidence']['final_audit_sha256']
        audit=read(root/'final_fusion_audit.json');plan=read(root/'plan.json')
        assert audit['passed'] and audit['model_checkpoints_unchanged'] and audit['single_model_AP_exact']
        assert audit['fusion_independent_AP_matched'] and audit['trident_archived_arrays_exact']
        assert plan['code']['commit']==release['evaluated_code_commit'] and plan['training_runs']==0
        assert plan['probability']==release['fusion']['probability'] and plan['geometry']==release['fusion']['geometry']
        for shift,frozen in release['shifts'].items():
            entry=root/shift
            assert sha(entry/'binding.json')==frozen['binding_sha256']
            assert sha(entry/'result.json')==frozen['result_sha256']
            result=read(entry/'result.json');binding=read(entry/'binding.json')
            assert result['passed'] and result['training_runs']==0
            for file,digest in frozen['prediction_artifacts'].items():assert sha(entry/file)==digest,shift+':'+file
            for name,expected in frozen['models'].items():
                model=Path(binding[name]['entry'])
                assert sha(model/'config.json')==expected['config_sha256']
                assert sha(model/'best.pt')==expected['checkpoint_sha256']
                assert read(model/'termination.json')['decision']['best_epoch']==expected['selected_epoch']
                for domain in ('source','target'):
                    assert result['results'][domain][name]['original_AP_exact']
            for domain in ('source','target'):
                assert result['acceptance'][domain]['original_vs_single_domain_dense_exact']
                assert result['results'][domain]['fusion']['independent_AP_matched']
    return dict(passed=True,release=release['release'],means_percent=means,
        delta_vs_proto_pp=release['results']['delta_vs_proto_pp'],external_artifacts_checked=run_root is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path)
    print(json.dumps(verify(p.parse_args().run_root),ensure_ascii=False))
