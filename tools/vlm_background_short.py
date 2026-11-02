"""以元数据预选短验证；复用已审真实帧和成功请求，保留长队列现场。"""
import copy
import os
from pathlib import Path
from tools.baseline_protocol import read, save, sha
from tools.vlm_background_protocol import SPEC, digest, verify_response

ORIGIN = Path('/vepfs/vlms/vlm_background_pilot3_4efeaad_1103')


def select_first(groups, count=8):
    result = {}
    for name, group in groups.items():
        keys = sorted(group['candidate_map'], key=lambda k: digest(['vlm-bg-short-v1', name, k, group['candidate_map'][k]]))[:count]
        result[name] = {k: group['candidate_map'][k] for k in keys}
    return result


def select(groups, count=8):
    first = select_first(groups, count)
    result = {}
    for name, group in groups.items():
        eligible = [k for k in group['candidate_map'] if k not in first[name]]
        keys = sorted(eligible, key=lambda k: digest(['vlm-bg-replication-v1', name, k, group['candidate_map'][k]]))[:count]
        assert len(keys) == count
        result[name] = {k:group['candidate_map'][k] for k in keys}
    return result


def prepare_short(root):
    pause = read(ORIGIN/'budget_pause_receipt.json')
    assert pause['all_owned_processes_exited'] and not pause['full_queue_automatic_restart_authorized']
    for pid in pause['identities']:
        stat = Path('/proc')/pid/'stat'
        if stat.exists():
            # 不因PID复用把外部当旧worker，也不读外部命令环境。
            fields = stat.read_text().rsplit(')', 1)[1].split()
            assert fields[0] == 'Z' or fields[19] != pause['identities'][pid]['starttime']
    original_ready = read(ORIGIN/'prepare_complete.json')
    audit = read(ORIGIN/'independent_input_audit.json')
    assert audit['passed'] and audit['prepare_complete_sha256'] == sha(ORIGIN/'prepare_complete.json')
    for path, expected in original_ready['artifacts'].items(): assert sha(path) == expected
    original_groups = read(ORIGIN/'groups.json')
    mappings = select(original_groups)
    rows = {r['candidate_id']: r for r in read(ORIGIN/'inference_manifest.json')['cases']}
    cases = {}; groups = {}
    for name, mapping in mappings.items():
        job = copy.deepcopy(original_groups[name]); entry = root/'groups'/name
        entry.mkdir(parents=True, exist_ok=False)
        os.link(job['original'], entry/'original.pt')
        config = ORIGIN/'groups'/name/'config.json'
        assert sha(config) == job['config_sha256']
        os.link(config, entry/'config.json')
        job.update(candidate_map=mapping, original=str(entry/'original.pt'))
        groups[name] = job
        for cid in mapping.values(): cases.setdefault(cid, rows[cid])
    first = next(r for r in cases.values() if r['observable'])
    reused = {}; withheld = []
    # 首个候选四种请求重新真实核验；其余成功响应原样硬链接。
    # 不读预测选项或GT决定样本/是否复用，仅按完整request身份。
    for path in sorted((ORIGIN/'responses').glob('*.json')):
        response = read(path); verify_response(response['request'], response)
        if response['request']['times'] == first['times']:
            withheld.append(str(path)); continue
        dest = root/'responses'/path.name; dest.parent.mkdir(exist_ok=True)
        os.link(path, dest); reused[str(path)] = sha(path)
    os.link(ORIGIN/'model_identity.json', root/'model_identity.json')
    save(root/'groups.json', groups)
    save(root/'inference_manifest.json', dict(spec=SPEC,cases=list(cases.values()),GT_in_inference_input=False))
    save(root/'short_scope.json', dict(origin=str(ORIGIN),origin_prepare_sha256=sha(ORIGIN/'prepare_complete.json'),
        origin_input_audit_sha256=sha(ORIGIN/'independent_input_audit.json'),pause_receipt_sha256=sha(ORIGIN/'budget_pause_receipt.json'),
        predetermined_selection=mappings,selection_uses_GT_or_responses=False,reused_response_sha256=reused,
        first_case_acceptance_responses_withheld=withheld,all_unselected_predictions_retained=True,
        original_cases=original_ready['cases'],selected_cases=len(cases),preparation_repeated=False))
    artifacts=[root/'groups.json',root/'inference_manifest.json',root/'model_identity.json',root/'short_scope.json']
    save(root/'prepare_complete.json',dict(cases=len(cases),groups=len(groups),observable=sum(r['observable'] for r in cases.values()),
        unique_physical_videos=len({r['video_id'] for r in cases.values()}),artifacts={str(p):sha(p) for p in artifacts},
        original_prediction_inputs=original_ready['original_prediction_inputs'],no_detector_forward_or_training=True))
