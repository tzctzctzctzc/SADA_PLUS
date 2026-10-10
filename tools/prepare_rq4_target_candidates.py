"""仅CPU：复用冻结SO目标候选，先按shift选20%，再切出双上下文。"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from tools.prepare_rq4_source_candidates import source_candidate
from tools.rq4_protocol import build_budget, file_sha, select_candidates
from tools.rq4_source_acceptance import context, probe, save


def check_input(meta, rows):
    if (meta.get('domain') != 'target' or meta.get('split') != 'train'
            or meta.get('seed') != 6 or meta.get('selection_uses_gt') is not False
            or meta.get('max_per_video') != 30 or meta.get('max_proposals') != 200
            or meta.get('selected_proposals') != len(rows) or not 0 < len(rows) <= 200):
        raise ValueError('requires frozen RQ3 seed6 target training proposal pool')
    keys = [(r['video_id'], r['candidate_id']) for r in rows]
    if len(set(keys)) != len(keys) or max(Counter(r['video_id'] for r in rows).values()) > 30:
        raise ValueError('duplicate or expanded target candidate pool')
    if not {r['video_id'] for r in rows} <= set(meta['expected_video_ids']):
        raise ValueError('target membership mismatch')


def original_candidate(row, duration, shift):
    result = source_candidate(row, duration, shift, 2)
    result['candidate_id'] = row['candidate_id']
    return result


def paired_candidates(row):
    result = []
    for scale in (2, 3):
        item = deepcopy(row)
        item['candidate_id'] += '-context' + str(scale)
        item['window'] = context(row['proposal'], row['video_duration'], scale)
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('proposals', 'videos', 'data-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.proposals.read_text().splitlines() if s.strip()]
    meta_path = Path(str(args.proposals)+'.meta.json')
    meta = json.loads(meta_path.read_text())
    check_input(meta, rows)
    provenance = meta['selection_teacher_provenance']
    binding = provenance['trusted_matched_baseline']
    shift = binding['shift']
    done = json.loads((Path(binding['root'])/'complete.json').read_text())
    cfg_path = Path(done['so'])
    cfg = json.loads(cfg_path.read_text())
    if done['shift'] != shift or done['seed'] != 6 or cfg['seed'] != 6:
        raise ValueError('SO proposal baseline identity mismatch')
    best, = (cfg_path.parent/'saved_ckpt/best_ckpt').glob('*.pt')
    if file_sha(best) != provenance['selection_teacher_checkpoint_sha256']:
        raise ValueError('SO checkpoint changed')
    annotation = Path(cfg['dataset']['json_file'])
    if not annotation.is_absolute():
        annotation = args.data_root/annotation
    # 只读取划分成员，不读取目标类别或动作区间。
    database = json.loads(annotation.read_text())['database']
    splits = {s.lower() for s in cfg['tgt_train_split']}
    members = sorted(k for k, v in database.items() if v['subset'].lower() in splits)
    if members != sorted(meta['expected_video_ids']):
        raise ValueError('target training split differs from SO proposal provenance')
    args.output.mkdir(parents=True, exist_ok=False)
    videos, pool, rejected = {}, [], []
    for row in rows:
        vid = row['video_id']
        if vid not in videos:
            video = args.videos/(vid+'.MP4')
            videos[vid] = dict(path=str(video), probe=probe(video))
        try:
            pool.append(original_candidate(row, float(videos[vid]['probe']['format']['duration']), shift))
        except ValueError as exc:
            rejected.append(dict(candidate_id=row['candidate_id'], reason=str(exc)))
    budget = build_budget(pool, shift)
    selected = select_candidates(pool, .2)
    if not selected:
        raise ValueError('empty fixed target budget')
    requests = []
    save(args.output/'budget.json', budget)
    for row in selected:
        info = videos[row['video_id']]
        n, d = map(float, info['probe']['streams'][0]['r_frame_rate'].split('/'))
        for item in paired_candidates(row):
            start, end = item['window']
            clip = args.output/(item['candidate_id']+'.mkv')
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-ss', str(start),
                '-i', info['path'], '-t', str(end-start), '-map', '0:v:0', '-an',
                '-vf', 'scale=640:-2', '-c:v', 'ffv1', '-threads', '2', str(clip)], check=True)
            if abs(float(probe(clip)['format']['duration'])-(end-start)) > 2/(n/d)+.01:
                raise ValueError('target clip duration mismatch')
            requests.append(dict(candidate=item, clip_path=str(clip), clip_sha256=file_sha(clip),
                clip_start_sec=start, clip_end_sec=end))
        print('TARGET_PAIR_READY', shift, row['candidate_id'], len(requests), flush=True)
    save(args.output/'requests.json', dict(schema='rq4-target-requests-v1', split='target_train',
        target_gt_used=False, shift=shift, seed=6, contexts=[2, 3],
        proposals_path=str(args.proposals), proposals_sha256=file_sha(args.proposals),
        metadata_path=str(meta_path), metadata_sha256=file_sha(meta_path),
        checkpoint_sha256=provenance['selection_teacher_checkpoint_sha256'],
        budget_path=str(args.output/'budget.json'), budget_file_sha256=file_sha(args.output/'budget.json'),
        budget_payload_sha256=budget['payload_sha256'], rejected_geometry=rejected,
        target_video_ids=members, videos=videos, requests=requests))
    print('TARGET_REQUESTS_READY', shift, len(selected), len(requests), flush=True)


if __name__ == '__main__':
    main()
