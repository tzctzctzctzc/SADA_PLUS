"""复用RQ3有限源域候选及真实FPN锚点，准备双上下文诊断；不导出训练门控。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from tools.rq4_protocol import file_sha, validate_candidate
from tools.rq4_source_acceptance import context, probe, save


def source_candidate(row, duration, shift, scale):
    # 只白名单读取预测；源GT字段不进入教师请求。
    levels = row['source_fpn_levels']
    if len(levels) != 1 or levels[0] not in (0, 1, 2):
        raise ValueError('expected the original strongest FPN level')
    proposal = [row['proposal_start_sec'], row['proposal_end_sec']]
    result = dict(candidate_id=f"{shift}-{row['candidate_id']}-context{scale}",
        video_id=row['video_id'], class_id=row['teacher_top1_class'], proposal=proposal,
        window=context(proposal, duration, scale), video_duration=duration,
        anchor_sec=row['absolute_center_time'], fpn_level=levels[0])
    if not 0 <= result['class_id'] < 10:
        raise ValueError('prediction outside common ten classes')
    return validate_candidate(result)


def check_budget(meta, rows):
    if (meta.get('domain') != 'source' or meta.get('split') != 'val'
            or meta.get('seed') != 6 or meta.get('selection_uses_gt') is not False
            or meta.get('max_per_video') != 30 or meta.get('max_proposals') != 200
            or meta.get('selected_proposals') != len(rows) or not 0 < len(rows) <= 200):
        raise ValueError('not the frozen RQ3 seed6 limited source budget')
    ids = [(r['video_id'], r['candidate_id']) for r in rows]
    if len(set(ids)) != len(rows) or max(Counter(r['video_id'] for r in rows).values()) > 30:
        raise ValueError('duplicate source candidate or per-video budget exceeded')
    if not {r['video_id'] for r in rows} <= set(meta['expected_video_ids']):
        raise ValueError('candidate outside source validation membership')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposals', type=Path, required=True)
    parser.add_argument('--videos', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.proposals.read_text().splitlines() if s.strip()]
    meta = json.loads(Path(str(args.proposals) + '.meta.json').read_text())
    check_budget(meta, rows)
    provenance = meta['selection_teacher_provenance']
    binding = provenance['trusted_matched_baseline']
    shift = binding['shift']
    if shift not in {f'S{i}' for i in range(1, 7)} or binding['seed'] != 6:
        raise ValueError('wrong source proposal baseline identity')
    baseline = Path(binding['root'])
    completed = json.loads((baseline/'complete.json').read_text())
    cfg = json.loads(Path(completed['so']).read_text())
    if completed['seed'] != 6 or completed['shift'] != shift or cfg['seed'] != 6:
        raise ValueError('source proposal SO baseline seed/shift mismatch')
    best, = (Path(completed['so']).parent/'saved_ckpt/best_ckpt').glob('*.pt')
    if file_sha(best) != provenance['selection_teacher_checkpoint_sha256']:
        raise ValueError('source proposal checkpoint SHA mismatch')
    annotation = Path(cfg['dataset']['json_file'])
    if not annotation.is_absolute():
        annotation = args.data_root/annotation
    database = json.loads(annotation.read_text())['database']
    subsets = {s.lower() for s in cfg['src_val_split']}
    members = sorted(k for k, v in database.items() if v['subset'].lower() in subsets)
    if members != sorted(meta['expected_video_ids']):
        raise ValueError('source validation membership differs from baseline')
    args.output.mkdir(parents=True, exist_ok=False)
    requests, videos, rejected = [], {}, []
    for row in rows:
        vid = row['video_id']
        video = args.videos/(vid + '.MP4')
        if vid not in videos:
            videos[vid] = dict(path=str(video), probe=probe(video))
        info = videos[vid]['probe']
        duration = float(info['format']['duration'])
        numerator, denominator = map(float, info['streams'][0]['r_frame_rate'].split('/'))
        fps = numerator/denominator
        try:
            pair = [source_candidate(row, duration, shift, s) for s in (2, 3)]
        except ValueError as exc:
            rejected.append(dict(candidate_id=row['candidate_id'], reason=str(exc)))
            continue
        for candidate in pair:
            window = candidate['window']
            clip = args.output/(candidate['candidate_id'] + '.mkv')
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-ss', str(window[0]),
                '-i', str(video), '-t', str(window[1]-window[0]), '-map', '0:v:0', '-an',
                '-vf', 'scale=640:-2', '-c:v', 'ffv1', '-threads', '2', str(clip)], check=True)
            actual = float(probe(clip)['format']['duration'])
            if abs(actual-(window[1]-window[0])) > 2/fps+.01:
                raise ValueError('clip duration differs from requested context')
            requests.append(dict(candidate=candidate, clip_path=str(clip), clip_sha256=file_sha(clip),
                clip_start_sec=window[0], clip_end_sec=window[1]))
        print('SOURCE_CANDIDATE_READY', shift, row['candidate_id'], len(requests), flush=True)
    if not requests:
        raise ValueError('no geometrically valid source candidates')
    save(args.output/'requests.json', dict(diagnostic_only=True, split='source_validation',
        source_subsets=sorted(subsets), shift=shift, seed=6, contexts=[2, 3],
        prediction_file_sha256=file_sha(args.proposals),
        prediction_metadata_sha256=file_sha(str(args.proposals)+'.meta.json'),
        checkpoint_sha256=provenance['selection_teacher_checkpoint_sha256'],
        anchor_note='original SO prediction FPN anchor; never a proposal midpoint',
        budget=dict(max_per_video=30, max_per_shift=200, input_candidates=len(rows)),
        rejected_geometry=rejected, videos=videos, requests=requests))
    print('SOURCE_REQUESTS_READY', shift, len(requests), 'rejected_geometry', len(rejected), flush=True)


if __name__ == '__main__':
    main()
