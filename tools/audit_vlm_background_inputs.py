"""独立只读输入审计：重选视频、逐候选核原预测、按真实PTS重算采样。"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def audit(root):
    import av
    import numpy as np
    import torch
    done = read(root / 'prepare_complete.json')
    for path, expected in done['artifacts'].items():
        assert sha(path) == expected
    manifest = read(root / 'inference_manifest.json')
    assert manifest['GT_in_inference_input'] is False
    rows = manifest['cases']
    by_id = {r['candidate_id']: r for r in rows}
    assert len(by_id) == len(rows) == done['cases']
    groups = read(root / 'groups.json')
    assert set(groups) == {f'S{s}_{d}' for s in range(1, 7) for d in ('source_val', 'target_train')}
    source = Path('/vepfs/vlms/hard_background_pilots24_8e80d74_1103')
    memberships = 0
    grouped = {}
    for name, job in groups.items():
        path = source / job['shift'] / (job['domain'] + '_predictions.pt')
        assert sha(path) == done['original_prediction_inputs'][str(path)]
        original_rows = torch.load(path, map_location='cpu', weights_only=False)['conditional']
        eligible = [r for r in original_rows if bool((r['scores'] >= .3).any())]
        selected = min(eligible, key=lambda r: digest(['vlm-bg-three-v1', job['shift'], job['domain'], r['video']]))
        assert selected['video'] == job['video']
        assert sha(job['original']) == job['original_sha256']
        saved = torch.load(job['original'], map_location='cpu', weights_only=False)
        for key in ('segments', 'scores', 'labels'):
            assert torch.equal(saved[key], selected[key]), (name, key)
        indices = selected['scores'].ge(.3).nonzero().flatten().tolist()
        assert set(job['candidate_map']) == {str(i) for i in indices}
        for i in indices:
            row = by_id[job['candidate_map'][str(i)]]
            segment = selected['segments'][i].tolist()
            label = int(selected['labels'][i])
            assert row['video_id'] == job['video'] and row['proposal'] == segment and row['label'] == label
            assert 0 <= label < 10
            assert row['view_id'] == digest([row['video_id'], segment])
            assert row['candidate_id'] == digest([row['view_id'], label])
        memberships += len(indices)
        grouped[name] = {'video': job['video'], 'high_score_memberships': len(indices), 'all_arrays_exact': True}
    videos = {}
    for video in sorted({r['video_id'] for r in rows}):
        index = read(root / 'video_indices' / (video + '.json'))
        path = Path(index['path']); stat = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (index['size'], index['mtime_ns'])
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            tb = float(stream.time_base)
            pts = sorted(packet.pts for packet in container.demux(stream) if packet.pts is not None)
        assert digest(pts) == index['pts_sha256'] and tb == index['time_base']
        assert len(pts) == len(set(pts)) == index['frame_count']
        times = np.asarray(pts, dtype=np.float64) * tb
        video_rows = [r for r in rows if r['video_id'] == video]
        for row in video_rows:
            assert set(row) == {'candidate_id','view_id','video_id','proposal','label','observable','indices','times','visible_interval','inside_unique','frame_records'}
            start, end = row['proposal']; a = max(start, times[0]); b = min(end, times[-1])
            assert row['visible_interval'] == [a, b]
            if b <= a:
                assert not row['observable'] and row['indices'] == []
                continue
            pad = max(.5, .25 * (end-start))
            left = max(times[0], a-pad); right = min(times[-1], b+pad)
            requested = [left+(a-left)*(i+.5)/2 for i in range(2)]
            requested += [a+(b-a)*(i+.5)/12 for i in range(12)]
            requested += [b+(right-b)*(i+.5)/2 for i in range(2)]
            # 独立全数组argmin，同距时取最早帧。
            expected = [int(np.argmin(np.abs(times-t))) for t in requested]
            assert row['indices'] == expected
            assert row['times'] == times[expected].tolist()
            unique = len({i for i in expected[2:14] if a <= times[i] < b})
            visible = start >= times[0] and end <= times[-1] and unique >= 4
            assert row['inside_unique'] == unique and row['observable'] == bool(visible)
            assert len(row['frame_records']) == (16 if visible else 0)
            for i, frame in zip(expected, row['frame_records']):
                assert frame == index['sampled_frames'][str(i)]
                assert frame['pts'] == pts[i] and frame['time_sec'] == times[i]
        # 每视频固定一帧精确重解码；其余帧核PTS/元数据，不冒称逐帧像素复核。
        keys = sorted(map(int, index['sampled_frames']))
        chosen = keys[len(keys)//2]; record = index['sampled_frames'][str(chosen)]
        assert sha(record['path']) == record['sha256']
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(pts[chosen], stream=stream, backward=True, any_frame=False)
            found = False
            for frame in container.decode(stream):
                if frame.pts == pts[chosen]:
                    assert hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest() == record['raw_pixel_sha256']
                    found = True; break
                if frame.pts > pts[chosen]: break
            assert found
        videos[video] = {'cases': len(video_rows), 'all_requested_PTS_recomputed': True, 'exact_pixel_sample_index': chosen}
    report = dict(passed=True, groups=grouped, videos=videos, memberships=memberships, cases=len(rows),
        observable=sum(r['observable'] for r in rows), target_val_read=False, GT_read=False,
        all_original_arrays_exact=True, frame_pixel_scope='one deterministic frame per video; all sample indices independently recomputed',
        prepare_complete_sha256=sha(root/'prepare_complete.json'), auditor_sha256=sha(__file__), not_model_or_AP_evidence=True)
    out = root / 'independent_input_audit.json'
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(report, indent=2)); os.replace(tmp, out)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--wait-prepare', action='store_true')
    args = parser.parse_args()
    while args.wait_prepare and not (args.root/'prepare_complete.json').exists():
        assert not (args.root/'error.json').exists() and not (args.root/'stop.json').exists()
        state = read(args.root/'state.json')
        assert Path('/proc'/Path(str(state['owner_pid']))).exists(), 'preparation owner exited'
        time.sleep(10)
    audit(args.root)
