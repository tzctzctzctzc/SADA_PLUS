"""补齐S2所需P30_08:3467；沿用已核验的官方RGB时间映射，生成独立恢复根。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import av
import cv2
import numpy as np

from tools.prepare_rq3_candidates import exact_rgb_frames
from tools.rq2_dense_protocol import sha256_file


def md5(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/vepfs/vlms/data/EPIC_KITCHENS'))
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    old = args.data_root / 'rq2_frame_recovery_qwen_0924'
    video = args.data_root / 'videos/P30_08.MP4'
    archive_path = args.data_root / 'rq2_frame_recovery/source/P30_08.tar'
    old_sha = '5a8041d6355e1e5b9d9f1e268f903c58dd604efa906d780f69268d8996483b90'
    if sha256_file(old / 'manifest.json') != old_sha:
        raise ValueError('original recovery manifest changed')
    if md5(video) != 'bd32d6e5c5815beb42310563b0ce750e' or md5(archive_path) != '7dea2ad436564ea7713c86b5fe5a62e5':
        raise ValueError('official MP4/RGB archive checksum differs')
    manifest = json.loads((old / 'manifest.json').read_text())
    if 'P30_08:3467' in manifest['frames'] or args.output_root.exists():
        raise ValueError('recovery must use a fresh destination and a genuinely missing index')
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        timestamps = sorted(p.pts for p in container.demux(stream) if p.pts is not None)
        if len(timestamps) != stream.frames or len(set(timestamps)) != len(timestamps):
            raise ValueError('MP4 sample index is ambiguous')
    # 官方RGB为60Hz、1-based；使用实际PTS，不用文件平均fps或相邻帧替代。
    tar_index = lambda index: round(timestamps[index] * time_base * 60) + 1
    controls = {}
    for index in (3456, 3497):
        controls.update(exact_rgb_frames(video, {index}))
    validation = []
    with tarfile.open(archive_path, 'r:') as archive:
        def rgb_frame(index):
            name = './frame_%010d.jpg' % index
            raw = archive.extractfile(name).read()
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('official RGB frame cannot be decoded: ' + name)
            return name, raw, image

        for index, frame in sorted(controls.items()):
            expected = tar_index(index)
            reference = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
            errors = {}
            for candidate in range(expected - 2, expected + 3):
                image = rgb_frame(candidate)[2][..., ::-1]
                small = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
                errors[candidate] = float(np.abs(reference - small).mean())
            if min(errors, key=errors.get) != expected:
                raise ValueError('RGB/MP4 temporal mapping failed at frame %d' % index)
            validation.append(dict(mp4_frame_index=index, predicted_tar_index=expected,
                                   best_tar_index=min(errors, key=errors.get), errors=errors))
        member, raw, recovered = rgb_frame(tar_index(3467))
    # 原131帧逐项验真复制，旧根保持可重现。
    args.output_root.mkdir(parents=True, exist_ok=False)
    for entry in manifest['frames'].values():
        source = old / entry['path']
        frame = np.load(source, allow_pickle=False)
        if frame.dtype != np.uint8 or hashlib.sha256(frame.tobytes()).hexdigest() != entry['frame_sha256']:
            raise ValueError('existing recovery pixel hash differs')
        destination = args.output_root / entry['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    relative = Path('P30_08/3467.npy')
    np.save(args.output_root / relative, recovered, allow_pickle=False)
    manifest['frames']['P30_08:3467'] = dict(
        path=relative.as_posix(), frame_index=3467, color_order='bgr', dtype='uint8',
        shape=list(recovered.shape), frame_sha256=hashlib.sha256(recovered.tobytes()).hexdigest(),
        source_kind='official_rgb_frame_tar_time_aligned_to_mp4_pts',
        video_md5=md5(video), rgb_tar_md5='7dea2ad436564ea7713c86b5fe5a62e5',
        rgb_tar_member=member, rgb_tar_frame_index=tar_index(3467),
        rgb_member_sha256=hashlib.sha256(raw).hexdigest(), mapping_validation=validation,
        mp4_frame_time_sec=float(timestamps[3467] * time_base))
    manifest.update(parent_manifest_sha256=old_sha, recovery_reason='RQ3 S2 source proposal exact frame 3467')
    output = args.output_root / 'manifest.json'
    output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(manifest=str(output), sha256=sha256_file(output), count=len(manifest['frames']),
                          recovered_member=member, validation=validation)), flush=True)


if __name__ == '__main__':
    main()
