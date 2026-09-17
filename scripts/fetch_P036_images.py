"""Fetch images for one preregistered P036 stage without touching the other stage."""
import concurrent.futures
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path('/vepfs/vlms/1')
assert Path.cwd().resolve() == ROOT
assert len(sys.argv) == 2 and sys.argv[1] in {'development', 'confirmation'}
stage = sys.argv[1]
config_path = Path('configs/P036_risk_controlled_source_veto.json')
config = json.loads(config_path.read_text())
if stage == 'confirmation':
    result_path = Path('results/P036_development.json')
    assert result_path.exists(), 'development result is required before confirmation access'
    assert json.loads(result_path.read_text())['continuation_gate_evaluation']['all_pass']
records = config['stages'][stage]
signatures = records['image_signatures']
directory = ROOT / 'data/PinPoint/images'
directory.mkdir(parents=True, exist_ok=True)

def inspect(signature, data):
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        image_format = image.format
        image = image.convert('RGB')
        assert min(image.size) > 0
        pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return {
        'signature': signature,
        'path': str((directory / (signature + '.jpg')).relative_to(ROOT)),
        'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
        'pixel_sha256': pixel_hash,
        'size': list(image.size),
        'format': image_format,
    }

def fetch(signature):
    url = f'https://i.pinimg.com/736x/{signature[:2]}/{signature[2:4]}/{signature[4:6]}/{signature}.jpg'
    path = directory / (signature + '.jpg')
    if path.exists():
        try:
            return dict(inspect(signature, path.read_bytes()), url=url, cached=True)
        except Exception:
            pass
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(10, 30))
            response.raise_for_status()
            entry = inspect(signature, response.content)
            partial = path.with_suffix('.part')
            partial.write_bytes(response.content)
            partial.replace(path)
            return dict(entry, url=url, cached=False)
        except Exception as exception:
            error = str(exception)
            if attempt < 3:
                time.sleep(1 + attempt)
    return {'signature': signature, 'url': url, 'error': error}

entries = []
journal_path = Path(f'runs/P036_{stage}_image_fetch.jsonl')
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(fetch, signature): signature for signature in signatures}
    with journal_path.open('w') as journal:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            entries.append(result)
            journal.write(json.dumps(result) + '\n')
            journal.flush()
            if len(entries) % 100 == 0:
                print(len(entries), '/', len(signatures), 'failed', sum('error' in x for x in entries), flush=True)
entries.sort(key=lambda entry: entry['signature'])
failures = [entry for entry in entries if 'error' in entry]
report = {
    'complete': not failures,
    'stage': stage,
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'n_expected': len(signatures),
    'n_success': len(entries) - len(failures),
    'n_failed': len(failures),
    'proxy_environment_used': any(os.environ.get(name) for name in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']),
    'entries': entries,
}
manifest_path = Path(f'data/PinPoint/P036_{stage}_image_manifest.json')
manifest_path.write_text(json.dumps(report, indent=2) + '\n')
print('complete', report['complete'], 'success', report['n_success'], 'failed', report['n_failed'])
sys.exit(0 if report['complete'] else 1)
