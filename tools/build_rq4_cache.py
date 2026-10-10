"""仅CPU：将两教师原生response转换为可训练边界缓存，不启动教师或学生。"""
import argparse
import json
from pathlib import Path
from tools.rq4_protocol import build_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='JSON: candidates, responses, teacher, shift, identity, min_iou, ambiguity_margin')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    result = build_cache(**payload)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(records=len(result['records']), payload_sha256=result['payload_sha256'])))


if __name__ == '__main__':
    main()
