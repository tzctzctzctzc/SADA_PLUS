"""固定首轮96的单变量对照：只去掉上下文，复用原区间内12帧。"""
import copy
from tools.baseline_protocol import read, save, sha
from tools.vlm_background_protocol import SPEC
from tools.vlm_background_short import prepare_short


def inside_row(row):
    result = copy.deepcopy(row)
    for key in ('indices','times','frame_records'):
        assert len(row[key]) == 16
        result[key] = row[key][2:14]
    # 原静态帧索引8在新12帧中对应6，像素/时标不更换。
    assert result['frame_records'][6] == row['frame_records'][8]
    return result


def prepare_inside(root):
    prepare_short(root)
    path = root/'inference_manifest.json'
    original = read(path)
    original_sha = sha(path)
    rows = [inside_row(row) for row in original['cases']]
    save(path,dict(spec=SPEC,cases=rows,GT_in_inference_input=False))
    comparison = root/'inside_only_comparison.json'
    save(comparison,dict(original_16frame_manifest_sha256=original_sha,
        original_96_root='/vepfs/vlms/vlm_background_short96_d200a08_1103',
        only_changed_input='remove original frames0,1,14,15; keep exact original2:14',
        static_original8_equals_new6=True,prompt_options_coefficients_unchanged=True,
        cases=[row['candidate_id'] for row in rows],selection_uses_GT=False))
    ready=read(root/'prepare_complete.json')
    ready['artifacts'][str(path)]=sha(path)
    ready['artifacts'][str(comparison)]=sha(comparison)
    save(root/'prepare_complete.json',ready)
