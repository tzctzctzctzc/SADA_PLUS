import sys
import types
from PIL import Image
from tools.run_qwen3_vl_classification import mixed_frame_grid


def test_uniform_input_is_unchanged(tmp_path):
    p=tmp_path/'same.png'
    Image.new('RGB',(456,256)).save(p)
    assert mixed_frame_grid([p]*8,224**2,448**2)=={}


def test_mixed_input_uses_smallest_frame_and_keeps_files(tmp_path,monkeypatch):
    paths=[]
    for i,size in enumerate([(456,256),(1920,1080)]):
        p=tmp_path/(str(i)+'.png');Image.new('RGB',size).save(p);paths.append(p)
    before=[p.read_bytes() for p in paths]
    def resize(h,w,**kw):
        assert (h,w)==(256,456)
        assert kw==dict(factor=32,min_pixels=224**2,max_pixels=448**2)
        return 256,448
    monkeypatch.setitem(sys.modules,'qwen_vl_utils.vision_process',types.SimpleNamespace(smart_resize=resize))
    assert mixed_frame_grid(paths,224**2,448**2)==dict(resized_height=256,resized_width=448)
    assert before==[p.read_bytes() for p in paths]
