import json
import pytest
from tools.cott_grouping_decoder import GroupingDecoder,to_intervals


class CharTokenizer:
    all_special_ids=[127]
    def __len__(self):return 128
    def encode(self,s,**kw):return list(map(ord,s))
    def decode(self,ids,skip_special_tokens=False,**kw):
        return ''.join(chr(i) for i in ids if not (skip_special_tokens and i==127))


def replay(raw,grid=[0.,.5,1.,1.5,2.,2.5]):
    d=GroupingDecoder(CharTokenizer(),0,grid,['TAKE','OPEN','OTHER'],127)
    for t in map(ord,raw):d.feed(t)
    assert d.done
    # finish also independently replays a trace not processed by a callback.
    fresh=GroupingDecoder(CharTokenizer(),0,grid,['TAKE','OPEN','OTHER'],127)
    return fresh.finish(list(map(ord,raw))+[127])


def test_zero_single_multiple_and_touching_groups_are_not_edited():
    assert replay('{"intervals":[]}')[0]=={'intervals':[]}
    raw='{"intervals":[{"start_pair":0,"end_pair":2,"label":"TAKE","description":"take plate"},{"start_pair":2,"end_pair":5,"label":"OPEN","description":"open door"}]}'
    rows,proof=replay(raw)
    assert rows['intervals'][0]['end']==rows['intervals'][1]['start']==1.
    assert rows['intervals'][1]['end']==2.5
    assert proof['manual_interval_edits']==proof['deleted_intervals']==0
    assert proof['generated_index_intervals']==json.loads(raw)


@pytest.mark.parametrize('a,b',[(1,4),(0,2),(3,3),(3,6)])
def test_overlap_empty_out_of_range_cannot_be_decoded(a,b):
    raw='{"intervals":[{"start_pair":0,"end_pair":3,"label":"TAKE","description":"take plate"},{"start_pair":'+str(a)+',"end_pair":'+str(b)+',"label":"OPEN","description":"open door"}]}'
    with pytest.raises(AssertionError):replay(raw)
    with pytest.raises(AssertionError):to_intervals(json.loads(raw),[0,.5,1,1.5,2,2.5],['TAKE','OPEN'])


def test_truncation_eos_in_string_invalid_label_and_empty_text_are_rejected():
    for raw in ['{"intervals":[','{"intervals":[{"start_pair":0,"end_pair":1,"label":"CUT","description":"cut"}]}',
        '{"intervals":[{"start_pair":0,"end_pair":1,"label":"TAKE","description":""}]}']:
        with pytest.raises(AssertionError):replay(raw)
    d=GroupingDecoder(CharTokenizer(),0,[0,1],['TAKE'],127)
    with pytest.raises(AssertionError):d.feed(127)


def test_real_pts_grid_not_uniform_time_and_no_float_rounding():
    grid=[0.,.5005,1.001,1.5015]
    obj={'intervals':[dict(start_pair=1,end_pair=3,label='TAKE',description='take plate')]}
    out=to_intervals(obj,grid,['TAKE'])
    assert out['intervals'][0]['start']==.5005 and out['intervals'][0]['end']==1.5015
