import json
import pytest
from tools.rq3_temporal_teachers import TEACHERS, check_loading
from tools.rq3_extra_teachers import frozen_protocol
from tools.rq3_protocol import read_frozen_protocol, GENERATIVE_MODELS
from tools.run_rq3_matrix import load_entry_plan
from tools.run_rq3_trusted_queue import TEACHERS as QUEUE_TEACHERS


def test_exact_new_36(tmp_path):
    entries = [dict(teacher=t,strategy=s,shift=f'S{i}') for t in TEACHERS
               for s in ('hard','verify','prototype') for i in range(1,7)]
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(schema='rq3-expanded-fresh-matrix-v1', seed=6, fresh=True, entries=entries)))
    assert len(load_entry_plan(plan)) == 36
    for t in TEACHERS:
        assert t in GENERATIVE_MODELS and t in QUEUE_TEACHERS
        anchor = tmp_path/(t+'.json'); anchor.write_text(json.dumps(frozen_protocol(t)))
        assert read_frozen_protocol(anchor,t) == frozen_protocol(t)


@pytest.mark.parametrize('key',['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
def test_no_silent_model_substitution(key):
    with pytest.raises(ValueError): check_loading({key:['codec.weight']})
    check_loading({key:[]})
