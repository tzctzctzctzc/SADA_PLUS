import json
import pytest
from tools.run_rq4_entry import supervision, acceptance_ready


def rows(path, accepted=0, loss=0, gradient=0):
    row = dict(epoch=1)
    for key in ('queried','matched','routable','accepted','crop_rejected','anchor_rejected',
                'gate_rejected','loss','raw_loss','gradient_norm'):
        row['t_rq4_'+key] = 0
    row.update(t_rq4_queried=2, t_rq4_matched=1, t_rq4_routable=1,
               t_rq4_accepted=accepted,t_rq4_loss=loss,t_rq4_gradient_norm=gradient)
    for key in ('cls','reg','downstream'): row['t_target_gt_'+key+'_loss'] = 0
    path.write_text(json.dumps(row))
    return row


def test_zero_coverage_is_valid_and_retained(tmp_path):
    path = tmp_path/'epochs.jsonl'; rows(path)
    assert supervision(path,1)['intervened'] is False


def test_valid_intervention(tmp_path):
    path = tmp_path/'epochs.jsonl'; rows(path,1,.1,.3)
    assert supervision(path,1)['effective_gradient'] is True


@pytest.mark.parametrize('key,value', [('t_target_gt_cls_loss',.1),
    ('t_rq4_loss',float('nan')), ('t_rq4_accepted',3), ('epoch',2)])
def test_reject_invalid_history(tmp_path,key,value):
    path = tmp_path/'epochs.jsonl'; row = rows(path)
    row[key] = value; path.write_text(json.dumps(row))
    with pytest.raises(ValueError): supervision(path,1)


def test_reject_detached_loss(tmp_path):
    path = tmp_path/'epochs.jsonl'; rows(path,1,.1,0)
    with pytest.raises(ValueError,match='no gradient'): supervision(path,1)


def test_reject_unbound_acceptance(tmp_path):
    path = tmp_path/'acceptance.json'
    path.write_text(json.dumps(dict(status='passed',code_commit='wrong')))
    with pytest.raises(ValueError,match='code/status'): acceptance_ready(path,'timelens2_8b','correct')


def test_reject_missing_strategy_acceptance(tmp_path):
    path = tmp_path/'acceptance.json'
    path.write_text(json.dumps(dict(status='passed',code_commit='correct',reports=[])))
    with pytest.raises(ValueError,match='both strategies'): acceptance_ready(path,'timelens2_8b','correct')


def test_finite_matrix_no_duplicates():
    # Linux-only orchestration module uses fcntl for a cross-process lock.
    pytest.importorskip('fcntl')
    from tools.run_rq4_matrix import matrix_entries
    entries=matrix_entries()
    assert len(entries)==len(set(entries))==24
    assert {e[2] for e in entries}=={'S1','S2','S3','S4','S5','S6'}


def test_smoke_prioritizes_its_code_before_trainer_import():
    import inspect
    from tools.smoke_rq4_training import main
    code=inspect.getsource(main)
    assert code.index('sys.path.insert') < code.index('import libs.trainer')
