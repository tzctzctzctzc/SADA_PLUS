from copy import deepcopy
import pytest
from tools import run_probabilistic_source_six as driver


def reference(shift):
    return dict(seed=123,init_rand_seed=123,loader=dict(batch_size=2),opt=dict(learning_rate=.0001),
        model=dict(adv_method=['conditional_rev']),baseline=dict(shift=shift,regime='uda',
        code={},tmpdir='/old',epoch_budget=dict(iterations_per_epoch=172)))


@pytest.mark.parametrize('shift',driver.REMAINING)
def test_all_remaining_directions_keep_source_model_and_reference_budget(shift):
    old=reference(shift);cfg=driver.configuration(old,{'commit':'frozen'},'/tmp/task')
    driver.assert_config(cfg,old)
    assert cfg['opt']==old['opt'] and cfg['baseline']['epoch_budget']==old['baseline']['epoch_budget']
    assert cfg['baseline']['regime']=='so' and cfg['model']['adv_method']==['None']
    assert cfg['probabilistic_teacher']['prepared_cache'] is None
    assert cfg['probabilistic_teacher']['variant']=='source' and old==reference(shift)


def test_refuses_repeated_s1_target_learning_and_parameter_changes():
    for shift in ('S1','S7'):
        old=reference(shift)
        with pytest.raises(ValueError):driver.assert_config(driver.configuration(old,{},'/tmp/task'),old)
    old=reference('S2');cfg=driver.configuration(old,{},'/tmp/task')
    for section,key,value in (('opt','learning_rate',.002),('loader','batch_size',4),
                              ('probabilistic_teacher','variant','ema')):
        bad=deepcopy(cfg);bad[section][key]=value
        with pytest.raises(ValueError):driver.assert_config(bad,old)


def test_summary_requires_six_unique_source_results():
    rows=[dict(shift=s,variant='source',regime='so',map=.6) for s in driver.SHIFTS]
    assert driver.validate_results(rows)==pytest.approx(.6)
    with pytest.raises(ValueError):driver.validate_results(rows[:-1])
    rows[-1]=deepcopy(rows[0])
    with pytest.raises(ValueError):driver.validate_results(rows)
