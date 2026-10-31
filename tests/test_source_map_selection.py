import math
import pytest
from tools.baseline_protocol import decision,termination,POLICY

def test_map_maximum_and_earliest_tie_select_checkpoint():
    result=termination([[5,.1],[10,.6],[15,.6],[20,.4],[25,.5]])
    assert result['best_epoch']==10 and result['best_map']==.6
    assert result['reason']=='source_patience_exhausted'
    assert POLICY['selection']=='source_validation_map_0.1_to_0.5'

def test_invalid_map_cannot_select_checkpoint():
    for score in (-.01,1.01,math.nan,math.inf):
        with pytest.raises(ValueError):decision([[5,score]])

def test_latest_improvement_has_priority_at_max_budget():
    history=[[epoch,epoch/100.] for epoch in range(5,51,5)]
    result=termination(history)
    assert result['best_epoch']==50 and result['reason']=='max_epochs'

def test_late_rebound_cannot_rescue_already_stopped_run():
    with pytest.raises(ValueError):termination([[5,.5],[10,.5],[15,.4],[20,.3],[25,.9]])
