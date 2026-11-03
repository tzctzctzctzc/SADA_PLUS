import numpy as np
import pandas as pd
from tools.audit_head_replacements import metrics


def test_full_ten_classes_AP_and_anyclass_background():
    gt=pd.DataFrame([('v',c*3.,c*3.+1.,c) for c in range(10)],columns=['video-id','t-start','t-end','label'])
    pred=gt.copy();pred['score']=.9
    extra=pd.DataFrame([('v',40.,41.,0,.4),('v',3.,4.,0,.4)],columns=pred.columns)
    result=metrics(gt,pd.concat([pred,extra],ignore_index=True))
    np.testing.assert_allclose(result['map_by_tiou'],np.ones(5))
    assert result['highscore_pure_task_background']==1
    assert result['highscore_nonBG']==11
