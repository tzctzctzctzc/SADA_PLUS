from tools.cott_quality_diagnostics import counts,top_per_video,assess,DIAGNOSTICS

def p(v='a',score=.8,start=0):
    return dict(video_id=v,confidence=score,start=start,end=start+1,label='TAKE')

def test_same_budget_does_not_silently_pad_shorter_prediction_sets():
    rows=[p(),p(score=.7,start=2),p('b')]
    selected=top_per_video(rows,dict(a=1,b=4))
    assert len(selected)==2 and sum(x['video_id']=='a' for x in selected)==1

def test_score_curve_and_equal_budget_show_score_calibration_difference():
    gt=[p()]
    methods=dict(cott=[p(score=.9)],teacher=[p(score=.4)],direct=[p(score=.8,start=3)])
    out=assess(methods,gt,['a'])
    assert out['score_grid']['teacher']['0.5']['0.5']['tp']==0
    assert out['equal_count_to_cott']['teacher']['0.5']['tp']==1
    assert out['equal_count_to_cott']['cott']['0.5']['tp']==1
    assert out['equal_count_to_cott']['direct']['0.5']['fp']==1
    assert DIAGNOSTICS['no_best_grid_selection']
