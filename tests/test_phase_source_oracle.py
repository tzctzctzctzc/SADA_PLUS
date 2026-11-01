import numpy as np
from tools.phase_source_oracle import oracle_predictions

def test_class_and_endpoint_oracle_remain_distinct_and_keep_scores_and_duplicates():
 p={'video-id':np.array(['a','a','a']), 't-start':np.array([1.,1.,20.]),'t-end':np.array([3.,3.,22.]),'label':np.array([0,1,0]),'score':np.array([.8,.7,.6])}
 gt={'a':[{'segment':[0.,4.],'label_id':0}]}
 cls,n=oracle_predictions(p,gt,'class');assert n==2 and cls['label'].tolist()==[0,0,0]
 assert np.array_equal(cls['t-start'],p['t-start'])
 start,n=oracle_predictions(p,gt,'start');assert n==1 and start['t-start'].tolist()==[0.,1.,20.]
 assert np.array_equal(start['t-end'],p['t-end'])
 both,n=oracle_predictions(p,gt,'both');assert n==1 and both['t-start'][0]==0 and both['t-end'][0]==4
 for out in [cls,start,both]:assert np.array_equal(out['score'],p['score']) and len(out['score'])==3
 assert p['t-start'].tolist()==[1.,1.,20.]
