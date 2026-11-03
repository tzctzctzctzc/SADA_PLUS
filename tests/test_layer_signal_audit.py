import numpy as np
from tools.audit_layer_signals import conditional
from tools.layer_signal_math import conditional_auc


def test_independent_conditional_AUC_matches_pair_enumeration():
    g=np.random.default_rng(123);n=70;bg=g.random(n)>.5;v=np.array(['v'+str(i%7) for i in range(n)]);classes=np.arange(n)%2;s=g.random(n);length=g.random(n);value=g.random(n)
    edges={str(c):dict(score=[.25,.5,.75],length=[.5]) for c in (0,1)}
    a=conditional(bg,value,classes,s,length,v,edges);b=conditional_auc(bg,value,classes,s,length,v,edges)
    for k,x in a.items():assert x==b[k]
