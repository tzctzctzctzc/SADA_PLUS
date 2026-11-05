"""源验证视频内划分的诊断校准；目标GT只用于离线评估，不用于拟合。"""
import hashlib
import numpy as np


def auc(label,score):
    label=np.asarray(label,dtype=bool);score=np.asarray(score,float)
    if not label.any() or label.all():return None
    order=np.argsort(score,kind='stable');value=score[order]
    starts=np.r_[0,np.flatnonzero(value[1:]!=value[:-1])+1];ends=np.r_[starts[1:],len(value)]
    ranks=np.empty(len(value));ranks[order]=np.repeat((starts+ends-1)/2+1,ends-starts)
    positives=label.sum();return float((ranks[label].sum()-positives*(positives+1)/2)/(positives*(len(label)-positives)))


def sigmoid(x):return np.exp(-np.logaddexp(0.,-x))


def metrics(label,probability):
    y=np.asarray(label,float);p=np.asarray(probability,float)
    if not len(y):return dict(count=0)
    bins=[];ece=0.
    for i in range(10):
        use=(p>=i/10)&(p<(i+1)/10 if i<9 else p<=1.)
        if use.any():
            confidence=float(p[use].mean());accuracy=float(y[use].mean());ece+=use.mean()*abs(confidence-accuracy)
            bins.append(dict(lo=i/10,count=int(use.sum()),predicted_quality=confidence,observed_quality=accuracy))
    return dict(count=len(y),positive_rate=float(y.mean()),auroc=auc(y,p),brier=float(np.mean((p-y)**2)),
        log_loss=float(np.mean(-y*np.log(np.clip(p,1e-12,1))-(1-y)*np.log(np.clip(1-p,1e-12,1)))),ece=float(ece),bins=bins)


def features(row):
    left=np.asarray(row['left'],float);right=np.asarray(row['right'],float)
    bins=left.shape[1];dist_l=np.arange(bins-1,-1,-1);dist_r=np.arange(bins)
    means_l=left@dist_l;means_r=right@dist_r
    var_l=np.maximum(left@(dist_l**2)-means_l**2,0);var_r=np.maximum(right@(dist_r**2)-means_r**2,0)
    stride=np.asarray(row['stride'],float);centre=np.asarray(row['centre'],float)
    duration=np.maximum(np.diff(np.asarray(row['segments']),axis=1).ravel(),stride)
    entropy_l=-(left*np.log(np.clip(left,1e-30,1))).sum(1)/np.log(bins)
    entropy_r=-(right*np.log(np.clip(right,1e-30,1))).sum(1)/np.log(bins)
    edge_l=(left*((centre[:,None]-stride[:,None]*dist_l)<0)).sum(1)
    edge_r=(right*((centre[:,None]+stride[:,None]*dist_r)>row['video_length'])).sum(1)
    std_l=np.sqrt(var_l)*stride/duration;std_r=np.sqrt(var_r)*stride/duration
    score=np.clip(np.asarray(row['scores'],float),1e-6,1-1e-6)
    nuisance=np.column_stack((np.log(score/(1-score)),np.log(duration),np.log(stride),np.eye(10)[np.asarray(row['labels'])]))
    uncertainty=np.column_stack((entropy_l,entropy_r,std_l,std_r,left.max(1),right.max(1),edge_l,edge_r))
    return nuisance,uncertainty,dict(entropy=(entropy_l+entropy_r)/2,relative_std=np.sqrt(std_l**2+std_r**2),left_std=std_l,right_std=std_r)


def match_quality(segments,labels,truth_segments,truth_labels):
    n=len(segments);best=np.zeros(n);error=np.full((n,2),np.nan);matched=np.zeros(n,dtype=bool)
    for category in range(10):
        ids=np.flatnonzero(labels==category);gt=truth_segments[truth_labels==category]
        if not len(gt):continue
        for start in range(0,len(ids),1024):
            pick=ids[start:start+1024];pred=segments[pick]
            intersection=np.maximum(np.minimum(pred[:,None,1],gt[None,:,1])-np.maximum(pred[:,None,0],gt[None,:,0]),0)
            union=np.diff(pred,axis=1)+np.diff(gt,axis=1).ravel()[None,:]-intersection
            overlap=intersection/np.maximum(union,1e-12);chosen=overlap.argmax(1);best[pick]=overlap[np.arange(len(pick)),chosen]
            matched[pick]=True;error[pick]=np.abs(pred-gt[chosen])/np.maximum(np.diff(gt[chosen],axis=1),1e-12)
    return best,error,matched


def source_partition(videos):
    # 划分固定且仅依赖源视频ID；目标视频/标签均不参与选择。
    ordered=sorted(set(videos),key=lambda v:hashlib.sha256(('uncertainty1105:'+v).encode()).hexdigest())
    assert len(ordered)>=4
    return set(ordered[::2]),set(ordered[1::2])


def fit_logistic(x,y,video):
    x=np.asarray(x,float);y=np.asarray(y,float);video=np.asarray(video)
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-4);z=np.column_stack((np.ones(len(x)),(x-mean)/scale))
    _,inverse,count=np.unique(video,return_inverse=True,return_counts=True)
    weight=1/count[inverse];weight/=weight.sum();beta=np.zeros(z.shape[1]);ridge=np.eye(z.shape[1])*.001;ridge[0,0]=0
    def objective(b):return float(weight@(np.logaddexp(0,z@b)-y*(z@b))+.5*b@ridge@b)
    for _ in range(30):
        p=sigmoid(z@beta);gradient=z.T@(weight*(p-y))+ridge@beta
        hessian=z.T@((weight*p*(1-p))[:,None]*z)+ridge+np.eye(z.shape[1])*1e-9
        step=np.linalg.solve(hessian,gradient);rate=1.;old=objective(beta)
        while rate>1e-6 and objective(beta-rate*step)>old:rate*=.5
        beta-=rate*step
        if np.max(np.abs(rate*step))<1e-6:break
    return dict(mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),fit_scope='source validation fit videos only; diagnostic, not deployed')


def predict(model,x):
    z=np.column_stack((np.ones(len(x)),(x-np.asarray(model['mean']))/np.asarray(model['scale'])))
    return sigmoid(z@np.asarray(model['beta']))


def rank_correlation(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    def rank(v):
        v=np.asarray(v);u,inv,c=np.unique(v,return_inverse=True,return_counts=True)
        starts=np.r_[0,np.cumsum(c)[:-1]];return (starts+(c-1)/2)[inv]
    return float(np.corrcoef(rank(x),rank(y))[0,1])
