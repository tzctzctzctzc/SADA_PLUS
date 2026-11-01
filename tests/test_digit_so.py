"""固定协议和官方DiGIT数值接口；CPU只用作者reference算子，正式不替换CUDA算子。"""
import copy, io, json, os, runpy, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
# Import-only placeholders: no fake numerical output is returned.
try: import MultiScaleTemporalDeformableAttention
except ImportError: sys.modules['MultiScaleTemporalDeformableAttention']=types.ModuleType('MultiScaleTemporalDeformableAttention')
try: import nms_1d_cpu
except ImportError: sys.modules['nms_1d_cpu']=types.ModuleType('nms_1d_cpu')
from tools import digit_engine as e
from tools.digit_protocol import VENDOR,read,sha,terminal,decision
from models.digit.ops import temporal_deform_attn as op
from models.digit.model import PostProcess
torch.set_num_threads(2)

def cpu_reference(value,lens,starts,locations,weights,step):
    assert not value.is_cuda
    return op.deform_attn_core_pytorch(value,lens,locations,weights)

class ProtocolTests(unittest.TestCase):
    def test_upstream_blobs(self):
        for p,h in read(VENDOR/'upstream.json')['files'].items():self.assertEqual(sha(VENDOR/p),h,p)
    def test_first_stop(self):
        h=[[5,3.],[10,2.],[15,2.1],[20,2.2],[25,2.3]]
        self.assertEqual(terminal(h)['best_epoch'],10)
        with self.assertRaises(AssertionError):terminal(h+[[30,2.4]])
        with self.assertRaises(AssertionError):decision([[5,1.],[15,.9]])
    def test_units_short_boundary_and_unlabeled(self):
        seconds=torch.tensor([[0.,.04],[1.,2.7]])
        item=dict(video_id='v',feats=torch.zeros(4,32),fps=30.,duration=20.,feat_stride=16,feat_num_frames=32,
                  segments=(seconds*30-16)/16,labels=torch.tensor([2,7]))
        _,i=e.digit_item(item)
        torch.testing.assert_close(i['segments']+i['offset'],seconds)
        self.assertLess(float(i['segments'][0,0]),0.)
        cw=torch.stack([i['segments'].mean(-1),(i['segments'][:,1]-i['segments'][:,0]).log()],-1)
        out=dict(pred_logits=torch.tensor([[[0.,3.],[3.,0.]]]),pred_segments=cw[None],mask=torch.zeros(1,2,dtype=torch.bool))
        p=PostProcess()(out,torch.tensor([20.]),torch.tensor([17.]),torch.tensor([16]),i['offset'][None],duration_thresh=.05*16/30)[0]
        torch.testing.assert_close(p['segments'],seconds);self.assertEqual(len(p['scores']),2)
        _,clean=e.digit_item({k:v for k,v in item.items() if k not in ('segments','labels')})
        self.assertNotIn('segments',clean);self.assertNotIn('labels',clean)
    def test_dataset_empty_negative_retained(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);np.savez(d/'negative.npz',feats=np.zeros((32,4),np.float32))
            (d/'source.json').write_text(json.dumps(dict(database={'negative':dict(duration=18.,fps=30.,annotations=[])})))
            cfg=dict(dataset=dict(feat_folder=str(d),file_ext='.npz',downsample_rate=1,default_fps=30,feat_stride=16,num_frames=32,max_seq_len=2304,trunc_thresh=.3,crop_ratio=[.9,1.]))
            ds=e.DiGITDataset(cfg,d/'source.json',True)
            self.assertEqual(len(ds),1);self.assertEqual(ds[0][1]['segments'].shape,(0,2))
    def test_official_model_derivatives_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(op.MSDeformAttnFunction,'apply',side_effect=cpu_reference):
            d=Path(tmp)
            for i in range(4):np.savez(d/('v%d.npz'%i),feats=np.random.default_rng(i).normal(size=(64,16)).astype(np.float32))
            db={('v%d'%i):dict(duration=35.,fps=30.,annotations=[dict(segment=[3.,6.],label_id=i%2)]) for i in range(4)}
            db['v3']['annotations']=[]
            (d/'source.json').write_text(json.dumps(dict(database=db)))
            args={k:v for k,v in runpy.run_path(str(VENDOR/'config/digit/_base_.py')).items() if not k.startswith('_')}
            args.update({k:v for k,v in runpy.run_path(str(VENDOR/'config/digit/internvideo2/thumos14.py')).items() if not k.startswith('_')})
            # Small numerical fixture only; formal config keeps full 512/6/6 architecture.
            args.update(feature_dim=16,hidden_dim=64,enc_layers=2,dec_layers=2,dim_feedforward=128,
                        num_feature_levels=3,num_sampling_levels=3,max_queries=16,num_classes=10,lr_drop=50)
            cfg=dict(model=args,dataset=dict(feat_folder=str(d),file_ext='.npz',downsample_rate=1,default_fps=30,feat_stride=16,num_frames=32,max_seq_len=2304,trunc_thresh=.3,crop_ratio=[.9,1.]),
                loader=dict(num_workers=0),benchmark=dict(source_train=str(d/'source.json'),source_val=str(d/'source.json')))
            ctx=e.build(cfg,device='cpu',workers=0);batch=next(iter(ctx['train']))
            before={k:e.digest(v) for k,v in e.group_parameters(ctx['model']).items()}
            for i in range(3):loss,norms=e.step(ctx,batch,True)
            self.assertTrue(all(v>0 and np.isfinite(v) for v in norms.values()),norms)
            self.assertTrue(all(before[k]!=e.digest(v) for k,v in e.group_parameters(ctx['model']).items()))
            buffer=io.BytesIO();torch.save(e.capture(ctx,cfg,3,[],[]),buffer);buffer.seek(0);state=torch.load(buffer,weights_only=False)
            e.restore(ctx,copy.deepcopy(state));self.assertEqual(e.digest(e.capture(ctx,cfg,3,[],[])),e.digest(state))
            a,_=e.step(ctx,batch);da=e.digest(e.capture(ctx,cfg,4,[],[]))
            e.restore(ctx,copy.deepcopy(state));b,_=e.step(ctx,batch);db=e.digest(e.capture(ctx,cfg,4,[],[]))
            self.assertEqual(a,b);self.assertEqual(da,db)
            self.assertTrue(np.isfinite(e.validation(ctx)))
            x,info=e.device_batch(batch,'cpu');ctx['model'].eval()
            with torch.no_grad():
                a=ctx['model'](x,info);b=ctx['model'](x,[{k:v for k,v in row.items() if k not in ('segments','labels')} for row in info])
            self.assertEqual(e.digest(a),e.digest(b))
    def test_ema_first_update(self):
        m=torch.nn.Linear(2,1);ema=torch.optim.swa_utils.AveragedModel(m,multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.999))
        with torch.no_grad():m.weight.add_(2.)
        ema.update_parameters(m);self.assertEqual(e.digest(m.state_dict()),e.digest(ema.module.state_dict()))
    def test_shared_admission_no_lock(self):
        from tools import digit_gpu as gpu
        with tempfile.TemporaryDirectory() as d,patch.object(gpu,'snapshot',return_value=dict(index=0,uuid='unit',free_mib=50000)):
            admission=gpu.admit(0,d,12288,'unit')
            self.assertEqual(read(admission['lease'])['policy'],'shared_without_public_exclusive_lock')
            self.assertNotIn('lock',read(admission['lease']))
        with patch.object(gpu,'snapshot',return_value=dict(index=0,free_mib=1)):
            with self.assertRaises(BlockingIOError):gpu.admit(0,'unused',12288,'unit')
    def test_offline_evaluator_perfect_ap(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('digit_test_eval',CODE/'libs/evaluation/evaluator_anet.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gt.json';p.write_text(json.dumps(dict(database={'v':dict(subset='target',annotations=[dict(label_id=2,segment=[1,2])])})))
            evaluator=module.ANETdetection(str(p),['target'],num_workers=1,action_subsampling='10_majority_classes')
            pred={'video-id':np.array(['v']),'t-start':np.array([1.]),'t-end':np.array([2.]),'label':np.array([2]),'score':np.array([.9])}
            self.assertEqual(evaluator.evaluate(pred,verbose=False)[1],1.)
    def test_replay_difference_rejects_real_change(self):
        a=dict(weight=torch.tensor([1.,2.]),rng=torch.tensor([1,2],dtype=torch.uint8))
        b=copy.deepcopy(a);b['weight'][0]+=1e-7
        self.assertTrue(e.repeat_difference(a,b)['allclose'])
        b['weight'][0]+=.01;self.assertFalse(e.repeat_difference(a,b)['allclose'])
        b['rng'][0]=3
        with self.assertRaises(AssertionError):e.repeat_difference(a,b)

if __name__=='__main__':unittest.main(verbosity=2)
