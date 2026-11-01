from copy import deepcopy
import torch
from tools.temporal_energy_transfer import source_record_to_device
from tools.temporal_interval_energy import TemporalIntervalEnergy, score_intervals, energy_loss


def record():
    return dict(features=[torch.arange(512*40,dtype=torch.float32).reshape(1,512,40)/10000],
                masks=[torch.ones(1,1,40,dtype=torch.bool)],
                segments=torch.tensor([[10.,20.]]),gt_labels=torch.tensor([0]),
                anchors=torch.tensor([[9.,20.],[10.,21.]]),
                truths=torch.tensor([[10.,20.],[10.,20.]]),
                classes=torch.tensor([0,0]),instances=torch.tensor([0,0]),
                levels=torch.tensor([0,0]),length=(0.,40.),video_id='fixture')


def test_transfer_preserves_values_metadata_and_rng():
    before=record();rng=torch.get_rng_state().clone();after=source_record_to_device(before,'cpu')
    assert torch.equal(rng,torch.get_rng_state())
    assert after['length']==before['length'] and after['video_id']==before['video_id']
    for key in ('features','masks'):
        assert all(torch.equal(a,b) for a,b in zip(before[key],after[key]))
    for key in ('segments','gt_labels','anchors','truths','classes','instances','levels'):
        assert torch.equal(before[key],after[key])
    assert after is not before


def test_same_batch_order_has_exact_loss_gradient_and_adam_updates():
    torch.manual_seed(123);old=TemporalIntervalEnergy();new=deepcopy(old)
    optimizers=[torch.optim.AdamW(h.parameters(),lr=.0003) for h in (old,new)]
    cpu=record();cached=source_record_to_device(cpu,'cpu')
    for ids in (torch.tensor([1,0]),torch.tensor([0,1])):
        losses=[];gradients=[]
        for head,opt,data in zip((old,new),optimizers,(source_record_to_device(cpu,'cpu'),cached)):
            a=data['anchors'][ids];boxes=torch.stack((a,data['truths'][ids]),1)
            score,accept,valid=score_intervals(head,data['features'],data['masks'],a,boxes,data['levels'][ids],data['classes'][ids],[1])
            q=torch.tensor([[.8,1.],[.8,1.]])
            loss=energy_loss(score,accept,q,valid,torch.ones(2,dtype=torch.bool)).mean()
            opt.zero_grad();loss.backward();losses.append(loss.detach())
            gradients.append([p.grad.clone() for p in head.parameters()]);opt.step()
        assert torch.equal(*losses)
        assert all(torch.equal(a,b) for a,b in zip(*gradients))
        assert all(torch.equal(a,b) for a,b in zip(old.parameters(),new.parameters()))
