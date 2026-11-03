from copy import deepcopy
import io
import torch
from tools.head_replacements import TaskInteraction,SPEC
from tools.run_head_replacements import TASKS,configuration


def test_zero_residual_preserves_original_values_including_padding():
    block=TaskInteraction(8,4)
    x=torch.randn(2,8,12)
    mask=torch.ones(2,1,12,dtype=torch.bool);mask[:,:,-3:]=False
    c,r=block([x],[mask])
    assert torch.equal(c[0],x) and torch.equal(r[0],x)


def test_both_tasks_and_cross_gates_learn_and_checkpoint_replays():
    torch.manual_seed(21)
    block=TaskInteraction(8,4);optimizer=torch.optim.AdamW(block.parameters(),lr=.01)
    x=torch.randn(2,8,10);mask=torch.ones(2,1,10,dtype=torch.bool)
    def step():
        optimizer.zero_grad(set_to_none=True)
        c,r=block([x],[mask]);loss=(c[0]-.1).square().mean()+2*(r[0]+.3).square().mean()
        loss.backward();optimizer.step()
        return loss.detach(),deepcopy(block.state_dict())
    step();step()
    assert block.cls_gate.weight.grad.norm()>0 and block.reg_gate.weight.grad.norm()>0
    assert block.cls_output.weight.grad.norm()>0 and block.reg_output.weight.grad.norm()>0
    buffer=io.BytesIO();torch.save(dict(model=block.state_dict(),opt=optimizer.state_dict()),buffer);buffer.seek(0)
    saved=torch.load(buffer,weights_only=False)
    first=step();block.load_state_dict(saved['model']);optimizer.load_state_dict(saved['opt']);second=step()
    assert torch.equal(first[0],second[0])
    assert all(torch.equal(v,second[1][k]) for k,v in first[1].items())


def test_invalid_padding_cannot_change_valid_interaction_features():
    block=TaskInteraction(8,4)
    with torch.no_grad():
        block.cls_output.weight.normal_();block.reg_output.weight.normal_()
    x=torch.randn(1,8,12);mask=torch.ones(1,1,12,dtype=torch.bool);mask[:,:,8:]=False
    a=block([x],[mask]);changed=x.clone();changed[:,:,8:]=999
    b=block([changed],[mask])
    assert torch.equal(a[0][0][:,:,:8],b[0][0][:,:,:8])
    assert torch.equal(a[1][0][:,:,:8],b[1][0][:,:,:8])


def test_all_six_both_heads_preserve_training_configuration():
    assert len(TASKS)==6 and len(set(TASKS))==6 and all(t.endswith('_trident') for t in TASKS)
    old=dict(model=dict(use_trident_head=False,adv_method=['conditional_rev']),
             baseline=dict(code={},tmpdir='old'),seed=123,loader=dict(batch_size=2))
    original=deepcopy(old)
    for variant in SPEC['variants']:
        cfg=configuration(old,variant,dict(commit='new'),'tmp')
        assert cfg['model']==old['model'] and cfg['seed']==123 and cfg['loader']==old['loader']
    assert old==original
