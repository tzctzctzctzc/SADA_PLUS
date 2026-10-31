from copy import deepcopy
from pathlib import Path
from runpy import run_path
import torch
from tools import ttt_recovery as core
from tools.tta_core import unlabeled,predict_full,parameter_digest

helpers=run_path(str(Path(__file__).with_name('test_tta.py')))


def test_decoder_preparation_preserves_detector_and_normal_predictions(tmp_path):
    model=helpers['small_model'](tmp_path);named,head=core.initialize(model)
    reference=deepcopy(model)
    for p in reference.parameters():p.requires_grad_(False)
    video=unlabeled(helpers['video'](64));original=predict_full(model,video)
    detector=parameter_digest(dict(model.named_parameters()));decoder=parameter_digest(dict(head.named_parameters()))
    optimizer=core.preparation_optimizer(model,head)
    loss,_=core.recovery_loss(model,reference,head,video,123)
    optimizer.zero_grad();loss.backward();optimizer.step()
    assert parameter_digest(dict(model.named_parameters()))==detector
    assert parameter_digest(dict(head.named_parameters()))!=decoder
    assert all(p.grad is None for p in model.parameters())
    after=predict_full(model,video)
    for k in ('segments','labels','scores'):assert torch.equal(original[k],after[k])
    initial={n:p.detach().clone() for n,p in named.items()}
    _,proof=core.adapt(model,reference,head,initial,video,1e-4,1)
    assert proof['changed'] and proof['trace'][0]['gradient_norm']>0


def test_protocol_uses_source_recovery_for_decoder_choice_and_reuses_baseline():
    from tools.baseline_protocol import CODE
    text=(CODE/'tools/run_ttt_recovery.py').read_text(encoding='utf-8')
    assert "chosen=min(history,key=lambda x:x['source_recovery'])" in text
    assert "target_jobs=6,reused_original_prototype=6" in text
    assert core.SPEC['preparation_variants']==['reconstruction']
