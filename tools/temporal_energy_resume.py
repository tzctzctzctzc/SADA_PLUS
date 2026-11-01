"""授权暂停后的源阶段接续，以及真实GPU上的搬运等价性验收。"""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import time
import torch
from tools.baseline_protocol import read, save, sha
from tools.temporal_energy_transfer import source_record_to_device, restore_source_checkpoint


def prepare_resume(root, old, identity, specification):
    receipt_path = Path(read(old/'monitor/authorized_pause_latest.json')['receipt'])
    receipt = read(receipt_path)
    assert receipt['all_owned_processes_exited']
    for process in [receipt['parent'], *receipt['workers'].values()]:
        stat = Path('/proc')/str(process['pid'])/'stat'
        assert not stat.exists() or stat.read_text().split(') ', 1)[1].split()[0] == 'Z'
    assert read(old/'plan.json')['specification'] == specification
    save(root/'plan.json', dict(schema=specification['schema'], specification=specification,
         code=identity, resume_from=str(old), pause_receipt=str(receipt_path),
         pause_receipt_sha256=sha(receipt_path), extra_training_budget=True,
         recovery='same source epoch history/optimizer/RNG; uncommitted partial epoch replay'))
    for name in ['inputs.json', 'acceptance.json']:
        save(root/('predecessor_'+name), read(old/name))
    acceptance = read(old/'acceptance.json')
    save(root/'acceptance.json', dict(code=identity, specification=specification,
         predecessor_acceptance=str(old/'acceptance.json'),
         predecessor_acceptance_sha256=sha(old/'acceptance.json'),
         minimum_free_mib=acceptance['minimum_free_mib'],
         new_worker_gpu_equivalence_required=True))
    for shift in ['S1','S2','S3','S4','S5','S6']:
        entry=root/'formal'/shift;entry.mkdir(parents=True)
        previous=old/'formal'/shift
        for name in ['config.json','inputs.json','source_train_gt.json','source_val_gt.json',
                     'source_val_unlabeled.json','target_val_unlabeled.json','target_train_unlabeled.json']:
            shutil.copyfile(previous/name,entry/name)
        for name in ['source_val_unlabeled.json','target_val_unlabeled.json','target_train_unlabeled.json']:
            assert all('annotations' not in v for v in read(entry/name)['database'].values())
        if shift not in receipt['saved']:continue
        snapshot=Path(receipt['snapshot_root'])/shift
        assert not (previous/'joint').exists()
        for name,digest in receipt['saved'][shift]['artifacts'].items():
            assert sha(snapshot/name)==digest
            shutil.copyfile(snapshot/name,entry/name)
        for role in ['source_train','source_val']:
            shutil.copyfile(previous/(role+'_cache_index.json'),entry/(role+'_cache_index.json'))
        save(entry/'resume.json', dict(source_checkpoint=str(snapshot/'latest_energy.pt'),
             source_checkpoint_sha256=sha(snapshot/'latest_energy.pt'),
             source_best_checkpoint=str(snapshot/'best_energy.pt'),
             source_best_checkpoint_sha256=sha(snapshot/'best_energy.pt'),
             prefix_epoch=receipt['saved'][shift]['epoch'],
             prefix_history_sha256=sha(snapshot/'energy_history.json'),
             old_acceptance=str(snapshot/'acceptance.json'),
             old_acceptance_sha256=sha(snapshot/'acceptance.json'),
             cache_indexes={role:sha(entry/(role+'_cache_index.json')) for role in ['source_train','source_val']}))
    return acceptance['minimum_free_mib']


def gpu_transfer_gate(head, optimizer, generator, saved, data, parts, specification, digest):
    """同一真实视频、相同初态/两次四步更新；不计入正式训练，之后还原全部状态。"""
    ids=torch.randperm(len(data['anchors']),generator=torch.Generator().manual_seed(303))[:16]
    batches=list(ids.split(specification['source_group_batch']))
    results=[]
    for cached in [False,True]:
        restore_source_checkpoint(head,optimizer,generator,saved,specification)
        head.train();torch.cuda.synchronize();start=time.perf_counter()
        resident=source_record_to_device(data,'cuda:0') if cached else None
        losses=[]
        for batch in batches:
            record=resident if cached else source_record_to_device(data,'cuda:0')
            loss=parts(record,batch)[0].mean();optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            optimizer.step();losses.append(float(loss))
        torch.cuda.synchronize()
        results.append(dict(seconds=time.perf_counter()-start,losses=losses,
            state_sha256=digest(dict(model=head.state_dict(),optimizer=optimizer.state_dict(),
                generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0)))))
        del resident,record
    assert results[0]['losses']==results[1]['losses']
    assert results[0]['state_sha256']==results[1]['state_sha256']
    restore_source_checkpoint(head,optimizer,generator,saved,specification)
    return dict(status='passed',steps=len(batches),actual_gpu_updates=True,
        exact_loss_optimizer_parameter_and_rng=True,formal_state_restored=True,
        baseline=results[0],cached=results[1],
        microbenchmark_speedup=results[0]['seconds']/results[1]['seconds'],
        timing_scope='one real video and four steps; not whole-epoch speedup')
