"""发布数据流与原Trainer验证早停；评测全部配置视频与全部对应GT。"""
from copy import deepcopy
import math

SCHEMA = 'sada-native-training-full-GT-evaluation-v3'
SPEC = dict(schema=SCHEMA, target_crop='release_GT_constrained_has_action_true',
    interval_sort='torch_sort_dim0_labels_follow_start_indices',
    train_generator='shared_CPU_default_generator_returned_by_fix_random_seed',
    validation_generator=None, persistent_workers=True,
    validation_every=2, early_stop_bad_validations=1, max_epochs=50,
    validation_order='target_train_then_source_val',test_every=2,
    debug_evaluation_order=['source_train','source_val','target_train','target_val'],
    evaluation='all_configured_prediction_videos_vs_all_corresponding_GT',
    validation_division='original_num_pos_no_clamp',
    target_GT_supervision=False, target_GT_sampling_and_native_statistics=True,
    initialization='fresh_seed123_no_trained_weights',
    resume='disabled_worker_process_RNG_not_reconstructible')


def manifests(database, cfg):
    source_roles=cfg['src_train_split']+cfg['src_val_split']
    source={k:v for k,v in database.items() if v['subset'].lower() in source_roles}
    target={k:v for k,v in database.items() if v['subset'].lower() in cfg['tgt_train_split']}
    validation={k for k,v in database.items() if v['subset'].lower() in cfg['tgt_val_split']}
    assert source and target and not set(source)&set(target) and not set(target)&validation
    assert any(v.get('annotations') for v in target.values())
    return {'database':source}, {'database':target}


def configuration(original, variant, identity, source, target, tmpdir, hashes):
    assert variant in ('sada','prototype')
    cfg=deepcopy(original)
    assert cfg['seed']==cfg['init_rand_seed']==123 and cfg['loader']['batch_size']==2
    assert cfg['model']['adv_method']==['conditional_rev']
    if variant=='sada':cfg.pop('prototype_capacity')
    cfg['dataset']['json_file']=str(source)
    cfg['baseline'].update(code=identity,tmpdir=str(tmpdir),schema=SCHEMA,
        target_manifest=str(target),target_gt_used=True,
        source_manifest_sha256=hashes['source'],target_manifest_sha256=hashes['target'],
        random_seeds=dict(model=123,source=123,target=123,validation=123,python_hash=123))
    cfg['baseline']['policy']=dict(max_epochs=50,valid_every=2,early_stop_bad_validations=1,
        selection='source_validation_cls_plus_reg',state='ema_state_dict',test_every=2)
    cfg['release_data_protocol']=deepcopy(SPEC)
    return cfg


def assert_pair(plain, prototype):
    value=deepcopy(prototype);branch=value.pop('prototype_capacity')
    assert value==plain and branch['variant']=='control'


def stream_pair(rows_a, rows_b):
    # 两组训练时长可不同；只比较共有的完整epoch，不补跑任何一组。
    common=min(len(rows_a),len(rows_b))
    return dict(common_epochs=common,passed=all(
        rows_a[i]['data_stream_sha256']==rows_b[i]['data_stream_sha256'] for i in range(common)))


def release_decision(history):
    assert history and [int(e) for e,_ in history]==list(range(2,int(history[-1][0])+1,2))
    best,best_epoch=10000.,None
    stopped=False
    for epoch,value in history:
        value=float(value)
        assert not stopped,'原版已经早停，不能继续训练'
        if value<best:best,best_epoch=float(value),int(epoch)
        else:stopped=True
    return dict(epoch=int(history[-1][0]),best_epoch=best_epoch,best_loss=best,
        reason='source_first_non_improvement' if stopped else 'max_epochs' if history[-1][0]==50 else 'running')
