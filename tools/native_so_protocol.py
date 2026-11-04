"""新协议SO：源数据拟合，固定UDA更新预算；目标数据只无梯度评测。"""
from copy import deepcopy

from tools.release_data_protocol import SPEC

SO_SPEC=deepcopy(SPEC)
SO_SPEC.update(target_crop='not_used_in_source_only_fit',
    target_GT_sampling_and_native_statistics=False,
    target_statistics='offline_validation_only_no_gradient')


def source_only_configuration(native,identity):
    cfg=deepcopy(native)
    assert cfg['baseline']['regime']=='uda'
    assert cfg['model']['adv_method']==['conditional_rev']
    assert 'prototype_capacity' not in cfg
    cfg['baseline'].update(regime='so',code=identity,target_gt_used=False)
    cfg['release_data_protocol']=deepcopy(SO_SPEC)
    cfg['source_only_control']=dict(
        fit='source_train_only_in_both_model_input_slots',
        second_slot='unlabeled_source_train_preserves_original_trainer_interface',
        runtime_adv_method=['None'],
        budget='same_iterations_per_epoch_and_schedule_as_paired_native_UDA',
        target_use='native_no_gradient_validation_and_debug_AP_only',
        selection='source_val_only',
        source_random_stream='not_claimed_identical_to_UDA_different_second_loader',
        interpretation='complete_adaptation_pipeline_vs_source_only_not_isolated_alignment_switch')
    return cfg
