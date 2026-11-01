"""DyFADet EMA自训练与严格匹配源优化步数的继续训练对照。"""
from pathlib import Path
from copy import deepcopy
import math,subprocess,os
from tools.dyfadet_protocol import read,save,sha,CODE,VENDOR

REFERENCE=Path('/vepfs/vlms/dyfadet_so6_694c64a_1101')
VARIANT=os.environ.get('DYFA_TEXT_VARIANT','tifad')
assert VARIANT in ('tifad','pda')
SCHEMA='dyfadet-'+VARIANT+'-decoupled-uda-v1'
SPEC=dict(seed=123,source_batch=2,target_batch=2,max_epochs=30,validation_every=5,patience=15,
    initialization='per_shift_audited_SO_best_EMA',epoch0_candidate=True,ema_decay=.999,
    learning_rate=1e-5,eta_min=1e-6,warmup_epochs=0,weight_decay=.05,
    target_threshold=.5,classification_weight='minimum_teacher_confidence_of_tied_shortest_instances',
    regression_temporal_scales=[.9,1.1],regression_min_iou=.7,regression_max_endpoint_fraction=.15,
    regression_matching='same_class_mutual_best_IoU_in_both_views',
    regression_targets='original_official_point_instance_assignment_mask_unstable_and_tied_instances_no_reassignment',
    temporal_mapping='align_corners_false_pixel_centers_inverse_actual_length_ratio',
    validation_checkpoint_every=5,regression_stability_is_accuracy=False,
    target_regression_weight=.5,target_ramp_epochs=5,channel_dropout=.1,
    target_supervision='original_point_assignment_positive_locations_only_other_locations_ignored',
    target_normalization='classification_unweighted_positive_count_regression_separate_stable_positive_count',source_loss='unchanged_official_DyFADet',
    target_crop='uniform_without_GT_shared_teacher_student_local_grid',
    augmentation='channel_dropout_only_preserves_time_coordinates',
    source_control='same_SO_initialization_and_actual_UDA_source_update_count',
    selection='source_equal_video_cls_plus_reg_best_EMA',
    raw_features='unchanged_SlowFast',precision='float32',vlm=False,adversarial=False,
    randomness='independent_source_target_model_streams_and_loader_generators',
    target_GT=False,scheduling='shared_without_public_exclusive_lock')

SPEC.update(text_variant=VARIANT,vlm=True,text_encoder='frozen_OpenAI_CLIP_B16',
    text_source='fixed_class_names_and_assistant_authored_start_middle_end_global_descriptions',
    target_video_VLM_queries=False,text_phases=4,feature_dimension=512,feature_levels=6,
    semantic_initialization='full_strength_Xavier_TiCA_and_standard_PDA_fusion_no_zero_gate',semantic_learning_rate=1e-4,
    source_loss=('official_DyFADet_plus_hard_and_Gaussian_soft_foreground' if VARIANT=='tifad' else 'official_DyFADet_plus_hard_foreground'),
    architecture=('six_level_bidirectional_text_visual_attention_with_salient_Gaussian_support' if VARIANT=='tifad' else
                  'four_phase_temporal_importance_filter_cross_attention_phase_token_MHSA_adaptive_fusion_classification_and_three_layer_regression_MLP'),
    adaptation_to_closed_set='retain_pretrained_DyFA_pyramid_and_closed_set_heads_full_text_feature_fusion_plus_class_similarity_foreground_probability_product_at_inference',
    foreground=('hard_focal_and_sigma1_Gaussian_soft_focal' if VARIANT=='tifad' else 'hard_focal'),
    PDA_descriptions_are_original_paper_GPT4o=False,PDA_stage_order=['start','middle','end','global'],
    TiFAD_salience='author_THUMOS_top1percent_threshold100_scaled_by_valid_length1152_sigmoid_grid_sigma_sum_max_normalization',
    pseudo_foreground_fusion=False,final_foreground_fusion=('sqrt_sigmoid_classification_times_soft_then_times_hard' if VARIANT=='tifad' else 'sigmoid_classification_times_hard'),
    pretrained_visual_learning_rate=1e-5,new_text_learning_rate=1e-4,epoch0_is_SO_prediction=False,
    transfer_not_original_paper_reproduction=True,source_control='same_text_architecture_and_exact_UDA_source_update_count_without_target_learning')

def identity():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
    up=read(VENDOR/'upstream.json')
    for p,h in up['files'].items():assert sha(VENDOR/p)==h,p
    return dict(commit=head,upstream=up['commit'],upstream_manifest_sha256=sha(VENDOR/'upstream.json'))

def decision(history):
    assert history and [x[0] for x in history]==list(range(0,history[-1][0]+1,5))
    assert history[-1][0]<=30 and all(math.isfinite(x[1]) for x in history)
    best=min(history,key=lambda x:x[1]);epoch=history[-1][0]
    return dict(epoch=epoch,best_epoch=best[0],best_loss=best[1],reason='source_patience' if epoch-best[0]>=15 else 'max_epochs' if epoch==30 else 'running')

def terminal(history):
    d=decision(history);assert d['reason']!='running'
    assert all(decision(history[:i])['reason']=='running' for i in range(1,len(history)))
    return d

def prepare(root,code):
    assert read(REFERENCE/'monitor/independent_final_audit.json')['count']==6
    cache=Path(os.environ['DYFA_TEXT_CACHE']);meta=read(cache/'provenance.json')
    assert sha(cache/'text.pt')==meta['text_sha256'] and not meta['target_video_access'] and not meta['target_GT_access']
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    for i in range(1,7):
        s='S'+str(i);ref=REFERENCE/'formal'/s;entry=root/'formal'/s;entry.mkdir(parents=True)
        cfg=read(ref/'config.json');old=read(Path(cfg['benchmark']['reference'])/'config.json')
        # Only the CPU coordinator reads the benchmark annotation database.
        offline=read(ref/'offline_evaluation.json');full=read(offline['annotation'])['database']
        members={}
        for role in ('source_train','source_val'):
            db=read(ref/(role+'.json'));save(entry/(role+'.json'),db);members[role]=list(db['database'])
        target={k:{f:v[f] for f in ('subset','duration','fps') if f in v} for k,v in full.items() if v['subset'].lower() in old['tgt_train_split']}
        assert target;save(entry/'target_train_unlabeled.json',dict(database=target));members['target_train']=list(target)
        for role in ('source_val','target_val'):
            db=read(ref/(role+'_unlabeled.json'));save(entry/(role+'_unlabeled.json'),db);members[role]=list(db['database'])
        assert len(set().union(*(set(v) for v in members.values())))==sum(map(len,members.values()))
        cfg['opt'].update(learning_rate=1e-5,eta_min=1e-6,epochs=30,warmup=False,warmup_epochs=0)
        cfg['dataset']['json_file']=str(entry/'source_train.json')
        cfg['benchmark']=dict(schema=SCHEMA,spec=SPEC,code=code,shift=s,reference=str(ref),
            reference_config_sha256=sha(ref/'config.json'),reference_complete_sha256=sha(ref/'complete.json'),
            reference_best_sha256=sha(ref/'best.pt'),steps_per_epoch=read(ref/'config.json')['benchmark']['steps_per_epoch'],
            source_train=str(entry/'source_train.json'),source_val=str(entry/'source_val.json'),target_train=str(entry/'target_train_unlabeled.json'),
            text=dict(variant=VARIANT,cache=str(cache/'text.pt'),sha256=sha(cache/'text.pt'),provenance=str(cache/'provenance.json'),provenance_sha256=sha(cache/'provenance.json')),
            inference={r:str(entry/(r+'_unlabeled.json')) for r in ('source_val','target_val')})
        save(entry/'config.json',cfg);save(entry/'offline_evaluation.json',offline)
        save(entry/'inputs.json',dict(members=members,manifests={str(p):sha(p) for p in entry.glob('*.json')},target_GT_removed=True))
    save(root/'plan.json',dict(schema=SCHEMA,spec=SPEC,code=code,shifts=['S'+str(i) for i in range(1,7)],modes=['uda','source_control']))

def guard_audit(path,h,commit):
    assert sha(path)==h;g=read(path);l=read(g['lease']);assert sha(g['lease'])==g['lease_sha256']
    assert l['commit']==commit and l['owner_pid']==g['owner_pid'] and l['observed']['uuid']==g['observed']['uuid']
    assert l['policy']=='shared_without_public_exclusive_lock' and g['observed']['free_mib']>=l['minimum_free_mib']>=12288

def audit(entry):
    import torch,json
    e=Path(entry);c=read(e/'config.json');b=c['benchmark'];complete=read(e/'complete.json')
    assert complete['schema']==SCHEMA and b['spec']==SPEC
    assert sha(b['text']['cache'])==b['text']['sha256'] and sha(b['text']['provenance'])==b['text']['provenance_sha256']
    for p,h in complete['artifacts'].items():assert sha(p)==h,p
    for p,h in read(e/'inputs.json')['manifests'].items():assert sha(p)==h,p
    ref=Path(b['reference'])
    for name in ['config','complete']:assert sha(ref/(name+'.json'))==b['reference_'+name+'_sha256']
    assert sha(ref/'best.pt')==b['reference_best_sha256']
    members=read(e/'inputs.json')['members'];assert len(set().union(*(set(v) for v in members.values())))==sum(map(len,members.values()))
    assert all('annotations' not in x for x in read(b['target_train'])['database'].values())
    acceptance=read(e.parents[1]/'acceptance.json');assert acceptance['code']==b['code']
    for proof in acceptance['reports'].values():
        assert sha(proof['path'])==proof['sha256'];p=read(proof['path']);guard_audit(p['guard'],p['guard_sha256'],b['code']['commit'])
        assert p['full_state_restore'] and p['next_step_numerical'] and p['target_gradient_positive'] and p['source_disabled_path_equal'] and p['source_disabled_calls_official']
        assert p['disabled_SO_dense_equal'] and p['full_strength_semantic_path'] and p['initial_semantic_max_delta']>0 and p['semantic_parameters_updated'] and p['text_counterfactual_max_delta']>0
        assert p['text_projection_gradient_norm']>0 and p['semantic_parameter_count']>0
        routing=p['split_loss_routing'];support=p['split_support']
        assert routing['classification_to_cls']>0 and routing['regression_to_reg']>0
        assert routing['classification_to_reg']==routing['regression_to_cls']==0
        assert 0<support['regression_proposals']<=support['classification_proposals'] and support['regression_points']>0
    stop=read(e/'uda/termination.json')['decision']['epoch']
    for mode in ('uda','source_control'):
        d=e/mode;t=read(d/'termination.json');h=t['history'];rows=read(d/'metrics.json')
        if mode=='uda':assert terminal(h)==t['decision']
        else:assert t['decision']['epoch']==stop and t['decision']['best_epoch']==min(h,key=lambda x:x[1])[0]
        assert [r['epoch'] for r in rows]==list(range(1,stop+1))
        assert [x[0] for x in h]==list(range(0,stop+1,5))
        assert [[r['epoch'],r['source_validation_loss']] for r in rows if 'source_validation_loss' in r]==h[1:]
        assert all(r['steps']==b['steps_per_epoch'] for r in rows)
        assert all(r['target_training_batches']==(b['steps_per_epoch'] if mode=='uda' else 0) for r in rows)
        for r in rows:
            assert all(math.isfinite(r[k]) for k in ('final_loss','cls_loss','reg_loss','target_cls','target_reg','ramp'))
            if mode=='uda' and r['target_regression_points']>0:
                assert set(r['target_gradient_norms'])=={'backbone','neck','cls_head','reg_head','semantic'}
                assert all(v>0 and math.isfinite(v) for v in r['target_gradient_norms'].values())
        init=read(d/'initialization.json');assert init['reference_best_sha256']==b['reference_best_sha256']
        assert init['student_sha256']==init['teacher_sha256']
        assert init['detector_sha256']==init['reference_ema_sha256'] and init['semantic_parameter_count']>0
        for name in ('best','latest'):
            state=torch.load(d/(name+'.pt'),map_location='cpu');assert state['epoch']==t['decision']['best_epoch' if name=='best' else 'epoch']
            assert state['config']==c and state['mode']==mode and state['step']==state['epoch']*b['steps_per_epoch']
            assert all(k in state for k in ('model','ema','optimizer','scheduler','normalizers','rng','target_rng','history','metrics','augmentation_rng'))
            assert state['rng']['cuda'][0].numel()>0 and state['target_rng']['cuda'][0].numel()>0
            if name=='latest':assert state['history']==h and state['metrics']==rows
            else:assert state['history']==[x for x in h if x[0]<=state['epoch']]
            if name=='latest':assert state['ledger_digest']==t['ledger_digest']
            del state
        events=[json.loads(l) for l in (d/'events.jsonl').read_text().splitlines()]
        assert [x['step'] for x in events]==list(range(1,stop*b['steps_per_epoch']+1))
        digest=''
        for x in events:
            import hashlib
            received=x.pop('sha256');assert x['previous']==digest
            digest=hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert received==digest
            assert set(x['source_videos'])<=set(members['source_train'])
            assert set(x['target_videos'])<=set(members['target_train'])
            if mode=='source_control':assert x['target_videos']==[]
        assert digest==t['ledger_digest']
        if mode=='uda':
            assert sum(x['target_positive_points'] for x in events)>0
            assert sum(x['target_regression_points'] for x in events)>0
            assert all(0<=x['target_regression_proposals']<=x['target_proposals'] for x in events)
            assert all(0<=x['target_cls_weight']<=1 for x in events)
            for epoch in range(5,stop+1,5):
                snap=torch.load(d/('epoch_%03d.pt'%epoch),map_location='cpu')
                assert snap['epoch']==epoch and snap['config']==c and snap['mode']==mode
                del snap
        guard_audit(t['guard'],t['guard_sha256'],b['code']['commit'])
        for role in ('source_val','target_val'):
            manifest=c['benchmark']['inference'][role];db=read(manifest)['database']
            assert all('annotations' not in x for x in db.values())
            proof=read(d/(role+'_inference.json'))
            assert proof['videos']==list(db) and not proof['target_GT_used'] and proof['manifest_sha256']==sha(manifest)
            report=read(d/'evaluation'/(role+'.json'));assert report['prediction_sha256']==sha(d/(role+'.npz'))
            assert report['selected_epoch']==t['decision']['best_epoch']
            # best0表示新架构未被源选模改进，不再声称等于原SO。
    g=read(e/'gpu_complete.json');guard_audit(g['guard'],g['guard_sha256'],b['code']['commit'])
    uda=read(e/'uda/evaluation/target_val.json')['map_0.1_to_0.5'];control=read(e/'source_control/evaluation/target_val.json')['map_0.1_to_0.5']
    so=read(ref/'evaluation/target_val.json')['map_0.1_to_0.5']
    return dict(shift=b['shift'],map=uda,reference_so_map=so,control_map=control,delta_pp=100*(uda-so),delta_control_pp=100*(uda-control),
                complete=str(e/'complete.json'),complete_sha256=sha(e/'complete.json'))
