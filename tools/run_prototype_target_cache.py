"""四卡目标实例缓存完整评估；固定既有六个检测器，不训练、不查询新VLM。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, code_identity, termination
from tools.prototype_target_cache import SCHEMA, SHIFTS, specification, metadata_only

REFERENCE = Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029')
INVENTORY = Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/inputs/feature_inventory.json')
ACTIVE = {}


def prepare(root, reference, inventory, identity):
    final = read(reference/'final_audit.json')
    if final.get('status') != 'passed' or final.get('count') != 12:
        raise ValueError('缺少已完成原版Prototype矩阵')
    completed = {r['complete']: r['complete_sha256'] for r in final['results']}
    inputs = dict(schema=SCHEMA, code=identity, specification=specification(),
                  reference_root=str(reference), reference_final_sha256=sha(reference/'final_audit.json'), entries={})
    for shift in SHIFTS:
        old = reference/'formal/control'/shift
        if completed.get(str(old/'complete.json')) != sha(old/'complete.json'):
            raise ValueError('旧完成身份或SHA改变')
        cfg, complete = read(old/'config.json'), read(old/'complete.json')
        cap = cfg['prototype_capacity']
        from libs.models.SADA.prototype_capacity import specification as old_spec
        if cfg['seed'] != 123 or cfg['loader']['batch_size'] != 2 or cap['specification'] != old_spec('control'):
            raise ValueError('只能使用已有seed123/batch2/原128维Prototype')
        for rel in ['best.pt', 'config.json', 'termination.json', 'evaluation/source_val.json', 'evaluation/target_val.json']:
            if sha(old/rel) != complete['artifacts'][str(old/rel)]:
                raise ValueError('原参照工件改变: '+str(old/rel))
        for path, digest in [(cap['prepared_cache'], cap['prepared_cache_sha256']),
                             (cfg['baseline']['target_manifest'], cfg['baseline']['target_manifest_sha256']),
                             (cfg['baseline']['annotation_for_offline_evaluation'], cfg['baseline']['annotation_sha256'])]:
            if sha(path) != digest:
                raise ValueError('输入工件SHA改变')
        db = read(cfg['baseline']['annotation_for_offline_evaluation'])['database']
        target = read(cfg['baseline']['target_manifest'])
        if any('annotations' in v for v in target['database'].values()):
            raise ValueError('目标训练文件携带GT')
        entry = root/'formal'/shift
        entry.mkdir(parents=True, exist_ok=False)
        save(entry/'config.json', cfg)
        manifests = {'target_train': target}
        for domain in ['source', 'target']:
            key = 'src_val_split' if domain == 'source' else 'tgt_val_split'
            manifests[domain+'_val'] = metadata_only(db, cfg[key])
        sets = [set(v['database']) for v in manifests.values()]
        if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
            raise ValueError('缓存和验证成员交叉')
        for name, body in manifests.items():
            if not body['database']:
                raise ValueError('空成员集合')
            save(entry/(name+'_unlabeled.json'), body, sort_keys=False)
        # 实际数据接口在CPU先走一遍，包含仓库按batch接收的设备转换函数。
        from tools.baseline_engine import dataset
        from libs.trainer.da_trainer import move_to_device
        for name in manifests:
            splits = cfg['tgt_train_split'] if name=='target_train' else cfg['src_val_split'] if name=='source_val' else cfg['tgt_val_split']
            ds = dataset(cfg, entry/(name+'_unlabeled.json'), splits, False, True)
            sample = move_to_device([ds[0]], 'cpu')[0]
            if sample['segments'].numel() or sample['labels'].numel() or sample['feats'].shape[0] != 2304:
                raise ValueError('真实CPU入口的目标GT隔离或SlowFast维度错误')
        record = dict(reference=str(old), reference_complete_sha256=sha(old/'complete.json'),
                      checkpoint_sha256=complete['artifacts'][str(old/'best.pt')],
                      config_sha256=sha(entry/'config.json'), cache_sha256=cap['prepared_cache_sha256'],
                      manifests={name: dict(count=len(v['database']), sha256=sha(entry/(name+'_unlabeled.json')))
                                 for name, v in manifests.items()})
        save(entry/'inputs.json', record)
        inputs['entries'][shift] = record
    # 全量冻结特征一次核验；不让四卡各自重复读取同一份大清单。
    features = read(inventory)
    from concurrent.futures import ThreadPoolExecutor
    def verify(row):
        if sha(row['path']) != row['sha256']:
            raise ValueError('冻结特征改变: '+row['path'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, features))
    inventory_paths = {row['path'] for row in features}
    for shift in SHIFTS:
        cfg = read(root/'formal'/shift/'config.json')
        for name in ('target_train', 'source_val', 'target_val'):
            for vid in read(root/'formal'/shift/(name+'_unlabeled.json'))['database']:
                f = Path(cfg['dataset']['feat_folder'])/((cfg['dataset'].get('file_prefix') or '')+vid+cfg['dataset']['file_ext'])
                if str(f) not in inventory_paths:
                    raise ValueError('使用了冻结特征清单外的文件')
    inputs.update(feature_inventory=str(inventory), feature_inventory_sha256=sha(inventory),
                  feature_count=len(features), status='passed')
    save(root/'inputs.json', inputs)
    return inputs


def inference_worker(args):
    from tools.baseline_gpu import first_cuda_guard
    guard = first_cuda_guard()  # 必须在第一次CUDA播种、建模之前执行。
    started = time.monotonic()
    import random
    import numpy as np
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.prototype_capacity import KNOWN_LABELS, feature_interval
    from libs.utils import fix_random_seed
    from libs.trainer.da_trainer import move_to_device
    from tools import baseline_engine as base
    from tools.prototype_target_cache import (selected_records, detector_features, detector_forward,
        encode_segments, duration_levels, memory_support, rescore, assert_same_prediction,
        evaluation_columns, append_prediction)

    entry = args.entry
    cfg, bound = read(entry/'config.json'), read(entry/'inputs.json')
    identity = code_identity()
    if identity != read(args.output/'plan.json')['code'] or sha(entry/'config.json') != bound['config_sha256']:
        raise ValueError('当前clean代码或配置身份改变')
    old = Path(bound['reference'])
    if sha(old/'best.pt') != bound['checkpoint_sha256']:
        raise ValueError('原源选checkpoint SHA改变')
    os.chdir(args.data_root)
    fix_random_seed(cfg['seed'], include_cuda=True)
    torch.cuda.manual_seed(cfg['seed'])
    model = make_meta_arch('sada', deepcopy(cfg), device='cuda:0').to('cuda:0')
    saved = torch.load(old/'best.pt', map_location='cpu')
    if saved['epoch'] != termination(read(old/'termination.json')['history'])['best_epoch']:
        raise ValueError('没有使用原源验证最佳EMA')
    epoch = saved['epoch']
    model.load_state_dict(saved['ema_state_dict'], strict=True)
    del saved
    model.eval().requires_grad_(False)
    branch, spec = model.rq3_capacity.branch, specification()
    if branch.classifier.in_features != 128 or branch.classifier.out_features != 10:
        raise ValueError('不是原128维10类辅助分支')
    before = base.tensor_digest(model.state_dict())
    def rng():
        return base.tensor_digest((random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state(0)))
    initial_rng = rng()

    def dataset(name):
        path = entry/(name+'_unlabeled.json')
        if sha(path) != bound['manifests'][name]['sha256']:
            raise ValueError('无标签成员文件改变')
        return base.dataset(cfg, path, cfg['tgt_train_split'] if name == 'target_train' else
                            cfg['src_val_split'] if name == 'source_val' else cfg['tgt_val_split'], False, True)

    target_ds = dataset('target_train')
    cap = cfg['prototype_capacity']
    if sha(cap['prepared_cache']) != bound['cache_sha256']:
        raise ValueError('VLM缓存改变')
    records, rejected = selected_records(read(cap['prepared_cache']),
                                         {v['id'] for v in target_ds.data_list}, KNOWN_LABELS)
    embeddings, labels, confidence, video_ids, ledger = [], [], [], [], []
    video_number = {v: i for i, v in enumerate(records)}
    sampling_checked = False
    with torch.no_grad():
        for index, info in enumerate(target_ds.data_list):
            vid = info['id']
            if vid not in records:
                continue
            video = move_to_device([target_ds[index]], 'cuda:0')[0]
            if video['segments'].numel() or video['labels'].numel():
                raise ValueError('缓存构建读取了目标GT')
            features, _ = detector_features(model, video)
            rows, boxes, levels = [], [], []
            for row in records[vid]:
                label = row['memory_label']
                if not bool(branch.source_prototype_ready[label]):
                    rejected.append(dict(candidate_id=row['candidate_id'], reason='source-prototype-not-ready'))
                    continue
                left, right = feature_interval(row, video['metadata'])
                left, right = max(0., left), min(float(video['feats'].shape[-1]), right)
                if right <= left:
                    rejected.append(dict(candidate_id=row['candidate_id'], reason='empty-visible-interval'))
                    continue
                rows.append(row); boxes.append([left, right]); levels.append(int(row['source_fpn_levels'][0]))
            if not rows:
                continue
            boxes = features[0].new_tensor(boxes)
            levels = torch.tensor(levels, device='cuda:0', dtype=torch.long)
            vector = encode_segments(branch, features, boxes, levels, model.fpn_strides)
            if not sampling_checked:
                sample = len(boxes)  # 单视频原40名单不超过40项，比较同一完整batch。
                original = branch.encode(features, [dict(segments=boxes[:sample], levels=levels[:sample])], model.fpn_strides)['embedding']
                difference = float((original-vector[:sample]).abs().max())
                if not torch.allclose(original, vector[:sample], rtol=1e-5, atol=2e-6):
                    raise ValueError('向量化采样与原编码器不等价')
                sampling_checked = True
                save(entry/'sampling_equivalence.json', dict(status='passed', maximum_absolute_error=difference))
            embeddings.append(vector)
            labels.extend(row['memory_label'] for row in rows)
            confidence.extend(row['memory_confidence'] for row in rows)
            video_ids.extend([video_number[vid]]*len(rows))
            ledger.extend(dict(candidate_id=row['candidate_id'], video_id=vid, label=row['memory_label'],
                               confidence=row['memory_confidence'], level=int(level), segment=box)
                          for row, level, box in zip(rows, levels.cpu().tolist(), boxes.cpu().tolist()))
            print('BANK_VIDEO', vid, len(rows), flush=True)
    if not embeddings:
        raise ValueError('没有可用目标实例，不能冒充已验证缓存')
    bank = dict(embedding=torch.cat(embeddings), label=torch.tensor(labels, device='cuda:0'),
                confidence=torch.tensor(confidence, device='cuda:0'), video=torch.tensor(video_ids, device='cuda:0'))
    if not torch.isfinite(bank['embedding']).all():
        raise ValueError('非有限实例表示')
    torch.save({k: v.cpu() for k, v in bank.items()}, entry/'bank.pt')
    bank_before = base.tensor_digest(bank)
    save(entry/'bank_ledger.json', dict(schema=SCHEMA, records=ledger, rejected=rejected,
        cache_sha256=bound['cache_sha256'], checkpoint_sha256=bound['checkpoint_sha256'],
        selected_budget=40, target_gt_used=False, validation_writes=0))

    original_checked = False
    observed = {}
    for domain in ('source', 'target'):
        domain_started = time.monotonic()
        ds = dataset(domain+'_val')
        dl = base.loader(ds, False, torch.Generator().manual_seed(cfg['seed']+2000006), 1, 2)
        columns = evaluation_columns()
        stats = dict(videos=len(ds), detections=0, changed_scores=0, support_sum=0.)
        for i, batch in enumerate(dl):
            video = move_to_device(batch, 'cuda:0')[0]
            if video['segments'].numel() or video['labels'].numel():
                raise ValueError('推理阶段接触验证GT')
            features, prediction = detector_forward(model, video)
            if not original_checked:
                legacy = model(video_list_src=[video], video_list_tgt=[video], mode='test', domain=domain, curr_epoch=epoch)
                assert_same_prediction(prediction, legacy[0][0])
                assert_same_prediction(prediction, legacy[1][0])
                save(entry/'inference_equivalence.json', dict(status='passed', video_id=video['metadata']['video_id'],
                    compared_fields=['segments', 'scores', 'labels', 'segments_pre_nms', 'scores_pre_nms', 'labels_pre_nms'],
                    exact_tensor_equality=True))
                original_checked = True
            metadata = video['metadata']
            boxes = (prediction['segments'].to('cuda:0')*float(metadata['fps'])-.5*float(metadata['feat_num_frames']))/float(metadata['feat_stride'])
            levels = duration_levels(boxes, model.reg_range)
            query = encode_segments(branch, features, boxes, levels, model.fpn_strides, spec['query_chunk'])
            support = memory_support(query, prediction['labels'].to('cuda:0'), bank, spec)
            updated = rescore(prediction['scores'].to('cuda:0'), support, spec['alpha']).cpu()
            if not (entry/'acceptance.json').exists():
                save(entry/'acceptance.json', dict(status='passed', schema=SCHEMA, code=identity,
                    real_video=video['metadata']['video_id'], bank_size=len(labels), embedding_dim=128,
                    original_detector_exactly_equal=True, sampling_equivalence_sha256=sha(entry/'sampling_equivalence.json'),
                    no_optimizer=True, target_gt_used=False, validation_cache_writes=0,
                    first_cuda_guard_sha256=sha(guard), specification=spec))
            append_prediction(columns, prediction, updated)
            stats['detections'] += len(updated)
            stats['changed_scores'] += int((updated != prediction['scores']).sum())
            stats['support_sum'] += float(support.sum())
            if i % 10 == 0:
                print('INFERENCE', domain, i, len(ds), stats['changed_scores'], flush=True)
        np.savez(entry/(domain+'_predictions.npz'), **{k: np.asarray(v) for k, v in columns.items()})
        stats['seconds'] = time.monotonic()-domain_started
        observed[domain] = stats
    after, final_rng = base.tensor_digest(model.state_dict()), rng()
    bank_after = base.tensor_digest(bank)
    if before != after or initial_rng != final_rng or bank_before != bank_after:
        raise ValueError('缓存评估改变了模型权重/缓冲区或全局随机流')
    if not all(v['changed_scores'] > 0 for v in observed.values()):
        raise ValueError('缓存没有实际影响预测分数')
    save(entry/'inference_proof.json', dict(status='passed', schema=SCHEMA, code=identity,
        state_before_sha256=before, state_after_sha256=after, rng_before=initial_rng, rng_after=final_rng,
        model_unchanged=True, no_optimizer=True, selected_epoch=epoch, checkpoint_sha256=bound['checkpoint_sha256'],
        bank_size=len(labels), bank_class_counts={str(c): labels.count(c) for c in range(10)},
        bank_state_before=bank_before, bank_state_after=bank_after, seconds=time.monotonic()-started,
        observations=observed, gpu_guard=str(guard), gpu_guard_sha256=sha(guard), specification=spec))


def evaluate_worker(args):
    import numpy as np
    from libs.evaluation import ANETdetection
    entry = args.entry
    cfg, bound = read(entry/'config.json'), read(entry/'inputs.json')
    original = Path(bound['reference'])
    annotation = cfg['baseline']['annotation_for_offline_evaluation']
    if sha(annotation) != cfg['baseline']['annotation_sha256']:
        raise ValueError('离线评估标注改变')
    outputs = {}
    for domain in ('source', 'target'):
        path = entry/(domain+'_predictions.npz')
        with np.load(path, allow_pickle=False) as payload:
            columns = {k: payload[k] for k in payload.files}
        splits = cfg['src_val_split'] if domain == 'source' else cfg['tgt_val_split']
        for mode, score in [('control', 'score'), ('cache', 'cache-score')]:
            pred = {k: columns[k] for k in ('video-id', 't-start', 't-end', 'label')}
            pred['score'] = columns[score]
            evaluator = ANETdetection(str(annotation), splits, num_workers=2, action_subsampling='10_majority_classes')
            ap, mean, *_ = evaluator.evaluate(pred, verbose=True)
            result = dict(map_by_tiou=[float(v) for v in ap],
                mode=mode, domain=domain, checkpoint_sha256=bound['checkpoint_sha256'],
                prediction_sha256=sha(path), split=splits, parameter_search=False)
            result['map_0.1_to_0.5'] = float(mean)
            save(entry/'evaluation'/(domain+'_'+mode+'.json'), result)
            outputs[domain+'_'+mode] = result
        reference_map = read(original/'evaluation'/(domain+'_val.json'))['map_0.1_to_0.5']
        difference = outputs[domain+'_control']['map_0.1_to_0.5']-reference_map
        if abs(difference) > 1e-7:
            raise ValueError('原版检测复现不一致，不解释缓存收益: '+domain+' '+str(difference))
    paths = ['config.json', 'inputs.json', 'bank.pt', 'bank_ledger.json', 'sampling_equivalence.json', 'acceptance.json',
             'inference_equivalence.json', 'inference_proof.json', 'source_predictions.npz', 'target_predictions.npz']
    paths += [n+'_unlabeled.json' for n in ('target_train', 'source_val', 'target_val')]
    paths += ['evaluation/'+k+'.json' for k in outputs]
    save(entry/'complete.json', dict(status='passed', schema=SCHEMA, shift=entry.name,
        code=code_identity(), specification=specification(), trained=False, reference=str(original),
        reference_complete_sha256=bound['reference_complete_sha256'],
        artifacts={str(entry/n): sha(entry/n) for n in paths}))


def audit_entry(entry):
    entry = Path(entry)
    record, inputs, proof = read(entry/'complete.json'), read(entry/'inputs.json'), read(entry/'inference_proof.json')
    if record['schema'] != SCHEMA or record['specification'] != specification() or record['trained']:
        raise ValueError('缓存实验身份改变')
    if record['code'] != proof['code'] or record['code'] != read(entry.parents[1]/'plan.json')['code']:
        raise ValueError('完成项代码与当前矩阵代码不一致')
    for path, digest in record['artifacts'].items():
        if sha(path) != digest:
            raise ValueError('缓存完成工件SHA改变: '+path)
    original = Path(inputs['reference'])
    if sha(original/'complete.json') != inputs['reference_complete_sha256']:
        raise ValueError('旧参照完成工件改变')
    if (not proof['model_unchanged'] or proof['state_before_sha256'] != proof['state_after_sha256'] or
            proof['rng_before'] != proof['rng_after'] or proof['bank_state_before'] != proof['bank_state_after']):
        raise ValueError('固定模型/RNG证明失败')
    if sha(proof['gpu_guard']) != proof['gpu_guard_sha256']:
        raise ValueError('GPU首CUDA凭证改变')
    guard = read(proof['gpu_guard']); lease = read(guard['lease'])
    if (sha(guard['lease']) != guard['lease_sha256'] or guard['observed']['uuid'] != lease['observed']['uuid'] or
            guard['observed']['free_mib'] < lease['minimum_free_mib']):
        raise ValueError('公共卡锁/共享显存凭证不一致')
    manifests = {n: set(read(entry/(n+'_unlabeled.json'))['database']) for n in ('target_train', 'source_val', 'target_val')}
    ledger = read(entry/'bank_ledger.json')
    bank_videos = {x['video_id'] for x in ledger['records']}
    if (not bank_videos <= manifests['target_train'] or bank_videos & (manifests['source_val'] | manifests['target_val']) or
            ledger['target_gt_used'] or ledger['validation_writes'] != 0 or len(ledger['records']) > 40):
        raise ValueError('缓存成员/预算/GT隔离失败')
    if len({x['candidate_id'] for x in ledger['records']}) != len(ledger['records']):
        raise ValueError('缓存候选重复')
    cfg = read(entry/'config.json')
    cache = read(cfg['prototype_capacity']['prepared_cache'])
    selected = {x['candidate_id']: x for rows in cache['by_video'].values() for x in rows if x['participation_selected']}
    for row in ledger['records']:
        original_row = selected[row['candidate_id']]
        q = original_row['class_probabilities']
        teacher_label = q.index(max(q))
        # class_id可能记录历史检测器类别；缓存标签必须核对VLM概率，不能偷换成它。
        if (row['video_id'] != original_row['video_id'] or row['label'] != teacher_label or
                row['level'] != original_row['source_fpn_levels'][0]):
            raise ValueError('缓存不是原40名单的类别/视频/FPN层')
    if not read(entry/'inference_equivalence.json')['exact_tensor_equality']:
        raise ValueError('原检测前向未证明等价')
    values = {domain: {mode: read(entry/'evaluation'/(domain+'_'+mode+'.json'))['map_0.1_to_0.5']
                       for mode in ('control', 'cache')} for domain in ('source', 'target')}
    for domain in ('source', 'target'):
        expected = read(original/'evaluation'/(domain+'_val.json'))['map_0.1_to_0.5']
        if abs(values[domain]['control']-expected) > 1e-7:
            raise ValueError('独立复核原检测mAP不一致')
    return dict(shift=entry.name, complete=str(entry/'complete.json'), complete_sha256=sha(entry/'complete.json'),
                source=values['source'], target=values['target'],
                delta_pp=100*(values['target']['cache']-values['target']['control']), bank_size=proof['bank_size'])


def spawn(args, shift, stage, lease=None):
    entry = args.output/'formal'/shift
    env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    if lease is not None:
        env.update(CUDA_VISIBLE_DEVICES=lease['uuid'], SADA_BASELINE_LEASE=lease['lease'],
                   SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
    command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--stage', stage,
               '--entry', str(entry), '--output', str(args.output), '--data-root', str(args.data_root)]
    with (entry/(stage+'.log')).open('x') as log:
        return subprocess.Popen(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(128+signum)


def main(args):
    from tools.run_sada_baseline import acquire
    args.output = args.output.resolve(); args.data_root = args.data_root.resolve()
    identity = code_identity()
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'plan.json', dict(schema=SCHEMA, code=identity, specification=specification(), shifts=SHIFTS,
        physical_gpus=[0, 1, 2, 3], maximum_own_cuda_per_gpu=1, trained=False, cpu_evaluators=2,
        reference=str(args.reference), scientific_comparisons='same-checkpoint-cache-off-vs-on',
        parameter_search=False, target_validation_cache_updates=False))
    completed=[];pending=list(SHIFTS);evaluation=[]
    try:
        save(args.output/'state.json', dict(phase='input_audit', pid=os.getpid(), completed=[], active={}))
        prepare(args.output, args.reference, args.inventory, identity)
        while pending or evaluation or ACTIVE:
            for key, task in list(ACTIVE.items()):
                code = task['process'].poll()
                if code is None:
                    continue
                if task.get('context'):
                    task['context'].__exit__(None, None, None)
                del ACTIVE[key]
                if code:
                    raise RuntimeError('阶段失败: '+key+' exit='+str(code))
                if task['stage'] == 'infer':
                    evaluation.append(task['shift'])
                else:
                    row = audit_entry(args.output/'formal'/task['shift'])
                    completed.append(row)
                    print('COMPLETE', json.dumps(row), flush=True)
            while evaluation and sum(t['stage']=='evaluate' for t in ACTIVE.values()) < 2:
                shift=evaluation.pop(0)
                ACTIVE[shift+'/evaluate']=dict(process=spawn(args,shift,'evaluate'),shift=shift,stage='evaluate',gpu=None)
            for shift in list(pending):
                found=acquire([0,1,2,3],args.data_root,args.output,12288,identity['commit'])
                if not found:
                    break
                context,lease=found
                try:process=spawn(args,shift,'infer',lease)
                except BaseException:
                    context.__exit__(None,None,None);raise
                pending.remove(shift)
                ACTIVE[shift+'/infer']=dict(process=process,shift=shift,stage='infer',gpu=lease['index'],context=context)
            save(args.output/'state.json', dict(phase='cache_and_evaluation', pid=os.getpid(), completed=completed,
                pending=pending, evaluation_pending=evaluation,
                active={k:dict(pid=t['process'].pid,shift=t['shift'],stage=t['stage'],gpu=t['gpu']) for k,t in ACTIVE.items()}))
            time.sleep(3)
        if {r['shift'] for r in completed} != set(SHIFTS) or len(completed) != 6:
            raise ValueError('不是精确六方向')
        save(args.output/'final_audit.json',dict(status='passed',schema=SCHEMA,count=6,trained=False,results=completed,
            control_mean=sum(r['target']['control'] for r in completed)/6,
            cache_mean=sum(r['target']['cache'] for r in completed)/6,
            mean_delta_pp=sum(r['delta_pp'] for r in completed)/6))
        save(args.output/'state.json',dict(phase='complete',completed=completed,active={},pid=os.getpid()))
    except BaseException as error:
        save(args.output/'error.json',dict(type=type(error).__name__,message=str(error)))
        save(args.output/'state.json',dict(phase='stopped' if isinstance(error,SystemExit) else 'failed',completed=completed,
                                          active={},pid=os.getpid()))
        raise
    finally:
        for task in ACTIVE.values():
            if task['process'].poll() is None:
                os.killpg(task['process'].pid, signal.SIGTERM)
        for task in ACTIVE.values():
            try:task['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(task['process'].pid, signal.SIGKILL);task['process'].wait()
            if task.get('context'):task['context'].__exit__(None,None,None)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--inventory',type=Path,default=INVENTORY)
    parser.add_argument('--stage',choices=('infer','evaluate'))
    parser.add_argument('--entry',type=Path)
    args=parser.parse_args()
    if args.stage=='infer':inference_worker(args)
    elif args.stage=='evaluate':evaluate_worker(args)
    else:
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        main(args)
