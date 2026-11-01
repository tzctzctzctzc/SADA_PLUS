"""组合格式修复：成功请求逐SHA复用，失败组合请求保留且计入技术重试成本。"""
import argparse,os,shutil,subprocess,sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.cott_protocol import *

def reuse(root,previous):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=Path(root);previous=Path(previous);old=read(previous/'plan.json')
    assert not list((previous/'videos').glob('*/complete.json'))
    assert sha(previous/'manifest.json')==old['manifest_sha256']
    assert sha(previous/'selection.json')==old['selection_sha256']
    assert sha(previous/'evaluation_inputs/source_only.json')==old['evaluation_inputs_sha256']
    assert dict(old['spec'],grouping_contract_version=SPEC['grouping_contract_version'],
        grouping_serialization=SPEC['grouping_serialization'])==SPEC
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
    root.mkdir(parents=True,exist_ok=False)
    for name in ['selection.json','evaluation_inputs/source_only.json','model_identity.json']:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(previous/name,target)
    manifest=read(previous/'manifest.json');no_annotations(manifest);manifest.update(code=code,spec=SPEC)
    save(root/'manifest.json',manifest);calls={};failed={};cost=[]
    for d in sorted((previous/'calls').glob('*/*')):
        if not d.is_dir():continue
        req=read(d/'request.json')
        if (d/'result.json').exists():
            assert req['stage'] in ['transition','direct']
            result=read(d/'result.json');assert digest(req)==result['request_sha256']
            assert sha(d/'request.json')==result['request_file_sha256']
            for f,h in result['attempt_sha256'].items():assert sha(d/f)==h
            validate(result['parsed'],req['stage'],req['lo'],req['hi'],req['pair_count'])
            dest=root/'calls'/d.parent.name/d.name;shutil.copytree(d,dest)
            calls[d.parent.name+'/'+d.name]=dict(previous=str(d),result_sha256=sha(d/'result.json'),
                request_sha256=result['request_sha256'],artifacts={p.name:sha(p) for p in d.glob('*.json')})
        else:
            for p in d.glob('attempt_*.json'):
                failed[str(p)]=sha(p);a=read(p);cost.append({k:a[k] for k in ['input_tokens','output_tokens','elapsed_sec','status']})
    ancestors=[];earlier=previous/'accepted_prefix_reuse.json'
    if earlier.exists():
        prior=read(earlier);ancestors.append(dict(path=str(earlier),sha256=sha(earlier)))
        ancestors+=prior.get('ancestor_receipts',[])
        failed.update(prior['failed_requests_sha256']);cost=prior['prior_failed_contract_cost']+cost
        # 首次transition契约失败也是技术成本，不因跨root接续而丢失。
        preparation=Path(prior['previous'])/'prepared_reuse.json'
        if preparation.exists():
            assert sha(preparation)==prior['previous_preparation_reuse_sha256']
            ancestors.append(dict(path=str(preparation),sha256=sha(preparation)))
            for name,h in read(preparation)['failed_attempt_sha256'].items():
                if name in failed:continue
                assert sha(name)==h;failed[name]=h;a=read(name)
                cost.append({k:a[k] for k in ['input_tokens','output_tokens','elapsed_sec','status']})
    binding=dict(previous=str(previous),previous_plan_sha256=sha(previous/'plan.json'),calls=calls,
        failed_requests_sha256=failed,prior_failed_contract_cost=cost,
        previous_preparation_reuse_sha256=sha(previous/'prepared_reuse.json') if (previous/'prepared_reuse.json').exists() else None,
        ancestor_receipts=ancestors,semantic_change=False,
        representation_change='consecutive transition index spans with constrained greedy legal serialization; finer boundaries remain in visual refinement',
        reason='paper Figure 10 requires non-overlap; constrain decoding rather than trim/drop output; no GT/model/frame/sample changes')
    save(root/'accepted_prefix_reuse.json',binding)
    save(root/'plan.json',dict(old,code=code,spec=SPEC,manifest_sha256=sha(root/'manifest.json'),successful_prefix_reuse_sha256=sha(root/'accepted_prefix_reuse.json')))
    save(root/'state.json',dict(phase='prepared',complete_videos=0,total_videos=len(manifest['videos']),accepted_calls_reused=len(calls)))
    print('REUSED_SUCCESSFUL_CALLS',len(calls),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--previous',required=True);a=p.parse_args();reuse(a.root,a.previous)
