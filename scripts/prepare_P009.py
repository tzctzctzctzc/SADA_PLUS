import json,hashlib
from pathlib import Path
factor=json.loads(Path('configs/P006_permutation_factorial.json').read_text())
lad=json.loads(Path('configs/P007_lad_bridge.json').read_text())
cal=lad['calibration'];ev={r['id']:r for r in lad['evaluation']}
ids=sorted(factor['ids'],key=lambda i:hashlib.sha256(('P009:'+i).encode()).hexdigest())
donors={idx:cal[j%len(cal)] for j,idx in enumerate(ids)}
used={h for idx in factor['ids'] for h in ev[idx]['image_hashes']}
assert not used & {h for r in cal for h in r['image_hashes']}
conditions=[{'id':'baseline','scale':False,'source':None},{'id':'scale','scale':True,'source':None}]
for source in ['clean','donor']:
 for kind in ['start','end','both']:
  conditions.append({'id':source+'_'+kind,'scale':True,'source':source,'kind':kind,'shift':0})
conditions += [{'id':'donor_shift_start','scale':True,'source':'donor','kind':'start','shift':1},{'id':'donor_shift_end','scale':True,'source':'donor','kind':'end','shift':1},{'id':'donor_shift_both','scale':True,'source':'donor','kind':'both','shift':1},
 {'id':'base_donor_both','scale':False,'source':'donor','kind':'both','shift':0},
 {'id':'base_donor_shift_both','scale':False,'source':'donor','kind':'both','shift':1}]
cfg={'id':'P009','purpose':'Causal localization of a fixed delimiter-scaling failure, not a deployable method',
 'model':'Qwen2.5-VL-7B-Instruct','factorial':factor,'donors':donors,'conditions':conditions,'n_rows':128,'n_forward_conditions':1664,
 'max_pixels':1605632,'max_new_tokens':16,'dtype':'bfloat16','attention_backend':'sdpa',
 'capture':'Unmodified-model input_layernorm output immediately before every decoder attention; start/end vectors stored separately',
 'patch':'Restore normalized vectors only, retaining target token/RoPE positions; token kind and donor order fixed before observing results',
 'donor_preprocessing':'Independent calibration images resized to each target image native dimensions before same Qwen preprocessing; assert identical input IDs and image grid',
 'negative_control':'Donor vectors shifted by one image position within each token kind, with original target RoPE retained',
 'integrity':'Clean both must reproduce baseline generation and first-step scores; it is a mathematical implementation control, not scientific gain',
 'decision':'If same-position unrelated donor fails to rescue or performs like shifted donor, reject the simple structural-boundary template mechanism. Clean recovery alone is insufficient.',
 'limitations':['Eight previously examined Mantis questions, repeated permutations dependent','Same-position vs shifted donors confound abstract ordinal index with absolute token/RoPE-derived context; successful pilot still needs length/position controls','No tuning or claim on Muir heldout']}
Path('configs/P009_boundary_restore.json').write_text(json.dumps(cfg,indent=2))
print('P009 registered donor map', {i:r['id'] for i,r in donors.items()})
