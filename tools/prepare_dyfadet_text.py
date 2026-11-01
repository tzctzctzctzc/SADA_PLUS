"""仅根据固定类别生成CLIP文本缓存；CPU运行，不读任何视频或目标标注。"""
import argparse,hashlib,json,sys,urllib.request,shutil
from pathlib import Path
import torch

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'vendor/clip_text'))
from model import build_model
from simple_tokenizer import SimpleTokenizer

LABELS=['TAKE','PUT','WASH','OPEN','CLOSE','INSERT','TURN_ON','CUT','TURN_OFF','POUR']
# 三个阶段按时间顺序保留；整体描述用于Ti-FAD类语义及PDA第四分支。
# 作者为本实现LLM，不冒称调用了原论文GPT-4o或看过目标视频。
DESCRIPTIONS=[
 ['A hand approaches an object resting on a surface.','A hand grasps and lifts the object.','The object is held away from its original resting place.','A person takes or picks up an object.'],
 ['A hand holds an object above a resting place.','The hand lowers and places the object on a surface.','The object rests on the surface and the hand releases it.','A person puts or places an object down.'],
 ['Hands bring an object into contact with water or a cleaning material.','Hands rinse rub or scrub the object to clean it.','The rinsing or scrubbing ends and the object is removed from the cleaning area.','A person washes or cleans an object.'],
 ['An object such as a door lid or container is closed.','A hand moves the door lid or closure away from its closed position.','The object is open and its interior or opening is exposed.','A person opens an object.'],
 ['An object such as a door lid or container is open.','A hand moves the door lid or closure towards its closed position.','The object is closed and its opening is covered.','A person closes an object.'],
 ['An object is outside an opening or container.','A hand moves the object through the opening.','The object is inside the container or fitted into the opening.','A person inserts an object into a container or opening.'],
 ['A device light flame or flow is off or inactive.','A hand operates a switch knob or control to activate it.','The device light flame or flow is on or active.','A person turns on a device light flame or flow.'],
 ['A cutting tool approaches an intact object.','The tool presses or moves through the object to divide it.','A cut or separated piece is visible and the cutting stroke ends.','A person cuts or slices an object.'],
 ['A device light flame or flow is on or active.','A hand operates a switch knob or control to deactivate it.','The device light flame or flow is off or inactive.','A person turns off a device light flame or flow.'],
 ['A container holds material and is positioned near a receiving location.','The container tilts and its contents flow out.','The flow stops and the container is returned or moved away.','A person pours contents from a container.']]

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main(args):
 assert not torch.cuda.is_initialized()
 expected='5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
 assert sha(args.checkpoint)==expected,'checkpoint differs from official OpenAI CLIP B16 release'
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 bpe=out/'bpe_simple_vocab_16e6.txt.gz'
 if args.bpe:shutil.copyfile(args.bpe,bpe)
 else:urllib.request.urlretrieve('https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz',bpe)
 assert sha(bpe)=='924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'
 tok=SimpleTokenizer(str(bpe));texts=[s for row in DESCRIPTIONS for s in row]
 tokens=torch.zeros((len(texts),77),dtype=torch.long)
 for i,t in enumerate(texts):
  ids=[tok.encoder['<|startoftext|>']]+tok.encode(t)+[tok.encoder['<|endoftext|>']]
  assert len(ids)<=77;tokens[i,:len(ids)]=torch.tensor(ids)
 state=torch.jit.load(args.checkpoint,map_location='cpu').state_dict()
 model=build_model(state).float().eval();del state
 with torch.no_grad():emb=model.encode_text(tokens).float();emb=emb/emb.norm(dim=-1,keepdim=True)
 assert emb.shape==(40,512) and torch.isfinite(emb).all()
 torch.save(dict(embeddings=emb.reshape(10,4,512),tokens=tokens,labels=LABELS,descriptions=DESCRIPTIONS),out/'text.pt')
 provenance=dict(encoder='OpenAI CLIP ViT-B/16 frozen text encoder',checkpoint=args.checkpoint,checkpoint_sha256=expected,
  bpe_sha256=sha(bpe),text_sha256=sha(out/'text.pt'),labels=LABELS,descriptions=DESCRIPTIONS,
  phase_order=['start','middle','end','global'],description_author='implementation_assistant_from_class_names_only',
  target_video_access=False,target_GT_access=False,training=False,
  source_files={p.name:sha(p) for p in (CODE/'vendor/clip_text').glob('*.py')},
  upstream='https://github.com/openai/CLIP',paper_faithfulness='PDA phase-text mechanism transfer; not original GPT-4o prompts or paper reproduction')
 (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2));print(json.dumps(provenance),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--bpe');main(p.parse_args())
