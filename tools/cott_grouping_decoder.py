"""只约束分组的序列化合法性；动作、边界和描述仍由同一模型选择。"""
import json


def to_intervals(obj, grid, labels):
    """整段相邻transition组转换为真实PTS；不裁剪、合并或删除任何生成行。"""
    assert len(grid)>1 and all(a<b for a,b in zip(grid,grid[1:]))
    rows=obj.get('intervals');assert isinstance(rows,list)
    end=0;result=[]
    for r in rows:
        a,b=r.get('start_pair'),r.get('end_pair')
        assert type(a) is int and type(b) is int and end<=a<b<len(grid)
        assert r.get('label') in labels and isinstance(r.get('description'),str) and r['description'].strip()
        result.append(dict(start=grid[a],end=grid[b],label=r['label'],description=r['description']))
        end=b
    return dict(intervals=result)


def safe_description_tokens(tokenizer):
    # ASCII英文描述；结构引号/转义/特殊token只能从固定JSON模板产生。
    special=set(tokenizer.all_special_ids);out=[]
    for i in range(len(tokenizer)):
        if i in special:continue
        s=tokenizer.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if s and all(32<=ord(c)<127 and c not in '\\"' for c in s):out.append(i)
    assert out
    return out


class GroupingDecoder:
    """单beam贪心解码的token-trie状态机，合法动作列表由模型一次性生成。"""
    def __init__(self,tokenizer,prompt_tokens,grid,labels,eos,safe_tokens=None):
        self.tok=tokenizer;self.prompt_tokens=prompt_tokens;self.grid=list(grid);self.n=len(grid)-1
        assert self.n>0 and all(a<b for a,b in zip(grid,grid[1:]))
        self.labels=list(labels);self.eos=[eos] if isinstance(eos,int) else list(eos)
        self.safe=list(safe_tokens if safe_tokens is not None else safe_description_tokens(tokenizer))
        self.safe_set=set(self.safe);self.used=[];self.previous_end=0;self.start=None
        self.description=[];self.has_text=False;self.done=False;self.stage='template';self.rows=0
        self.close=self.encode('"}')
        assert self.close[0] not in self.safe_set
        self.options([('{"intervals":[',('next',True))])

    def encode(self,s):
        ids=self.tok.encode(s,add_special_tokens=False)
        assert ids and self.tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==s
        return ids

    def options(self,choices):
        root={}
        for s,action in choices:
            node=root
            for token in self.encode(s):
                assert 'finish' not in node
                node=node.setdefault(token,{})
            assert not node
            node['finish']=action
        self.node=root;self.stage='template'

    def advance(self,action):
        kind,value=action
        if kind=='next':
            choices=[(']}',('done',None))]
            if self.previous_end<self.n:
                choices.append((('' if value else ',')+'{"start_pair":',('start',None)))
            self.options(choices)
        elif kind=='start':
            self.options([(str(i)+',"end_pair":',('end',i)) for i in range(self.previous_end,self.n)])
        elif kind=='end':
            self.start=value
            self.options([(str(i)+',"label":"',('label',i)) for i in range(self.start+1,self.n+1)])
        elif kind=='label':
            self.previous_end=value
            self.options([(label+'","description":"',('description',None)) for label in self.labels])
        elif kind=='description':
            self.description=[];self.has_text=False;self.stage='description'
        elif kind=='description_end':
            self.rows+=1;self.advance(('next',False))
        elif kind=='done':self.done=True;self.stage='done'
        else:raise AssertionError(kind)

    def allowed(self):
        if self.done:return self.eos
        if self.stage=='description':return self.safe+([self.close[0]] if self.has_text else [])
        return list(self.node)

    def feed(self,token):
        assert token in self.allowed(),('illegal structured token',token,self.stage)
        if self.done:return
        if self.stage=='description':
            if token in self.safe_set:
                self.description.append(token)
                self.has_text=self.has_text or bool(self.tok.decode([token],clean_up_tokenization_spaces=False).strip())
                return
            self.options([('"}',('description_end',None))])
        self.node=self.node[token]
        if 'finish' in self.node:self.advance(self.node['finish'])

    def __call__(self,batch_id,input_ids):
        assert batch_id==0
        ids=input_ids.tolist()[self.prompt_tokens:]
        assert ids[:len(self.used)]==self.used,'non-prefix/beam decoding is unsupported'
        for token in ids[len(self.used):]:self.feed(token)
        self.used=ids
        return self.allowed()

    def finish(self,ids):
        ids=list(ids);assert ids[:len(self.used)]==self.used
        for token in ids[len(self.used):]:self.feed(token)
        self.used=ids;assert self.done,'incomplete grouping output cannot become an empty prediction'
        raw=self.tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        parsed=json.loads(raw);intervals=to_intervals(parsed,self.grid,self.labels)
        assert len(intervals['intervals'])==self.rows
        return intervals,dict(format='ordered_transition_spans_v1',token_ids=ids,grid=self.grid,
            generated_index_intervals=parsed,all_tokens_legal=True,complete=True,
            manual_interval_edits=0,deleted_intervals=0,model_chose_boundaries=True)
