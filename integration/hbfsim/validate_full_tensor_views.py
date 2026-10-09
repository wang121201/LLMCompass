"""Audit analytical tensor views across the complete P32D2 plan, without replay."""
import dataclasses
import importlib.util
import pathlib
import sys
repo=pathlib.Path(__file__).resolve().parents[2]
a=repo/'integration/hbfsim'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
q=load('view_qwen',a/'qwen_hbfsim_cosim.py')
v=load('tensor_views',a/'mapper_tensor_addresses.py')
cost=q.LLMCompassCostModel(repo,a/'RTX4000Ada_xmu_profile_v3.json')
model=dataclasses.replace(q.ModelSpec.from_json(a/'qwen25_1p5b.json'),prefill_tokens=32,decode_steps=2)
plan=q.remap_plan_for_gddr_abstract(q.build_plan(model,cost))
view_count=0;operators=set()
def check(view):
    global view_count
    # Bounding-box corners verify a positive-stride view without enumerating
    # whole model weight matrices. Strided enumeration is tested separately.
    for row,column in [(0,0),(view.rows-1,view.columns-1)]:
        assert sum(n for _,_,n in view.requests(dict(row=row,rows=1,column=column,columns=1)))==2
    view_count+=1
def normal(access,rows,columns,row_stride=None,column_stride=2,offset=0):
    return v.MatrixView(access.address+offset,rows,columns,columns*2 if row_stride is None else row_stride,column_stride,access.address,access.bytes)
for op in plan.operators:
    step=0 if op.phase=='prefill' else int(op.phase.split('_')[1])
    tokens=model.prefill_tokens if step==0 else 1;context=model.prefill_tokens+step
    if op.name in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head']:
        m=1 if op.name=='lm_head' else tokens
        k=model.intermediate_size if op.name=='down_proj' else model.hidden_size
        n=(model.kv_hidden_size if op.name in ['k_proj','v_proj'] else model.intermediate_size if op.name in ['gate_proj','up_proj'] else model.vocab_size if op.name=='lm_head' else model.hidden_size)
        check(normal(op.reads[0],m,k))
        # PyTorch linear weights are [N,K], viewed as transposed [K,N].
        check(normal(op.reads[1],k,n,row_stride=2,column_stride=k*2))
        check(normal(op.writes[0],m,n));operators.add(op.index)
    elif op.name in ['attention_score','attention_value']:
        for head in range(model.attention_heads):
            if op.name=='attention_score':
                check(normal(op.reads[0],tokens,model.head_dim,row_stride=model.hidden_size*2,offset=head*model.head_dim*2))
                check(v.grouped_kv_view(op.reads[1].address,op.reads[1].bytes,context,model.head_dim,model.kv_heads,model.attention_heads,head,True))
                check(normal(op.writes[0],tokens,context,offset=head*tokens*context*2))
            else:
                check(normal(op.reads[0],tokens,context,offset=head*tokens*context*2))
                check(v.grouped_kv_view(op.reads[1].address,op.reads[1].bytes,context,model.head_dim,model.kv_heads,model.attention_heads,head,False))
                check(normal(op.writes[0],tokens,model.head_dim,row_stride=model.hidden_size*2,offset=head*model.head_dim*2))
        operators.add(op.index)
    elif op.name=='attention_softmax':
        check(normal(op.reads[0],model.attention_heads*tokens,context))
        check(normal(op.writes[0],model.attention_heads*tokens,context));operators.add(op.index)
    else:
        for access in op.reads+op.writes:
            assert 0<=access.address<access.address+access.bytes<=plan.hbm_arena_bytes
        operators.add(op.index)
assert len(operators)==len(plan.operators)==1521
print('PASS_FULL_PLAN_TENSOR_BOUNDS',len(operators),'operators',view_count,'matrix_views; NOT_FULL_SIMULATION')
