"""Parity checks for official batch selection and softmax main-boundary accounting."""
import importlib.util
import json
import pathlib
import sys
import os
import shutil
import tempfile
import torch
import scalesim

repo=pathlib.Path(__file__).resolve().parents[2]
adapter=repo/'integration/hbfsim'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
q=load('qwen_extended',adapter/'qwen_hbfsim_cosim.py')
cost=q.LLMCompassCostModel(repo,adapter/'RTX4000Ada_xmu_profile_v3.json')
assert cost.device.compute_module.total_systolic_array_flops == 106905600000000
work=pathlib.Path(tempfile.mkdtemp(prefix='llmcompass-extended-parity-'))
shutil.copytree(repo/'systolic_array_model',work/'systolic_array_model',ignore=shutil.ignore_patterns('temp','*.gz','*.npy'))
(work/'systolic_array_model/temp').mkdir();os.chdir(work)
batch=load('batch_accounting',adapter/'mapper_matmul.py')
softmax=load('softmax_accounting',adapter/'mapper_softmax.py')
coupling=load('extended_coupling',adapter/'mapper_event_coupling.py')
from software_model.matmul import BatchedMatmul
from software_model.softmax import Softmax
for original,patched,shape in [(BatchedMatmul,batch.BatchedMatmul,(12,1,128,32)),(BatchedMatmul,batch.BatchedMatmul,(12,32,128,32)),(BatchedMatmul,batch.BatchedMatmul,(12,32,32,128)),(Softmax,softmax.Softmax,(12,32,32))]:
    a,b=original(cost.dtype),patched(cost.dtype)
    for op in [a,b]:
        if len(shape)==4:
            bs,m,k,n=shape
            op(cost.Tensor([bs,m,k],cost.dtype),cost.Tensor([bs,k,n],cost.dtype))
        else:
            op(cost.Tensor(list(shape),cost.dtype))
        op.compile_and_simulate(cost.device,'heuristic-GPU')
    assert a.latency==b.latency
    if len(shape)==3:
        assert vars(a.best_mapping)==vars(b.best_mapping)
        assert b.simulate(b.computational_graph,b.best_mapping,cost.device)==b.best_cycle_count
        assert abs(coupling.analytical_cycles(b.execution_stages)-b.best_cycle_count)<1e-8
        for direction in ['read','write']:
            count=sum(e['rows']*e['columns']*cost.dtype.word_size for e in b.main_memory_events if e['direction']==direction)
            assert count==b.memory_accounting[f'main_{direction}_bytes']
    print(json.dumps(dict(operator=original.__name__,shape=shape,latency_seconds=b.latency,accounting=b.memory_accounting,selected_candidate=getattr(b,'selected_accounting_candidate',None))),flush=True)
print('PASS_EXTENDED_PARITY',flush=True)
