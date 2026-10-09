"""Compare unchanged official mapper cycles with additive byte instrumentation."""
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import torch
import scalesim

repo=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo))
work=pathlib.Path(tempfile.mkdtemp(prefix='llmcompass-mapper-accounting-'))
shutil.copytree(repo/'systolic_array_model',work/'systolic_array_model',ignore=shutil.ignore_patterns('temp','*.gz','*.npy'))
(work/'systolic_array_model/temp').mkdir(exist_ok=True)
os.chdir(work)
spec=importlib.util.spec_from_file_location('qwen_mapper_validation',repo/'integration/hbfsim/qwen_hbfsim_cosim.py')
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
cost=q.LLMCompassCostModel(repo,repo/'integration/hbfsim/RTX4000Ada_xmu_profile_v3.json')
assert cost.device.compute_module.total_systolic_array_flops == 106905600000000
from software_model.matmul import Matmul as Original
patched_spec=importlib.util.spec_from_file_location('instrumented_matmul',repo/'integration/hbfsim/mapper_matmul.py')
patched=importlib.util.module_from_spec(patched_spec);patched_spec.loader.exec_module(patched)
for m,k,n in [(1,1536,1536),(32,1536,1536),(64,1536,256),(33,129,65)]:
    before=Original(cost.dtype);after=patched.Matmul(cost.dtype)
    for op in [before,after]:
        op(cost.Tensor([m,k],cost.dtype),cost.Tensor([k,n],cost.dtype))
        op.compile_and_simulate(cost.device,'heuristic-GPU')
    assert before.latency==after.latency
    if after.best_mapping is not None:
        assert before.best_cycle_count==after.best_cycle_count
        assert vars(before.best_mapping)==vars(after.best_mapping)
        cycles=after.simulate(after.computational_graph,after.best_mapping,cost.device)
        assert cycles==after.best_cycle_count
    else:
        cycles=None
    for direction in ['read','write']:
        actual=sum(e['rows']*e['columns']*cost.dtype.word_size for e in after.main_memory_events if e['direction']==direction)
        assert actual==after.memory_accounting[f'main_{direction}_bytes']
    for event in after.main_memory_events:
        rows,cols={'A':(m,k),'B':(k,n),'C':(m,n)}[event['tensor']]
        assert 0<=event['row']<event['row']+event['rows']<=rows
        assert 0<=event['column']<event['column']+event['columns']<=cols
    print(json.dumps(dict(shape=[m,k,n],cycle_count=cycles,latency_seconds=after.latency,mapping=None if after.best_mapping is None else vars(after.best_mapping),accounting=after.memory_accounting,event_count=len(after.main_memory_events))),flush=True)
print('PASS_MAPPER_CYCLE_AND_MAPPING_PARITY',str(work),flush=True)
