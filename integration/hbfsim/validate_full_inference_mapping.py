"""Compile every distinct P32D2 operator shape and audit full-plan coverage.

No unsupported stage is substituted by the retired roofline/barrier path.
This is a coverage gate, not a full-inference completion receipt.
"""
import dataclasses
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
a=repo/'integration/hbfsim'
sys.path.insert(0,str(repo))
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
q=load('full_mapping_qwen',a/'qwen_hbfsim_cosim.py')
mat=load('full_mapping_matmul',a/'mapper_matmul.py')
soft=load('full_mapping_softmax',a/'mapper_softmax.py')
coupling=load('full_mapping_coupling',a/'mapper_event_coupling.py')
root=a/'results/full-inference-mapper-coverage-20261007-r1'
root.mkdir(exist_ok=False)
work=pathlib.Path(tempfile.mkdtemp(prefix='llmcompass-full-mapper-'))
shutil.copytree(repo/'systolic_array_model',work/'systolic_array_model',ignore=shutil.ignore_patterns('temp','*.gz','*.npy'))
(work/'systolic_array_model/temp').mkdir();os.chdir(work)

class CoverageCost(q.LLMCompassCostModel):
    def __init__(self,*args):
        super().__init__(*args);self.cache={};self.calls=[]

    def record(self,key,compile_one):
        if key not in self.cache:
            self.cache[key]=compile_one()
            print(json.dumps(dict(shape_key=key,**self.cache[key])),flush=True)
            (root/'distinct-shapes.json').write_text(json.dumps([dict(shape_key=k,**v) for k,v in self.cache.items()],indent=2)+'\n')
        self.calls.append((key,self.cache[key]))
        return self.cache[key]

    def matmul(self,m,k,n,bias=False,family='matmul'):
        def compile_one():
            op=mat.Matmul(self.dtype);op(self.Tensor([m,k],self.dtype),self.Tensor([k,n],self.dtype))
            latency=op.compile_and_simulate(self.device,'heuristic-GPU')
            if op.best_mapping is not None:
                cycles=op.simulate(op.computational_graph,op.best_mapping,self.device)
                assert abs(coupling.analytical_cycles(op.execution_stages)-cycles)<1e-6
                assert abs(latency-cycles/self.device.compute_module.clock_freq)<1e-12
                status='TILE_STAGE_READY'
                stages=op.execution_stages
            else:
                status='DECODE_SHORTCUT_STAGE_REQUIRED';stages=None
            return dict(status=status,official_seconds=latency,accounting=op.memory_accounting,events=op.main_memory_events,stages=stages,bias_requires_separate_stage=bias)
        self.record(('matmul',m,k,n,bias),compile_one)
        return super().matmul(m,k,n,bias,family)

    def batched_matmul(self,b,m,k,n,family):
        def compile_one():
            op=mat.BatchedMatmul(self.dtype)
            op(self.Tensor([b,m,k],self.dtype),self.Tensor([b,k,n],self.dtype))
            latency=op.compile_and_simulate(self.device,'heuristic-GPU')
            candidate=op.selected_accounting_candidate
            return dict(status='BATCH_INDEPENDENT_STAGE_REQUIRED' if candidate==0 else 'BATCH_MERGED_IO_DIRECTION_UNRESOLVED',official_seconds=latency,accounting=op.memory_accounting,selected_candidate=candidate)
        self.record(('batch',b,m,k,n),compile_one)
        return super().batched_matmul(b,m,k,n,family)

    def softmax(self,b,m,n):
        def compile_one():
            op=soft.Softmax(self.dtype);op(self.Tensor([b,m,n],self.dtype))
            latency=op.compile_and_simulate(self.device,'heuristic-GPU')
            assert op.simulate(op.computational_graph,op.best_mapping,self.device)==op.best_cycle_count
            return dict(status='SOFTMAX_STAGE_REQUIRED',official_seconds=latency,accounting=op.memory_accounting,events=op.main_memory_events,mapping=vars(op.best_mapping))
        self.record(('softmax',b,m,n),compile_one)
        return super().softmax(b,m,n)

    def vector(self,elements,ops_per_element,family,bytes_moved=0):
        def compile_one():
            cm=self.device.compute_module
            return dict(status='CUSTOM_VECTOR_STAGE_REQUIRED',compute_seconds=elements*ops_per_element/(cm.total_vector_flops_per_cycle*cm.clock_freq),logical_bytes=bytes_moved)
        self.record(('vector',elements,ops_per_element,family,bytes_moved),compile_one)
        return super().vector(elements,ops_per_element,family,bytes_moved)

cost=CoverageCost(repo,a/'RTX4000Ada_xmu_profile_v3.json')
model=dataclasses.replace(q.ModelSpec.from_json(a/'qwen25_1p5b.json'),prefill_tokens=32,decode_steps=2)
plan=q.remap_plan_for_gddr_abstract(q.build_plan(model,cost))
assert len(plan.operators)==len(cost.calls)==1521
counts={};ledger=[]
for op,(key,receipt) in zip(plan.operators,cost.calls):
    status=receipt['status'];counts[status]=counts.get(status,0)+1
    ledger.append(dict(index=op.index,phase=op.phase,layer=op.layer,name=op.name,shape_key=key,status=status))
(root/'operator-coverage.json').write_text(json.dumps(ledger,indent=2)+'\n')
summary=dict(status='COVERAGE_AUDIT_COMPLETE_NOT_FULL_SIMULATION',operator_count=len(ledger),distinct_shapes=len(cost.cache),coverage=counts,profile=str(a/'RTX4000Ada_xmu_profile_v3.json'),working_directory=str(work))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
