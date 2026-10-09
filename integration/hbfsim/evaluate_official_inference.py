"""Full Qwen matrix: official operator timing and direct boundary byte accounting.

No HBFSim time, cache filtering, fusion deletion, or fitted coefficient enters
the primary results. Qwen vector and projection-bias adaptations are labeled.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
import torch
import scalesim

repo=pathlib.Path(__file__).resolve().parents[2]
a=repo/'integration/hbfsim';sys.path.insert(0,str(repo))
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
q=load('official_eval_qwen',a/'qwen_hbfsim_cosim.py')
mat=load('official_eval_matmul',a/'mapper_matmul.py')
soft=load('official_eval_softmax',a/'mapper_softmax.py')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--architecture',type=pathlib.Path,default=a/'RTX4000Ada_xmu_profile_v3.json')
parser.add_argument('--output-root',type=pathlib.Path,required=True)
parser.add_argument('--hardware-root',type=pathlib.Path,required=True,
    help='Archived hardware collection root containing cases/<case>/summary.json')
args=parser.parse_args()
architecture=args.architecture.resolve()
hardware_root=args.hardware_root.resolve()
for p,d in [(32,2),(64,2),(128,2),(256,2),(512,2),(128,4),(128,8),(128,16)]:
    hardware_path=hardware_root/'cases'/f'p{p:03d}d{d:02d}'/'summary.json'
    if not hardware_path.is_file():
        parser.error('Missing hardware aggregate: '+str(hardware_path))
profile_hash=hashlib.sha256(architecture.read_bytes()).hexdigest()
root=args.output_root.resolve();root.mkdir(exist_ok=False)
(root/'source').mkdir()
for name in ['qwen_hbfsim_cosim.py','mapper_matmul.py','mapper_softmax.py','evaluate_official_inference.py']:
    shutil.copy2(a/name,root/'source'/name)
shutil.copy2(architecture,root/'source'/architecture.name)
work=pathlib.Path(tempfile.mkdtemp(prefix='llmcompass-official-eval-'))
shutil.copytree(repo/'systolic_array_model',work/'systolic_array_model',ignore=shutil.ignore_patterns('temp','*.gz','*.npy'))
(work/'systolic_array_model/temp').mkdir();os.chdir(work)
cache={}
# Only geometry-based ScaleSim lookup tables are reused. Operator latency and
# selected mappings must be recompiled after any profile change; the former
# unqualified seed contained v2 timings and is not safe for a different profile.
manifest=dict(status='RUNNING',profile=str(architecture),profile_sha256=profile_hash,
    source_sha256={name:hashlib.sha256((a/name).read_bytes()).hexdigest() for name in ['qwen_hbfsim_cosim.py','mapper_matmul.py','mapper_softmax.py','evaluate_official_inference.py']},
    timing_contract='Sum official compile_and_simulate latencies; bias and Qwen vector adaptations explicit; no HBFSim additive time.',
    traffic_contract='Direct mapper main/global/local boundary byte decisions; extra IO unclassified; no physical cache-counter accuracy claim.',
    seed_path=None,seed_sha256=None,operator_latency_seed_policy='disabled; compile each shape under current profile',
    geometry_lut_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'systolic_array_model').glob('look_up_table_*.csv')},
    scalesim_version='2.0.2',working_directory=str(work),
    environment={'python':sys.version,'torch':torch.__version__},
    hbfsim_in_primary_time=False,official_latency_parity_policy='Every unique shape is independently compiled with unchanged software_model classes')
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

class OfficialCost(q.LLMCompassCostModel):
    def __init__(self,*args):super().__init__(*args);self.calls=[]
    def compile(self,key):
        if key not in cache:
            if key[0]=='matmul':
                _,m,k,n=key;op=mat.Matmul(self.dtype)
                op(self.Tensor([m,k],self.dtype),self.Tensor([k,n],self.dtype))
            elif key[0]=='batch':
                _,b,m,k,n=key;op=mat.BatchedMatmul(self.dtype)
                op(self.Tensor([b,m,k],self.dtype),self.Tensor([b,k,n],self.dtype))
            else:
                _,b,m,n=key;op=soft.Softmax(self.dtype);op(self.Tensor([b,m,n],self.dtype))
            seconds=op.compile_and_simulate(self.device,'heuristic-GPU')
            original_class={'matmul':self.Matmul,
                'batch':self.BatchedMatmul,'softmax':self.Softmax}[key[0]]
            original=original_class(self.dtype)
            if key[0]=='matmul':original(self.Tensor([m,k],self.dtype),self.Tensor([k,n],self.dtype))
            elif key[0]=='batch':original(self.Tensor([b,m,k],self.dtype),self.Tensor([b,k,n],self.dtype))
            else:original(self.Tensor([b,m,n],self.dtype))
            original_seconds=original.compile_and_simulate(self.device,'heuristic-GPU')
            assert math.isclose(seconds,original_seconds,rel_tol=1e-12,abs_tol=1e-15),(key,seconds,original_seconds)
            if key[0]!='batch' and op.best_mapping is not None:
                cycles=op.simulate(op.computational_graph,op.best_mapping,self.device)
                assert abs(seconds-cycles/self.device.compute_module.clock_freq)<1e-12
            cache[key]=dict(seconds=seconds,official_source_seconds=original_seconds,official_latency_parity=True,
                accounting=op.memory_accounting,selected_candidate=getattr(op,'selected_accounting_candidate',None))
            print(json.dumps(dict(event='shape_compiled',shape=key,seconds=seconds)),flush=True)
        return cache[key]
    def matmul(self,m,k,n,bias=False,family='matmul'):
        r=self.compile(('matmul',m,k,n));cm=self.device.compute_module
        bias_s=m*n/(cm.total_vector_flops_per_cycle*cm.clock_freq) if bias else 0
        self.calls.append(dict(path='OFFICIAL_MATMUL',key=['matmul',m,k,n],**r,bias_seconds=bias_s,bias=bias))
        return q.Timing(family,2*m*k*n+(m*n if bias else 0),math.ceil((r['seconds']+bias_s)*1e9))
    def batched_matmul(self,b,m,k,n,family):
        r=self.compile(('batch',b,m,k,n));self.calls.append(dict(path='OFFICIAL_BATCHED_MATMUL',key=['batch',b,m,k,n],**r,bias_seconds=0))
        return q.Timing(family,2*b*m*k*n,math.ceil(r['seconds']*1e9))
    def softmax(self,b,m,n):
        r=self.compile(('softmax',b,m,n));self.calls.append(dict(path='OFFICIAL_SOFTMAX',key=['softmax',b,m,n],**r,bias_seconds=0))
        return q.Timing('softmax',b*m*n*5,math.ceil(r['seconds']*1e9))
    def vector(self,elements,ops_per_element,family,bytes_moved=0):
        t=super().vector(elements,ops_per_element,family,bytes_moved)
        self.calls.append(dict(path='QWEN_VECTOR_ADAPTER',seconds=t.compute_ns/1e9,bias_seconds=0,accounting=None))
        return t

base=q.ModelSpec.from_json(a/'qwen25_1p5b.json');rows=[]
for p,d in [(32,2),(64,2),(128,2),(256,2),(512,2),(128,4),(128,8),(128,16)]:
    model=dataclasses.replace(base,prefill_tokens=p,decode_steps=d)
    cost=OfficialCost(repo,architecture);plan=q.build_plan(model,cost)
    assert len(plan.operators)==len(cost.calls)==507*(d+1)
    ledger=[];phases={};counts={};read=write=unclassified=0;ns=0
    for op,call in zip(plan.operators,cost.calls):
        counts[call['path']]=counts.get(call['path'],0)+1
        if call['accounting'] is None:
            r=sum(x.bytes for x in op.reads);w=sum(x.bytes for x in op.writes);u=0
        else:
            acc=call['accounting'];r=acc['main_read_bytes'];w=acc['main_write_bytes'];u=acc.get('unclassified_extra_io_bytes',0)
            if call.get('bias'):r+=sum(x.bytes for x in op.reads if x.label.endswith('.bias'))
        seconds=call['seconds']+call.get('bias_seconds',0)
        duration=op.timing.compute_ns
        assert duration==math.ceil(seconds*1e9)
        row=dict(index=op.index,phase=op.phase,layer=op.layer,name=op.name,path=call['path'],
            model_ns=duration,known_read_bytes=r,known_write_bytes=w,unclassified_io_bytes=u,
            mapper_accounting=call['accounting'],bias_adapter_seconds=call.get('bias_seconds',0),selected_candidate=call.get('selected_candidate'))
        ledger.append(row);read+=r;write+=w;unclassified+=u;ns+=duration
        phase=phases.setdefault(op.phase,dict(model_ns=0,known_read_bytes=0,known_write_bytes=0,unclassified_io_bytes=0,operators=0))
        for key in ['model_ns','known_read_bytes','known_write_bytes','unclassified_io_bytes']:phase[key]+=row[key]
        phase['operators']+=1
    case=f'p{p:03d}d{d:02d}';out=root/case;out.mkdir()
    hardware=hardware_root/'cases'/case/'summary.json'
    hw=json.loads(hardware.read_text())['rois']['full'];hr=hw['dram_read_bytes']['median'];hwbytes=hw['dram_write_bytes']['median'];ht=hw['natural_cuda_event_ms']['median']
    summary=dict(case=case,status='PASS_FULL_OPERATOR_ACCOUNTING_NOT_HARDWARE_ACCEPTANCE',operator_count=len(ledger),paths=counts,
        model_ms=ns/1e6,known_read_bytes=read,known_write_bytes=write,unclassified_io_bytes=unclassified,
        total_boundary_bytes=read+write+unclassified,model_GBps=(read+write+unclassified)/ns,
        hardware_ms=ht,hardware_read_bytes=hr,hardware_write_bytes=hwbytes,hardware_GBps=(hr+hwbytes)/(ht*1e6),
        time_error_pct=100*((ns/1e6)/ht-1),bandwidth_error_pct=100*((read+write+unclassified)/ns/((hr+hwbytes)/(ht*1e6))-1),
        known_read_error_pct=100*(read/hr-1),known_write_error_pct=100*(write/hwbytes-1),
        read_error_range_pct=[100*(read/hr-1),100*((read+unclassified)/hr-1)],write_error_range_pct=[100*(write/hwbytes-1),100*((write+unclassified)/hwbytes-1)],
        phase_totals=phases,hardware_source=str(hardware),hardware_sha256=hashlib.sha256(hardware.read_bytes()).hexdigest())
    assert sum(x['model_ns'] for x in phases.values())==ns
    assert sum(x['known_read_bytes']+x['known_write_bytes']+x['unclassified_io_bytes'] for x in phases.values())==summary['total_boundary_bytes']
    (out/'operators.json').write_text(json.dumps(ledger,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows.append(summary);(root/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    (root/'shape-cache.json').write_text(json.dumps([dict(key=k,**v) for k,v in cache.items()],indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['phase_totals','paths']}),flush=True)
manifest.update(status='PASS_FULL_MATRIX_ACCOUNTING',completed_cases=len(rows))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PASS_FULL_MATRIX_ACCOUNTING',len(rows),flush=True)
