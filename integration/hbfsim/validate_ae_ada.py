"""Official operator scale validation and independent Ada diagnostics.

The official minimal-input overhead functions measure synchronized wall time.
They are not pure GPU launch latencies. Qwen/SGLang timing is not substituted
for this independent calibration or for the Figure 5 hardware observations.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import statistics
import sys
import tempfile
import math
import time
import csv
import subprocess
import shlex
import signal
import dataclasses
import ast
import ctypes
import torch
import scalesim
import pynvml

REPO=pathlib.Path(__file__).resolve().parents[2]
A=REPO/'integration/hbfsim'
sys.path.insert(0,str(REPO))
from software_model.matmul import Matmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.gelu import gelu_gpu
from software_model.utils import Tensor,data_type_dict

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def clock_boundary():
    """Boundary telemetry, not a continuous clock measurement or correction."""
    pynvml.nvmlInit()
    handle=pynvml.nvmlDeviceGetHandleByIndex(0)
    return {'sm_MHz':pynvml.nvmlDeviceGetClockInfo(handle,pynvml.NVML_CLOCK_SM),
            'memory_MHz':pynvml.nvmlDeviceGetClockInfo(handle,pynvml.NVML_CLOCK_MEM)}

def small_operator_examples():
    examples=[]
    for exponent in range(5,16):
        examples.append(('softmax',[4096,2**exponent],'AE_FIXED_M'))
    for exponent in range(5,16):
        examples.append(('softmax',[2**exponent,4096],'AE_FIXED_N'))
    for exponent in range(10,30):
        examples.append(('gelu',[2**exponent],'AE_ELEMENTS'))
    # Use recorded official calls, rather than guessed workload dimensions.
    shape_path=A/'results/official-inference-matrix-20261007-r2/shape-cache.json'
    for entry in json.loads(shape_path.read_text()):
        key=entry['key']
        if key[0]=='softmax':examples.append(('softmax',key[1:],'QWEN_UNFUSED_DIAGNOSTIC'))
    return examples

def native_function(name,eager=False):
    if name=='softmax':return lambda x:torch.softmax(x,dim=-1)
    if eager:return lambda x:torch.nn.functional.gelu(x,approximate='tanh')
    return gelu_gpu

def dual_clock_diagnostic(name,shape,dtype,eager=False,scrub=False):
    fn=native_function(name,eager)
    x=torch.randn(shape,dtype=dtype,device='cuda')
    trash=torch.empty(128*1024**2,dtype=torch.uint8,device='cuda') if scrub else None
    for _ in range(10):out=fn(x)
    torch.cuda.synchronize()
    wall=[];events=[];rounds=[]
    for _ in range(3):
        ws=[];es=[]
        for _ in range(50):
            if trash is not None:trash.fill_(7);torch.cuda.synchronize()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();before=time.perf_counter_ns()
            out=fn(x)
            end.record();torch.cuda.synchronize()
            ws.append((time.perf_counter_ns()-before)/1000);es.append(start.elapsed_time(end)*1000)
        wall.extend(ws);events.extend(es)
        rounds.append({'wall_us':statistics.median(ws),'event_us':statistics.median(es)})
    return dict(operator=name,shape=shape,dtype=str(dtype),eager=eager,
        cache_condition='128 MiB scrub intervention' if scrub else 'same-input warm repetition',
        wall_us=statistics.median(wall),event_us=statistics.median(events),rounds=rounds,
        boundary_clock=clock_boundary(),scope='Diagnostic event-bracketed wall time; event instrumentation differs from official primary timer; not pure kernel time')

def counter_worker(task):
    """One original GPU function, primed explicitly; no output trace or report."""
    torch.manual_seed(17)
    dtype=getattr(torch,task.get('dtype','float16'))
    x=torch.randn(task['shape'],dtype=dtype,device='cuda')
    fn=native_function(task['operator'],task.get('eager',False))
    trash=torch.empty(128*1024**2,dtype=torch.uint8,device='cuda')
    for _ in range(20):out=fn(x)
    torch.cuda.synchronize()
    if task.get('scrub'):trash.fill_(7);torch.cuda.synchronize()
    torch.cuda.nvtx.range_push('OperatorCounter')
    out=fn(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    assert out.shape==x.shape
    print('COUNTER_BOUNDARY',json.dumps(clock_boundary()),flush=True)

def interleaved_gelu_controls():
    """Rank/dtype controls with wall-only timing, not a new overhead formula."""
    specifications=[([1,1],torch.float32,False),([1,1],torch.float16,False),
                    ([1],torch.float32,False),([1],torch.float16,False),
                    ([65536],torch.float16,False),([65536],torch.float16,True)]
    cases=[]
    for shape,dtype,eager in specifications:
        x=torch.randn(shape,dtype=dtype,device='cuda');fn=native_function('gelu',eager)
        for _ in range(20):out=fn(x)
        torch.cuda.synchronize()
        cases.append((x,fn,dict(shape=shape,dtype=str(dtype),eager=eager,rounds_us=[])))
    for repeat in range(5):
        samples=[[] for _ in cases]
        # Rotate ordering to keep systematic drift from favoring the size-one control.
        for step in range(50):
            for offset in range(len(cases)):
                index=(step+offset+repeat)%len(cases);x,fn,_=cases[index]
                before=time.perf_counter_ns();out=fn(x);torch.cuda.synchronize()
                samples[index].append((time.perf_counter_ns()-before)/1000)
        for index,(_,_,row) in enumerate(cases):row['rounds_us'].append(statistics.median(samples[index]))
    rows=[]
    for _,_,row in cases:
        row['median_us']=statistics.median(row['rounds_us']);rows.append(row)
    return dict(rows=rows,scope='Wall-only interleaved function+sync, five rounds of fifty repeats; diagnostics only; no coefficient fitted',boundary_clock=clock_boundary())

def tensor_throughput_diagnostic():
    """Independent BF16 GEMM checks; measured rates never overwrite a profile."""
    torch.manual_seed(17)
    rows=[]
    original=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        for n in (2048,4096,8192):
            left=torch.randn((n,n),dtype=torch.bfloat16,device='cuda')
            right=torch.randn((n,n),dtype=torch.bfloat16,device='cuda')
            output=torch.empty((n,n),dtype=torch.bfloat16,device='cuda')
            for reduced in (False,True):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
                for _ in range(20):torch.mm(left,right,out=output)
                torch.cuda.synchronize()
                samples=[]
                for _ in range(5):
                    before=clock_boundary()
                    begin=torch.cuda.Event(enable_timing=True)
                    end=torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(20):torch.mm(left,right,out=output)
                    end.record();torch.cuda.synchronize()
                    after=clock_boundary()
                    samples.append(dict(per_call_ms=begin.elapsed_time(end)/20,
                        clock_before=before,clock_after=after))
                ms=statistics.median(x['per_call_ms'] for x in samples)
                flops=2*n*n*n
                clock=max(x[b]['sm_MHz'] for x in samples for b in ('clock_before','clock_after'))
                rows.append(dict(shape=[n,n,n],dtype='bfloat16',
                    allow_bf16_reduced_precision_reduction=reduced,flops=flops,
                    median_ms=ms,observed_tflops=flops/(ms*1e9),samples=samples,
                    configured_tflops_2175MHz_512_FLOPs_per_SM_cycle=53.4528,
                    endpoint_512_FLOPs_per_SM_cycle_tflops=48*512*clock*1e6/1e12,
                    scope='Warm square cuBLAS GEMM, twenty queued calls per event bracket; no kernel counter or numerical accumulation audit; endpoint clocks are not continuous telemetry'))
                print('TENSOR_DIAGNOSTIC',json.dumps(rows[-1]),flush=True)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original
    return dict(rows=rows,original_allow_bf16_reduced_precision_reduction=original,
        profile_changed=False,scope='Independent non-Qwen throughput consistency check; not a fitted peak, full-inference result, or native instruction proof')

def explicit_fp32_gemm_diagnostic():
    """Use the documented cuBLAS FP32 compute contract, not inferred torch flags."""
    lib_path=pathlib.Path('/usr/local/cuda-12.8/lib64/libcublas.so.12')
    lib=ctypes.CDLL(str(lib_path));ptr=ctypes.c_void_p;integer=ctypes.c_int
    for name in ('cublasCreate_v2','cublasDestroy_v2','cublasSetStream_v2','cublasGemmEx'):
        getattr(lib,name).restype=integer
    lib.cublasCreate_v2.argtypes=[ctypes.POINTER(ptr)]
    lib.cublasDestroy_v2.argtypes=[ptr];lib.cublasSetStream_v2.argtypes=[ptr,ptr]
    lib.cublasGemmEx.argtypes=[ptr,integer,integer,integer,integer,integer,ptr,
        ptr,integer,integer,ptr,integer,integer,ptr,ptr,integer,integer,integer,integer]
    def check(status):assert status==0,('cuBLAS status',status)
    handle=ptr();check(lib.cublasCreate_v2(ctypes.byref(handle)))
    check(lib.cublasSetStream_v2(handle,ptr(torch.cuda.current_stream().cuda_stream)))
    alpha=ctypes.c_float(1);beta=ctypes.c_float(0);torch.manual_seed(17);rows=[]
    try:
        for n in (256,2048,4096):
            left=torch.randn((n,n),dtype=torch.bfloat16,device='cuda')
            right=torch.randn((n,n),dtype=torch.bfloat16,device='cuda')
            output=torch.empty((n,n),dtype=torch.float32,device='cuda')
            def call():
                # Row-major left@right equals column-major right@left.
                # Installed CUDA headers: CUDA_R_16BF=14, CUDA_R_32F=0,
                # CUBLAS_COMPUTE_32F=68, CUBLAS_GEMM_DEFAULT_TENSOR_OP=99.
                check(lib.cublasGemmEx(handle,0,0,n,n,n,ctypes.byref(alpha),
                    ptr(right.data_ptr()),14,n,ptr(left.data_ptr()),14,n,
                    ctypes.byref(beta),ptr(output.data_ptr()),0,n,68,99))
            for _ in range(20):call()
            torch.cuda.synchronize();samples=[]
            for _ in range(5):
                before=clock_boundary();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):call()
                end.record();torch.cuda.synchronize()
                samples.append(dict(per_call_ms=start.elapsed_time(end)/20,clock_before=before,clock_after=clock_boundary()))
            ms=statistics.median(x['per_call_ms'] for x in samples)
            # This small independent check establishes layout and output precision,
            # not numerical equivalence of full Qwen or every tensor instruction.
            error=None
            if n==256:
                reference=left.float()@right.float()
                error=(output-reference).abs().max().item()
                assert torch.allclose(output,reference,rtol=1e-3,atol=1e-3)
            rows.append(dict(N=n,flops=2*n**3,median_ms=ms,observed_tflops=2*n**3/(ms*1e9),
                input_dtype='BF16',output_dtype='FP32',compute_contract='CUBLAS_COMPUTE_32F',
                max_abs_error_small_reference=error,samples=samples))
            print('EXPLICIT_FP32_GEMM',json.dumps(rows[-1]),flush=True)
    finally:check(lib.cublasDestroy_v2(handle))
    return dict(rows=rows,library_path=str(lib_path.resolve()),library_sha256=sha(lib_path),
        gpu_uuid='GPU-18ace299-5348-e6e4-d48c-1ee5a602859b',profile_changed=False,
        scope='Independent dense cuBLAS BF16 inputs with explicit FP32 compute/output; no workload-fit or model change')

def collect_counters(output):
    metrics=['dram__bytes_read.sum','dram__bytes_write.sum','gpu__time_duration.sum',
             'lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum']
    # Includes the problematic points, a beyond-L2 scale, and wrapper/eager controls.
    targets=[('softmax',[4096,1024],False),('softmax',[4096,32768],False),
             ('gelu',[65536],False),('gelu',[65536],True),
             ('gelu',[2**22],False),('gelu',[2**26],False)]
    results=[]
    prior=A/'results/official-operator-counters-20261007-r3'
    if (prior/'counter-comparison.json').exists():
        identities=json.loads((prior/'source-identities.json').read_text())
        assert all(sha(REPO/name)==digest for name,digest in identities.items())
        def worker_ast(path):
            tree=ast.parse(pathlib.Path(path).read_text())
            return {node.name:ast.dump(node,include_attributes=False) for node in tree.body
                if isinstance(node,ast.FunctionDef) and node.name in ('counter_worker','native_function','clock_boundary')}
        assert worker_ast(prior/'validate_ae_ada.py')==worker_ast(__file__)
        results=json.loads((prior/'counter-comparison.json').read_text())
        for row in results:row['origin_results_root']=str(prior)
        (output/'reuse-manifest.json').write_text(json.dumps(dict(
            source_root=str(prior),counter_sha256=sha(prior/'counter-comparison.json'),
            driver_sha256=sha(prior/'validate_ae_ada.py'),reused_invocations=len(results),
            worker_ast_equal=True,official_source_hashes_equal=True),indent=2)+'\n')
    for name,shape,eager in targets:
        for scrub in [False,True]:
            for repeat in range(2):
                task=dict(operator=name,shape=shape,eager=eager,scrub=scrub,repeat=repeat)
                if any(row['task']==task for row in results):continue
                cmd=['/usr/local/cuda-12.8/bin/ncu','--config-file','off','--csv','--page','raw',
                     '--replay-mode','application','--cache-control','none','--clock-control','none',
                     '--nvtx','--nvtx-include','OperatorCounter/','--metrics',','.join(metrics),
                     '/home/xmu/sgl/bin/python','-B',str(pathlib.Path(__file__).resolve()),
                     '--counter-task',json.dumps(task)]
                started=time.monotonic()
                def invoke(command):
                    p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                        text=True,start_new_session=True)
                    try:stdout,stderr=p.communicate(timeout=180)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid,signal.SIGTERM);stdout,stderr=p.communicate(timeout=15)
                        raise RuntimeError('Counter child timed out; process group terminated')
                    return p.returncode,stdout,stderr
                # The exact stopped eager target passed the root preflight.
                # Use that same authorized privilege path for every remaining
                # target; do not rely on profiler warning-string detection.
                command=shlex.join(['env',
                        'CUDA_VISIBLE_DEVICES=0','CUDA_HOME=/usr/local/cuda-12.8',
                        'PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin',
                        'PYTHONPATH=/tmp/llmcompass-scalesim202-20261007:/tmp/llmcompass-ae-deps-20261006',
                        'PYTHONNOUSERSITE=1',*cmd])
                # On this host root_sudo is a sudo credential variable, not
                # a shell command. Send it only on sudo's stdin; never print it.
                shell='set +x; if type root_sudo >/dev/null 2>&1; then root_sudo '+command+'; else printf \'%s\\n\' "${root_sudo:?}" | sudo -S -p \'\' '+command+'; fi'
                rc,stdout,stderr=invoke(['/bin/bash','-ic',shell]);used_root=True
                parsed=[]
                csv_rows=list(csv.reader(io.StringIO('\n'.join(line for line in stdout.splitlines() if line.startswith('"')))))
                assert csv_rows,('No counter CSV header',rc,stdout[-1000:],stderr[-1000:])
                header=csv_rows[0]
                if 'Metric Name' in header:
                    for values in csv_rows[1:]:
                        row=dict(zip(header,values))
                        if row.get('Metric Name') in metrics:parsed.append(row)
                else:
                    # Installed NCU raw-page CSV is wide: a units row followed
                    # by kernel rows. Keep only the explicitly requested metrics.
                    assert set(metrics).issubset(set(header)),header[:15]
                    units=dict(zip(header,csv_rows[1]))
                    for values in csv_rows[2:]:
                        row=dict(zip(header,values))
                        if not row.get('ID','').isdigit():continue
                        for metric in metrics:
                            parsed.append({'ID':row['ID'],'Kernel Name':row['Kernel Name'],
                                'Metric Name':metric,'Metric Unit':units[metric],'Metric Value':row[metric]})
                seen={row['Metric Name'] for row in parsed}
                assert rc==0 and seen==set(metrics),(rc,seen,stdout[-1000:],stderr[-1000:])
                result=dict(task=task,metrics=parsed,used_root_helper=used_root,
                    origin_results_root=str(output),
                    command=cmd,elapsed_host_seconds=time.monotonic()-started,
                    clock_boundaries=[json.loads(line.split('COUNTER_BOUNDARY ',1)[1]) for line in stdout.splitlines() if line.startswith('COUNTER_BOUNDARY ')],
                    scope='Single selected NVTX invocation; application replay; no cache/clock control; no ncu-rep or trace export')
                results.append(result)
                (output/'counter-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
                print('COUNTER_DONE',json.dumps(task),'kernels',len({v['ID'] for v in parsed}),flush=True)
    expected={(name,tuple(shape),eager,scrub,repeat) for name,shape,eager in targets for scrub in (False,True) for repeat in range(2)}
    observed={(row['task']['operator'],tuple(row['task']['shape']),row['task']['eager'],row['task']['scrub'],row['task']['repeat']) for row in results}
    assert observed==expected and len(results)==len(expected)==24
    return results

def full_inference_breakdown(output):
    """Join source FLOP definitions to frozen full ledgers; do not alter times."""
    spec=importlib.util.spec_from_file_location('full_breakdown_qwen',A/'qwen_hbfsim_cosim.py')
    q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
    root=A/'results/official-inference-matrix-20261007-r2'
    manifest=json.loads((root/'manifest.json').read_text())
    profile=root/'source'/pathlib.Path(manifest['profile']).name
    assert sha(profile)==manifest['profile_sha256']
    assert sha(A/'qwen_hbfsim_cosim.py')==manifest['source_sha256']['qwen_hbfsim_cosim.py']
    geometry=json.loads((A/'qwen25_1p5b.json').read_text())['architecture']
    assert [geometry[k] for k in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size')]==[1536,8960,28,12,2,151936]
    (output/'input-identities.json').write_text(json.dumps(dict(profile_sha256=sha(profile),
        qwen_adapter_sha256=sha(A/'qwen_hbfsim_cosim.py'),model_config_sha256=sha(A/'qwen25_1p5b.json'),
        frozen_matrix_manifest_sha256=sha(root/'manifest.json')),indent=2)+'\n')
    cost=q.LLMCompassCostModel(REPO,profile);base=q.ModelSpec.from_json(A/'qwen25_1p5b.json')
    results=[]
    for summary in json.loads((root/'comparison.json').read_text()):
        name=summary['case'];p=int(name[1:4]);d=int(name[5:7])
        plan=q.build_plan(dataclasses.replace(base,prefill_tokens=p,decode_steps=d),cost)
        path=root/name/'operators.json';ledger=json.loads(path.read_text())
        assert len(plan.operators)==len(ledger)
        phases={}
        for original,row in zip(plan.operators,ledger):
            assert (original.index,original.name,original.phase)==(row['index'],row['name'],row['phase'])
            groups=phases.setdefault(row['phase'],{})
            group=groups.setdefault(row['path'],dict(calls=0,model_ns=0,flops=0,read_bytes=0,write_bytes=0,unclassified_bytes=0))
            group['calls']+=1;group['model_ns']+=row['model_ns'];group['flops']+=original.timing.flops
            for key,field in [('read_bytes','known_read_bytes'),('write_bytes','known_write_bytes'),('unclassified_bytes','unclassified_io_bytes')]:group[key]+=row[field]
        total=sum(g['model_ns'] for groups in phases.values() for g in groups.values())
        assert total==round(summary['model_ms']*1e6)
        for field,expected in [('read_bytes','known_read_bytes'),('write_bytes','known_write_bytes'),('unclassified_bytes','unclassified_io_bytes')]:
            assert sum(g[field] for groups in phases.values() for g in groups.values())==summary[expected]
        hw_path=pathlib.Path(summary['hardware_source']);assert sha(hw_path)==summary['hardware_sha256']
        hardware=json.loads(hw_path.read_text())['rois']
        phases_report={}
        for phase,groups in phases.items():
            ns=sum(g['model_ns'] for g in groups.values())
            roi='Prefill' if phase=='prefill' else 'D'+phase.split('_')[1]
            hardware_ms=hardware.get(roi,{}).get('natural_cuda_event_ms',{}).get('median')
            for family,g in groups.items():
                g['model_ms']=g['model_ns']/1e6;g['phase_time_share_pct']=100*g['model_ns']/ns
                if family in ('OFFICIAL_MATMUL','OFFICIAL_BATCHED_MATMUL'):
                    g['ideal_tensor_compute_ms']=g['flops']/cost.device.compute_module.total_systolic_array_flops*1000
            phases_report[phase]=dict(model_ms=ns/1e6,hardware_phase_median_ms=hardware_ms,
                phase_error_pct=None if hardware_ms is None else 100*(ns/1e6/hardware_ms-1),groups=groups)
        results.append(dict(case=name,phases=phases_report,ledger_sha256=sha(path),hardware_sha256=sha(hw_path),
            full_model_ms=summary['model_ms'],full_hardware_ms=summary['hardware_ms'],full_error_pct=summary['time_error_pct'],
            scope='Frozen official-body Qwen ledger with declared adapters; independent ROI medians are not additive corrections; ideal FLOP bound is not simulated time'))
    (output/'full-inference-breakdown.json').write_text(json.dumps(results,indent=2)+'\n')
    return results

def paired_linear_groups():
    """Native call-layout intervention; not a model parameter or full pipeline."""
    torch.manual_seed(0)
    rows=[]
    scrub_buffer=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
    with torch.no_grad():
        for m in (1,512):
          for group,widths in [('qkv',(1536,256,256)),('gate_up',(8960,8960))]:
            x=torch.randn((m,1536),dtype=torch.bfloat16,device='cuda')
            weight=torch.randn((sum(widths),1536),dtype=torch.bfloat16,device='cuda')
            bias=torch.randn((sum(widths),),dtype=torch.bfloat16,device='cuda') if group=='qkv' else None
            pieces=weight.split(widths);biases=bias.split(widths) if bias is not None else (None,)*len(widths)
            def fused():return torch.nn.functional.linear(x,weight,bias)
            def separated():return tuple(torch.nn.functional.linear(x,w,b) for w,b in zip(pieces,biases))
            reference=fused();parts=separated()
            maximum_error=max(float((a.float()-b.float()).abs().max()) for a,b in zip(reference.split(widths,dim=-1),parts))
            assert all(torch.allclose(a,b,atol=0.5,rtol=0.03) for a,b in zip(reference.split(widths,dim=-1),parts))
            for condition in ('warm','scrub'):
                samples={'fused':[],'separated':[]};clocks=[]
                for repeat in range(7):
                    # Reverse order on alternate repeats; keep exact operands.
                    order=('fused','separated') if repeat%2==0 else ('separated','fused')
                    for mode in order:
                        fn=fused if mode=='fused' else separated
                        for _ in range(20):result=fn()
                        if condition=='scrub':scrub_buffer.fill_(repeat+1)
                        torch.cuda.synchronize()
                        before=clock_boundary()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();result=fn();end.record();end.synchronize()
                        samples[mode].append(start.elapsed_time(end))
                        clocks.append(dict(repeat=repeat,mode=mode,before=before,after=clock_boundary()))
                medians={mode:statistics.median(values) for mode,values in samples.items()}
                rows.append(dict(M=m,K=1536,output_widths=widths,group=group,condition=condition,
                    cuda_event_ms=medians,samples_ms=samples,clocks=clocks,numerical_max_abs_error=maximum_error,
                    separated_minus_fused_ms=medians['separated']-medians['fused'],
                    scope='Same BF16 operands/output contract; F.linear call layout only; phase/workflow not modeled; scrub is a residency intervention, not guaranteed cold cache; event intervals include within-call dispatch.'))
                print('PAIRED_LINEAR',json.dumps({k:v for k,v in rows[-1].items() if k not in ('samples_ms','clocks')}),flush=True)
    return dict(rows=rows,torch_version=torch.__version__,trace_exported=False,profile_changed=False,
        warning='No model request is removed; no observed latency is fitted into the hardware profile; these standalone contexts differ from full native inference.')


def inference_cause_audit(output):
    """Verify frozen inputs and compare model components without altering them."""
    old=A/'results/official-inference-matrix-20261007-r2'
    new=A/'results/official-inference-matrix-20261007-r3'
    native=pathlib.Path('/home/xmu/nvidiagds/simulators/wgslogs/qwen-ncu/qwen1.5-series/timing-diagnosis-20261007-r8')
    manifests=[json.loads((root/'manifest.json').read_text()) for root in (old,new)]
    assert all(m['status']=='PASS_FULL_MATRIX_ACCOUNTING' and m['completed_cases']==8 for m in manifests)
    assert manifests[1]['hbfsim_in_primary_time'] is False
    profiles=[]
    identities={}
    for root,manifest in zip((old,new),manifests):
        snapshot=root/'source'/pathlib.Path(manifest['profile']).name
        assert sha(snapshot)==manifest['profile_sha256']
        profiles.append(json.loads(snapshot.read_text()))
        for name,digest in manifest['source_sha256'].items():
            assert sha(root/'source'/name)==digest
        identities[str(root)]=dict(manifest_sha256=sha(root/'manifest.json'),profile_sha256=sha(snapshot),
            comparison_sha256=sha(root/'comparison.json'))
    for name in ('qwen_hbfsim_cosim.py','mapper_matmul.py','mapper_softmax.py'):
        assert manifests[0]['source_sha256'][name]==manifests[1]['source_sha256'][name]
    def differences(left,right,path=''):
        if isinstance(left,dict) and isinstance(right,dict):
            assert set(left)==set(right)
            return [row for key in left for row in differences(left[key],right[key],path+'/'+key)]
        return [] if left==right else [dict(path=path,old=left,new=right)]
    device_changes=differences(profiles[0]['device'],profiles[1]['device'])
    assert device_changes==[dict(path='/compute_chiplet/core/systolic_array/mac_per_cycle',old=0.25,new=0.5)]
    original_diff=subprocess.run(['git','diff','--exit-code','origin/ISCA_AE','--',
        'software_model','hardware_model','ae/figure5'],cwd=REPO,capture_output=True,text=True)
    assert original_diff.returncode==0 and original_diff.stdout==''
    shapes=json.loads((new/'shape-cache.json').read_text())
    assert all(row['official_latency_parity'] and math.isclose(row['seconds'],row['official_source_seconds'],rel_tol=1e-12,abs_tol=1e-15) for row in shapes)
    receipt=json.loads((native/'receipt.json').read_text())
    native_identity=json.loads((native/'input-identities.json').read_text())
    assert receipt['status']=='PASS_AGGREGATE_TIMING_DIAGNOSIS_NOT_NEW_ACCEPTANCE'
    assert receipt['source_unchanged'] and receipt['trace_exported'] is False
    assert sha(native/'diagnose_inference_timing.py')==receipt['driver_sha256']==native_identity['driver_sha256']
    assert receipt['native_sources']==native_identity['native_sources']
    for source in receipt['native_sources']:assert sha(source['path'])==source['sha256']
    reuse=json.loads((native/'reuse-manifest.json').read_text())
    prior=pathlib.Path(reuse['source_root'])
    assert reuse['unchanged_control_ast']
    assert sha(prior/'diagnose_inference_timing.py')==reuse['source_driver_sha256']
    def control_ast(path):
        names={'run','boundary_call','sample_clocks','ClockBoundary'}
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names}
    assert control_ast(prior/'diagnose_inference_timing.py')==control_ast(native/'diagnose_inference_timing.py')
    for filename,field in [('native-partitions.json','native_partitions_sha256'),('device-activity-windows.json','device_activity_sha256')]:
        assert sha(prior/filename)==sha(native/filename)==reuse[field]
    native_files=['receipt.json','input-identities.json','reuse-manifest.json','native-partitions.json','device-activity-windows.json','kernel-groups.json']
    identities[str(native)]={name:sha(native/name) for name in native_files}
    grouped=json.loads((native/'kernel-groups.json').read_text())
    device=json.loads((native/'device-activity-windows.json').read_text())
    baselines=json.loads((native/'native-partitions.json').read_text())
    assert len(grouped)==9 and len(device)==18 and len(baselines)==8
    assert {(x['case'],x['repeat']) for x in grouped}=={(c,i) for c in ('p032d02','p128d02','p512d02') for i in range(3)}
    assert {(x['case'],x['phase'],x['repeat']) for x in device}=={(c,p,i) for c in ('p032d02','p512d02') for p in ('Prefill','Decode1','Decode2') for i in range(3)}
    for row in grouped:
        assert all(g['phase'] in ('Prefill','Decode1','Decode2') for g in row['groups'])
        assert sum(g['kernels'] for g in row['groups'])==row['cuda_event_count']
        assert abs(sum(g['kernel_us'] for g in row['groups'])-row['all_cuda_us'])<0.001
        assert abs(row['all_grouped_us']-row['all_cuda_us'])<0.001
        for g in row['groups']:assert sum(g['kernel_names'].values())==g['kernels']
        for phase,interval in row['instrumented_phase_ms'].items():
            union=row['gpu_busy_union_ms'][phase];gap=row['gpu_interval_outside_recorded_activity_ms'][phase]
            assert union<=interval+0.005 and abs(interval-union-gap)<1e-9
    for row in device:
        assert sum(row['kernel_names'].values())==row['device_activity_count']
        assert row['gpu_busy_union_ms']<=row['cuda_event_ms']+0.005
        assert abs(row['cuda_event_ms']-row['gpu_busy_union_ms']-row['outside_recorded_activity_ms'])<1e-9
    model_groups={
        'token_embedding':'embedding','input_rmsnorm':'norm_residual','post_attention_rmsnorm':'norm_residual',
        'final_rmsnorm':'norm_residual','attention_residual':'norm_residual','mlp_residual':'norm_residual',
        'q_proj':'qkv','k_proj':'qkv','v_proj':'qkv','rope':'rope','kv_append':'attention',
        'attention_score':'attention','attention_softmax':'attention','attention_value':'attention',
        'o_proj':'o_proj','gate_proj':'gate_up','up_proj':'gate_up','silu':'silu_mul',
        'gated_multiply':'silu_mul','down_proj':'down','lm_head':'logits'}
    native_groups={g:g for g in ('embedding','qkv','rope','attention','o_proj','gate_up','silu_mul','down','logits','framework_or_other','sampling')}
    native_groups.update({g:'norm_residual' for g in ('input_norm','post_norm','final_norm')})
    old_rows={x['case']:x for x in json.loads((old/'comparison.json').read_text())}
    new_rows={x['case']:x for x in json.loads((new/'comparison.json').read_text())}
    comparisons=[]
    for case,current in new_rows.items():
        previous=old_rows[case]
        ledgers=[json.loads((root/case/'operators.json').read_text()) for root in (old,new)]
        assert len(ledgers[0])==len(ledgers[1])==507*(int(case.split('d')[1])+1)
        component={};profile_effect={}
        for l,r in zip(*ledgers):
            assert (l['index'],l['phase'],l['name'],l['path'])==(r['index'],r['phase'],r['name'],r['path'])
            if r['phase'].startswith('decode_'):
                assert {k:l[k] for k in ('model_ns','known_read_bytes','known_write_bytes','unclassified_io_bytes')}=={k:r[k] for k in ('model_ns','known_read_bytes','known_write_bytes','unclassified_io_bytes')}
            groups=component.setdefault(r['phase'],{})
            g=groups.setdefault(model_groups[r['name']],dict(model_ns=0,calls=0,known_read_bytes=0,known_write_bytes=0,unclassified_io_bytes=0))
            for key in ('model_ns','known_read_bytes','known_write_bytes','unclassified_io_bytes'):g[key]+=r[key]
            g['calls']+=1
            effect=profile_effect.setdefault(r['phase'],dict(old_ns=0,new_ns=0))
            effect['old_ns']+=l['model_ns'];effect['new_ns']+=r['model_ns']
        assert sum(g['model_ns'] for gs in component.values() for g in gs.values())==round(current['model_ms']*1e6)
        for field in ('known_read_bytes','known_write_bytes','unclassified_io_bytes'):
            assert sum(g[field] for gs in component.values() for g in gs.values())==current[field]
        native_samples=[x for x in grouped if x['case']==case]
        comparison={}
        for phase,groups in component.items():
            hardware_phase='Prefill' if phase=='prefill' else 'Decode'+phase.split('_')[1]
            comparison[phase]=[]
            for name,g in groups.items():
                samples=[]
                for sample in native_samples:
                    samples.append(sum(x['kernel_us'] for x in sample['groups'] if x['phase']==hardware_phase and native_groups[x['group']]==name)/1000)
                ms=g['model_ns']/1e6
                observed=statistics.median(samples) if samples else None
                comparison[phase].append(dict(group=name,model_ms=ms,diagnostic_native_kernel_median_ms=observed,
                    difference_ms=None if observed is None else ms-observed,**g))
        hardware_path=pathlib.Path(current['hardware_source']);assert sha(hardware_path)==current['hardware_sha256']
        hardware=json.loads(hardware_path.read_text())['rois']
        for phase,effect in profile_effect.items():
            effect.update(old_ms=effect['old_ns']/1e6,new_ms=effect['new_ns']/1e6,model_delta_ms=(effect['new_ns']-effect['old_ns'])/1e6)
            roi='Prefill' if phase=='prefill' else 'D'+phase.split('_')[1]
            effect['archived_hardware_ms']=hardware.get(roi,{}).get('natural_cuda_event_ms',{}).get('median')
            effect['new_error_pct']=None if effect['archived_hardware_ms'] is None else 100*(effect['new_ms']/effect['archived_hardware_ms']-1)
        comparisons.append(dict(case=case,old_full_ms=previous['model_ms'],new_full_ms=current['model_ms'],
            archived_hardware_ms=current['hardware_ms'],new_full_error_pct=current['time_error_pct'],
            profile_only_effect=profile_effect,model_to_native_group_comparison=comparison))
    windows=[]
    for case in ('p032d02','p512d02'):
        for phase in ('Prefill','Decode1','Decode2'):
            rows=[x for x in device if (x['case'],x['phase'])==(case,phase)]
            model_phase='prefill' if phase=='Prefill' else 'decode_'+phase[-1]
            model_ms=new_rows[case]['phase_totals'][model_phase]['model_ns']/1e6
            medians={k:statistics.median(x[k] for x in rows) for k in ('cuda_event_ms','gpu_busy_union_ms','outside_recorded_activity_ms','device_activity_count')}
            windows.append(dict(case=case,phase=phase,model_ms=model_ms,**medians,
                model_vs_gpu_activity_pct=100*(model_ms/medians['gpu_busy_union_ms']-1),
                model_vs_event_pct=100*(model_ms/medians['cuda_event_ms']-1)))
    result=dict(status='PASS_ROOT_CAUSE_ACCOUNTING_NOT_COMPLETE_CAUSAL_ATTRIBUTION',inputs=identities,
        only_device_change=device_changes,official_source_diff_bytes=0,unique_shape_latency_parity_count=len(shapes),
        hbfsim_primary_time_contribution_ns=0,profile_effect_and_group_comparisons=comparisons,
        same_run_device_activity_windows=windows,
        limits=['Rich CPU+CUDA profiling perturbs full event intervals; only diagnostic grouped kernel durations are used.',
            'Model logical groups and fused native groups are semantically aligned sets, not identical kernels.',
            'The activity residual is not pure CPU overhead and is not transported to unprofiled timing.',
            'Medians of components need not add to a median full interval.',
            'No per-group physical DRAM-counter attribution is established by this timing audit.',
            'Zero observed latency-porting differences do not prove every adapter or architectural assumption accurate.'])
    (output/'inference-causes.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def official_breakdown(op,name,device):
    """Read original tile-stage decisions; never infer bytes from rounded cycles."""
    if name=='gelu':
        cm=device.compute_module;vu=cm.core.vector_unit
        parallelism=cm.core_count*vu.vector_width*vu.vector_count
        padded=math.ceil(op.computational_graph.M/parallelism)*parallelism
        bytes_each=padded*op.computational_graph.data_type.word_size
        main=2*bytes_each/device.io_module.bandwidth
        local=2*bytes_each/cm.l2_bandwidth_per_cycle/cm.clock_freq
        compute=padded*(10+vu.flops_per_exp)/cm.total_vector_flops
        expected=max(main+local,compute)
        assert math.isclose(expected,op.compile_and_simulate(device,'heuristic-GPU'),rel_tol=1e-12)
        return dict(padded_elements=padded,main_read_bytes=bytes_each,main_write_bytes=bytes_each,
            main_io_ms=main*1000,global_io_ms=local*1000,compute_ms=compute*1000,
            body_ms=expected*1000,scope='Original GeLU max(compute, main IO + global IO), with original lane-alignment padding')
    if name not in ('softmax','layernorm'):
        return {'scope':'No new decomposition of this operator'}
    graph=op.computational_graph;mapping=op.best_mapping
    totals={'main_read_cycles':0,'onchip_cycles':0,'main_write_cycles':0,
            'main_read_bytes':0,'main_write_bytes':0}
    for first in range(0,graph.M,mapping.l2_tile_M):
        m=min(mapping.l2_tile_M,graph.M-first)
        tile=op.L2TileSimulator(m,graph.N,graph.data_type,mapping,device)
        totals['main_read_cycles']+=tile.read_cycle_count
        totals['onchip_cycles']+=tile.compute_cycle_count
        totals['main_write_cycles']+=tile.write_cycle_count
        # These are the operands of the original transfer decision, not cycle*BW.
        transferred=m*graph.N*graph.data_type.word_size
        totals['main_read_bytes']+=transferred
        totals['main_write_bytes']+=transferred
    cycles=sum(totals[k] for k in ('main_read_cycles','onchip_cycles','main_write_cycles'))
    assert math.isclose(cycles,op.best_cycle_count,rel_tol=1e-12,abs_tol=1e-8)
    totals['mapping']=vars(mapping)
    totals['scope']='Official serial outer read/onchip/write recurrence, unchanged'
    totals['component_ms']={k:totals[k]/device.compute_module.clock_freq*1000
        for k in ('main_read_cycles','onchip_cycles','main_write_cycles')}
    return totals

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=pathlib.Path)
    p.add_argument('--suite',choices=['representative','small-operators','counters','timer-controls','tensor-diagnostic','explicit-fp32','paired-linear','full-breakdown','inference-causes'],default='representative')
    p.add_argument('--counter-task',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.counter_task:counter_worker(json.loads(args.counter_task));return
    assert args.output is not None,'--output is required for collection'
    args.output.mkdir(exist_ok=False,parents=True)
    if args.suite not in ('full-breakdown','inference-causes'):
        assert torch.cuda.get_device_properties(0).multi_processor_count==48
        assert torch.cuda.get_device_capability(0)==(8,9)
    sources={str(path.relative_to(REPO)):sha(path) for path in
        [REPO/'software_model'/f'{s}.py' for s in ['matmul','softmax','layernorm','gelu','transformer']]
        +[REPO/'hardware_model/compute_module.py']}
    # Preserve the exact driver and immutable original operator identities.
    shutil.copy2(__file__,args.output/'validate_ae_ada.py')
    (args.output/'source-identities.json').write_text(json.dumps(sources,indent=2)+'\n')
    if args.suite=='paired-linear':
        result=paired_linear_groups()
        (args.output/'paired-linear-groups.json').write_text(json.dumps(result,indent=2)+'\n')
        assert sources=={name:sha(REPO/name) for name in sources}
        (args.output/'receipt.json').write_text(json.dumps(dict(status='PASS_PAIRED_LINEAR_CALL_LAYOUT_DIAGNOSTIC',completed_conditions=len(result['rows']),
            sources=sources,driver_sha256=sha(__file__),trace_exported=False,profile_changed=False),indent=2)+'\n')
        print('PASS_PAIRED_LINEAR_CALL_LAYOUT_DIAGNOSTIC',len(result['rows']),flush=True);return
    if args.suite=='inference-causes':
        result=inference_cause_audit(args.output)
        assert sources=={name:sha(REPO/name) for name in sources}
        (args.output/'receipt.json').write_text(json.dumps(dict(status=result['status'],completed_cases=len(result['profile_effect_and_group_comparisons']),
            unique_shape_latency_parity_count=result['unique_shape_latency_parity_count'],sources=sources,
            driver_sha256=sha(__file__),trace_exported=False,profile_changed=False),indent=2)+'\n')
        print(result['status'],len(result['profile_effect_and_group_comparisons']),flush=True);return
    if args.suite=='explicit-fp32':
        result=explicit_fp32_gemm_diagnostic()
        (args.output/'explicit-fp32-gemm.json').write_text(json.dumps(result,indent=2)+'\n')
        assert sources=={name:sha(REPO/name) for name in sources}
        (args.output/'receipt.json').write_text(json.dumps(dict(status='PASS_EXPLICIT_FP32_GEMM_DIAGNOSTIC',sources=sources,driver_sha256=sha(__file__),completed_cases=len(result['rows']),trace_exported=False,profile_changed=False),indent=2)+'\n')
        print('PASS_EXPLICIT_FP32_GEMM_DIAGNOSTIC',len(result['rows']),flush=True);return
    if args.suite=='tensor-diagnostic':
        result=tensor_throughput_diagnostic()
        (args.output/'tensor-throughput-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
        assert sources=={name:sha(REPO/name) for name in sources}
        (args.output/'receipt.json').write_text(json.dumps(dict(status='PASS_INDEPENDENT_TENSOR_DIAGNOSTIC_NOT_PROFILE_CALIBRATION',sources=sources,driver_sha256=sha(__file__),completed_cases=len(result['rows']),trace_exported=False,profile_changed=False),indent=2)+'\n')
        print('PASS_INDEPENDENT_TENSOR_DIAGNOSTIC_NOT_PROFILE_CALIBRATION',len(result['rows']),flush=True);return
    if args.suite=='full-breakdown':
        results=full_inference_breakdown(args.output)
        receipt=dict(status='PASS_FULL_LEDGER_BREAKDOWN_NOT_NEW_SIMULATION_OR_ACCURACY',completed_cases=len(results),sources=sources,
            driver_sha256=sha(__file__),trace_exported=False,full_window_time_gate='FAIL',cases_within_ten_percent=sum(abs(r['full_error_pct'])<=10 for r in results))
        (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(receipt['status'],len(results),flush=True);return
    if args.suite=='timer-controls':
        result=interleaved_gelu_controls()
        (args.output/'interleaved-gelu-controls.json').write_text(json.dumps(result,indent=2)+'\n')
        (args.output/'receipt.json').write_text(json.dumps(dict(status='PASS_INTERLEAVED_TIMER_CONTROLS_NOT_MODEL_CALIBRATION',sources=sources,driver_sha256=sha(__file__),trace_exported=False),indent=2)+'\n')
        print('TIMER_CONTROLS',json.dumps(result),flush=True);return
    if args.suite=='counters':
        results=collect_counters(args.output)
        receipt={'status':'PASS_INDEPENDENT_OPERATOR_COUNTER_DIAGNOSTIC','invocations':len(results),
            'sources':sources,'driver_sha256':sha(__file__),'trace_exported':False}
        assert sources=={name:sha(REPO/name) for name in sources}
        (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(receipt['status'],len(results),flush=True);return
    classes={'matmul':Matmul,'softmax':Softmax,'layernorm':LayerNorm,'gelu':GeLU}
    measured={}
    for name,cls in classes.items():
        # Use the original method, including its original 50 repetitions and
        # synchronization conventions. Silence verbose sample-list printing.
        rounds=[]
        for _ in range(3):
            with contextlib.redirect_stdout(io.StringIO()):
                value=cls.gpu_kernel_launch_overhead()
            assert value>0;rounds.append(value)
        measured[name]=dict(seconds=statistics.median(rounds),round_medians_seconds=rounds,
            method=f'software_model.{name}.{cls.__name__}.gpu_kernel_launch_overhead',
            units='seconds',scope='official synchronized minimal-input wall-time baseline')
        print('OFFICIAL_OVERHEAD',name,json.dumps(measured[name]),flush=True)
    profile=json.loads((A/'RTX4000Ada_xmu_profile_v3.json').read_text())
    profile['name']='NVIDIA RTX 4000 Ada v4: precision correction and official independently measured overhead'
    profile['device']['operator_overhead_seconds']={k:v['seconds'] for k,v in measured.items()}
    profile['device']['operator_overhead_seconds']['policy']='Official synchronized microbenchmarks; not Qwen fit, not a pure GPU-event correction'
    profile['calibration_status']['official_overhead_measurements']=measured
    profile['calibration_status']['class']='RTX4000_ADA_OFFICIAL_OPERATOR_BASELINE_MEASURED_V4'
    profile['calibration_status']['precision_evidence']['unchanged']='48 SM, 2.175 GHz, 342 GB/s, 40 MiB global buffer; dense BF16/FP32 tensor throughput retained; only existing official overhead fields independently measured, not fitted to validation shapes'
    profile_path=args.output/'RTX4000Ada_xmu_profile_v4.json'
    profile_path.write_text(json.dumps(profile,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('ae_ada_cost',A/'qwen_hbfsim_cosim.py')
    q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
    cost=q.LLMCompassCostModel(REPO,profile_path)
    chip=profile['device']['compute_chiplet'];array=chip['core']['systolic_array']
    expected=chip['core_count']*chip['core']['sublane_count']*array['array_width']*array['array_height']*array['mac_per_cycle']*2*profile['device']['frequency_Hz']
    assert cost.device.compute_module.total_systolic_array_flops==expected
    work=pathlib.Path(tempfile.mkdtemp(prefix='llmcompass-ada-ae-'))
    shutil.copytree(REPO/'systolic_array_model',work/'systolic_array_model',ignore=shutil.ignore_patterns('temp','*.gz','*.npy'))
    (work/'systolic_array_model/temp').mkdir();os.chdir(work)
    dtype=data_type_dict['fp16'];rows=[]
    examples=small_operator_examples() if args.suite=='small-operators' else [
        ('matmul',[32,12288,12288],'AE_REPRESENTATIVE'),('softmax',[4096,1024],'AE_REPRESENTATIVE'),
        ('layernorm',[4096,4096],'AE_REPRESENTATIVE'),('gelu',[65536],'AE_REPRESENTATIVE')]
    # Independent calibration above is frozen before these validation shapes.
    for name,shape,group in examples:
        op=classes[name](dtype)
        if name=='matmul':
            m,k,n=shape;op(Tensor([m,k],dtype),Tensor([k,n],dtype))
        else:op(Tensor(shape,dtype))
        # Retain the original method and its iteration count in each repeat.
        hardware_rounds=[];clocks=[]
        for _ in range(3):
            hardware_rounds.append(op.run_on_gpu());clocks.append(clock_boundary())
        hardware=statistics.median(hardware_rounds)
        body=op.compile_and_simulate(cost.device,'heuristic-GPU')
        overhead=measured[name]['seconds'];total=body+overhead
        row=dict(operator=name,shape=shape,model_body_ms=body*1000,
            official_overhead_ms=overhead*1000,model_total_ms=total*1000,
            official_gpu_synchronized_wall_ms=hardware*1000,error_pct=100*(total/hardware-1),
            hardware_dtype='bfloat16' if name=='matmul' else 'float16',
            model_storage_type='fp16; two-byte BF16 storage proxy' if name=='matmul' else 'fp16',
            hardware_method='Unmodified official run_on_gpu; native iterations and warmup retained',
            hardware_round_medians_ms=[v*1000 for v in hardware_rounds],
            breakdown=official_breakdown(op,name,cost.device),paper_shape=group.startswith('AE_'),
            group=group,iterations_per_round=op.iterations,boundary_clocks=clocks)
        rows.append(row);(args.output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('OFFICIAL_OPERATOR',json.dumps(row),flush=True)
    diagnostics=[]
    if args.suite=='small-operators':
        tasks=[('softmax',[4096,1024],torch.float16,False,False),
               ('softmax',[4096,1024],torch.float16,False,True),
               ('gelu',[1,1],torch.float32,False,False),
               ('gelu',[1],torch.float16,False,False),
               ('gelu',[65536],torch.float16,False,False),
               ('gelu',[65536],torch.float16,True,False),
               ('gelu',[2**26],torch.float16,False,False)]
        for task in tasks:
            diagnostics.append(dual_clock_diagnostic(*task))
            (args.output/'timer-diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
            print('TIMER_DIAGNOSTIC',json.dumps(diagnostics[-1]),flush=True)
    assert sources=={name:sha(REPO/name) for name in sources}
    receipt=dict(status='PASS_SMALL_OPERATOR_GRID_NOT_FULL_LLM_ACCEPTANCE' if args.suite=='small-operators' else 'PASS_REPRESENTATIVE_AE_EXECUTION_NOT_FULL_AE_MATRIX',
        sources=sources,driver_sha256=sha(__file__),profile_sha256=sha(profile_path),
        completed_operators=len(rows),working_directory=str(work),
        scope='Original Figure 5 Softmax and GeLU grids plus recorded Qwen Softmax shapes' if args.suite=='small-operators' else 'Four original Figure 5 operator shapes; Ada instead of A100; no Qwen-fitted coefficient',
        planned_operators=len(examples),suite=args.suite,trace_exported=False,
        time_gate='PASS' if all(abs(r['error_pct'])<=10 for r in rows) else 'FAIL',
        overhead_clock='Synchronized wall-time baseline; not a pure launch or CUDA-event duration',
        environment={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0),
            'published_environment':'Python 3.9, PyTorch 2.0, CUDA 11.7; not the active environment'},
        untouched='Official software_model source unchanged; no cache/fusion/HBFSim additions')
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'],receipt['time_gate'],flush=True)

if __name__=='__main__':main()
