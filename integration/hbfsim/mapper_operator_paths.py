"""Compute-only vector/decode and independent-batch completion coupling.

These policies do not use a memory-inclusive roofline duration as compute.
Merged BatchedMatmul is deliberately not silently converted to this path.
"""
from mapper_event_coupling import dependent_compute_stage, lower_stages

def access_execution(Transaction, operator, compute_seconds, clock_hz):
    accesses=operator.reads+operator.writes
    events=[dict(direction=a.kind,access=a) for a in accesses]
    stages=dependent_compute_stage(compute_seconds,len(operator.reads),len(operator.writes),clock_hz)
    return lower_stages(Transaction,stages,events,clock_hz,
                        lambda e:[(e['access'].target,e['access'].address,e['access'].bytes)])

def vector_execution(Transaction,operator,device):
    cm=device.compute_module
    seconds=operator.timing.flops/(cm.total_vector_flops_per_cycle*cm.clock_freq)
    return access_execution(Transaction,operator,seconds,cm.clock_freq)

def decode_execution(Transaction,operator,m,k,n,device):
    if m!=1 and n!=1:
        raise ValueError('official vector shortcut applies only to M=1 or N=1')
    return vector_execution(Transaction,operator,device)

def independent_batch_execution(Transaction,views,m,k,n,device,selected_candidate):
    if selected_candidate!=0:
        raise ValueError('merged batch approximation needs a distinct execution policy')
    if m!=1 and n!=1:
        raise ValueError('tiled independent batches require mapper stage composition')
    cm=device.compute_module
    seconds=2*m*k*n/(cm.total_vector_flops_per_cycle*cm.clock_freq)
    all_transactions=[];previous=();identifiers=set()
    for batch,matrix_views in enumerate(views):
        events=[dict(direction='read',tensor='A',row=0,rows=m,column=0,columns=k),
                dict(direction='read',tensor='B',row=0,rows=k,column=0,columns=n),
                dict(direction='write',tensor='C',row=0,rows=m,column=0,columns=n)]
        stages=dependent_compute_stage(seconds,2,1,cm.clock_freq)
        txs,frontier=lower_stages(Transaction,stages,events,cm.clock_freq,
                                  lambda e:matrix_views[e['tensor']].requests(e))
        def name(identifier):return f'batch{batch}-{identifier}'
        for t in txs:
            dependencies=tuple(name(d) for d in t.dependencies) if t.dependencies else previous
            identifier=name(t.id)
            assert identifier not in identifiers
            identifiers.add(identifier)
            all_transactions.append(Transaction(id=identifier,target=t.target,op=t.op,
                addr=t.addr,bytes=t.bytes,issue_ns=t.issue_ns,
                duration_ns=getattr(t,'duration_ns',0),dependencies=dependencies))
        previous=tuple(name(i) for i in frontier)
    if not previous:raise ValueError('empty independent batch')
    return all_transactions,set(previous)
