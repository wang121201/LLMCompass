"""Lower the official outer-buffer pipeline to memory-completion dependencies.

Tile compute cycles retain official local/global transfers; off-chip IO cycles
are replaced by HBFSim transaction completions, not added to their durations.
"""
def analytical_cycles(stages):
    return sum((max(s['read_cycles'],s['compute_cycles']) if s['overlap']
                else s['read_cycles']+s['compute_cycles'])+s['write_cycles'] for s in stages)

def dependent_compute_stage(compute_seconds, read_event_count, write_event_count, clock_hz):
    """Coarse dependency coupling for vector/official decode shortcut paths.

    This retains their configured pure-compute cost, not their max(IO,compute)
    latency. It is explicitly a coupling policy, not tile-mapper parity.
    """
    if compute_seconds<0 or clock_hz<=0:
        raise ValueError('invalid compute cost or clock')
    return [dict(kind='dependent_compute_only',
                 read_events=list(range(read_event_count)),read_cycles=0,
                 compute_cycles=compute_seconds*clock_hz,
                 write_events=list(range(read_event_count,read_event_count+write_event_count)),
                 write_cycles=0,overlap=False)]

def lower_stages(Transaction, stages, memory_events, clock_hz, event_requests):
    """event_requests(event) supplies address-correct (target, addr, bytes) rows.

    Return all transactions and final blocking frontier. Main-memory events
    must be emitted once, in mapper order. Requests within a tensor event may
    execute concurrently; consecutive events follow the original IO order.
    """
    if clock_hz<=0:
        raise ValueError('positive configured clock required')
    txs=[]; frontier=(); used=[]
    def memory_chain(indices, dependencies):
        for index in indices:
            used.append(index)
            event=memory_events[index]; ids=[]
            for row,(target,addr,count) in enumerate(event_requests(event)):
                if count<=0:raise ValueError('nonpositive memory request')
                name=f'mem{index}-row{row}'
                txs.append(Transaction(id=name,target=target,op='R' if event['direction']=='read' else 'W',addr=addr,bytes=count,issue_ns=0,dependencies=tuple(dependencies)))
                ids.append(name)
            if not ids:raise ValueError('empty mapper transfer')
            dependencies=tuple(ids)
        return tuple(dependencies)
    for index,stage in enumerate(stages):
        reads=memory_chain(stage['read_events'],frontier)
        name=f'tile{index}-onchip'
        # Fractional ns is intentional: integer mapper cycles / SM clock.
        compute_dependencies=frontier if stage['overlap'] else reads
        txs.append(Transaction(id=name,target='BARRIER',op=None,addr=0,bytes=0,issue_ns=0,duration_ns=stage['compute_cycles']*1e9/clock_hz,dependencies=tuple(compute_dependencies)))
        join=tuple(dict.fromkeys((*reads,name)))
        frontier=memory_chain(stage['write_events'],join)
    if used!=list(range(len(memory_events))):
        raise ValueError('mapper memory event ordering/coverage mismatch')
    return txs,set(frontier)

def audit_completion_dependencies(transactions, memory_finishes, frontier, blocking_finish_ns):
    """Check completion causality, reconstructing analytical barrier finishes.

    All timestamps must use the same batch-relative origin. This validates
    dependency timing, not native CUDA execution or numerical tensor values.
    """
    ends={}
    for tx in transactions:
        ready=max([tx.issue_ns]+[ends[d] for d in tx.dependencies])
        if tx.target=='BARRIER':
            ends[tx.id]=ready+tx.duration_ns
        else:
            ends[tx.id]=memory_finishes[tx.id]
            if ends[tx.id]+1e-6<ready:
                raise ValueError('memory completion violates dependency frontier')
    expected=max(ends[x] for x in frontier)
    if abs(expected-blocking_finish_ns)>1e-5:
        raise ValueError('blocking completion does not close against DAG frontier')
    return ends

def self_test():
    from types import SimpleNamespace
    events=[dict(direction='read'),dict(direction='write')]
    stage=dict(read_events=[0],read_cycles=100,compute_cycles=200,
               write_events=[1],write_cycles=30,overlap=True)
    txs,frontier=lower_stages(SimpleNamespace,[stage],events,1e9,lambda e:[('HBM',0,32)])
    assert analytical_cycles([stage])==230
    assert txs[1].dependencies==() and txs[1].duration_ns==200
    assert set(txs[2].dependencies)=={txs[0].id,txs[1].id}
    assert frontier=={txs[2].id}
    fast=audit_completion_dependencies(txs,{txs[0].id:100,txs[2].id:230},frontier,230)
    slow=audit_completion_dependencies(txs,{txs[0].id:1000,txs[2].id:1030},frontier,1030)
    assert fast[txs[1].id]==slow[txs[1].id]==200
    assert slow[txs[2].id]-fast[txs[2].id]==800
    try:audit_completion_dependencies(txs,{txs[0].id:1000,txs[2].id:230},frontier,230)
    except ValueError:pass
    else:raise AssertionError('write before input completion accepted')
    stage['overlap']=False
    txs,_=lower_stages(SimpleNamespace,[stage],events,1e9,lambda e:[('HBM',0,32)])
    assert analytical_cycles([stage])==330
    assert txs[1].dependencies==(txs[0].id,)
    print('PASS_PIPELINED_AND_SERIAL_DEPENDENCIES')

if __name__=='__main__':self_test()
