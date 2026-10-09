"""Bounded row/column lowering for analytical tensor allocations.

These are model addresses, not CUDA virtual addresses or measured cache lines.
Strides support transposed KV views and grouped-query head reuse without
allocating duplicate KV tensors or deleting logical requests.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class MatrixView:
    base: int
    rows: int
    columns: int
    row_stride: int
    column_stride: int
    allocation_start: int
    allocation_bytes: int
    element_bytes: int = 2

    def requests(self,event):
        r0,nr,c0,nc=event['row'],event['rows'],event['column'],event['columns']
        if not (0<=r0<r0+nr<=self.rows and 0<=c0<c0+nc<=self.columns):
            raise ValueError('mapper tile outside matrix view')
        if self.column_stride==self.element_bytes:
            spans=((self.base+r*self.row_stride+c0*self.column_stride,nc*self.element_bytes) for r in range(r0,r0+nr))
        elif self.row_stride==self.element_bytes:
            spans=((self.base+r0*self.row_stride+c*self.column_stride,nr*self.element_bytes) for c in range(c0,c0+nc))
        else:
            spans=((self.base+r*self.row_stride+c*self.column_stride,self.element_bytes) for r in range(r0,r0+nr) for c in range(c0,c0+nc))
        for address,count in spans:
            if not self.allocation_start<=address<address+count<=self.allocation_start+self.allocation_bytes:
                raise ValueError('mapper request outside tensor allocation')
            yield 'HBM',address,count

def grouped_kv_view(base,allocation_bytes,context,head_dim,kv_heads,query_heads,head,score):
    if query_heads%kv_heads or not 0<=head<query_heads:
        raise ValueError('invalid grouped-query head mapping')
    kv_head=head//(query_heads//kv_heads)
    shifted=base+kv_head*head_dim*2
    if score:
        return MatrixView(shifted,head_dim,context,2,kv_heads*head_dim*2,base,allocation_bytes)
    return MatrixView(shifted,context,head_dim,kv_heads*head_dim*2,2,base,allocation_bytes)

def self_test():
    base=4096;context=34;dim=128;kv=2;heads=12
    size=context*dim*kv*2
    for head in range(heads):
        for score in [True,False]:
            view=grouped_kv_view(base,size,context,dim,kv,heads,head,score)
            event=dict(row=0,rows=view.rows,column=0,columns=view.columns)
            spans=list(view.requests(event))
            assert sum(n for _,_,n in spans)==context*dim*2
        left=grouped_kv_view(base,size,context,dim,kv,heads,head,True)
        assert left.base==base+(head//6)*dim*2
    view=MatrixView(0,3,5,10,2,0,30)
    assert list(view.requests(dict(row=1,rows=2,column=2,columns=3)))==[('HBM',14,6),('HBM',24,6)]
    try:list(view.requests(dict(row=0,rows=4,column=0,columns=1)))
    except ValueError:pass
    else:raise AssertionError('out-of-bounds tile accepted')
    print('PASS_STRIDED_GQA_AND_BOUNDS')

if __name__=='__main__':self_test()
