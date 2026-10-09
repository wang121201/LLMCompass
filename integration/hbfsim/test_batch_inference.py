"""Static-request isolation, shape and aggregate-semantics regression tests."""
import dataclasses
import json
from pathlib import Path
import unittest

import qwen_hbfsim_cosim as q
from evaluate_batch_inference import validate_request_regions
import semantic_traffic_breakdown as semantic


class FakeCost:
    def __init__(self): self.calls=[]
    def vector(self,elements,ops_per_element,family,bytes_moved=0):
        self.calls.append(('vector',elements,ops_per_element,family))
        return q.Timing(family,elements*ops_per_element,1)
    def matmul(self,m,k,n,bias=False,family='matmul'):
        self.calls.append(('matmul',m,k,n,family))
        return q.Timing(family,2*m*k*n+(m*n if bias else 0),1)
    def batched_matmul(self,b,m,k,n,family):
        self.calls.append(('batch',b,m,k,n,family))
        return q.Timing(family,2*b*m*k*n,1)
    def softmax(self,b,m,n):
        self.calls.append(('softmax',b,m,n))
        return q.Timing('softmax',b*m*n*5,1)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.base=q.ModelSpec.from_json(Path(__file__).parent/'qwen25_1p5b.json')
    def plan(self,batch):
        cost=FakeCost()
        model=dataclasses.replace(self.base,batch_size=batch,prefill_tokens=128,decode_steps=8)
        return q.build_plan(model,cost),cost
    def test_all_batches_cover_the_same_full_graph(self):
        for batch in [1,2,4,8,16,32]:
            plan,cost=self.plan(batch)
            self.assertEqual(len(plan.operators),4563)
            self.assertEqual(len(cost.calls),4563)
            self.assertEqual(len(set(op.phase for op in plan.operators)),9)
            self.assertEqual(validate_request_regions(plan)['kv_regions'],batch*28*2)
    def test_weights_are_shared_not_replicated(self):
        baseline,_=self.plan(1)
        for batch in [2,4,8,16,32]:
            plan,_=self.plan(batch)
            self.assertEqual(plan.weights.objects,baseline.weights.objects)
            for op in plan.operators:
                if op.name=='q_proj':
                    self.assertEqual(len([a for a in op.reads if a.label.endswith('q_proj.weight')]),1)
    def test_batched_linear_attention_and_last_token_shapes(self):
        plan,cost=self.plan(32)
        self.assertIn(('matmul',4096,1536,8960,'gate_projection'),cost.calls)
        self.assertIn(('matmul',32,1536,151936,'tied_lm_head'),cost.calls)
        self.assertIn(('batch',384,128,128,128,'grouped_query_attention_score'),cost.calls)
        self.assertIn(('softmax',384,128,128),cost.calls)
        self.assertIn(('batch',384,1,128,136,'grouped_query_attention_score'),cost.calls)
    def test_kv_addresses_detect_alias_and_wrong_append_offset(self):
        plan,_=self.plan(2)
        ops=list(plan.operators)
        index=next(i for i,op in enumerate(ops) if op.name=='kv_append' and op.phase=='prefill')
        op=ops[index];writes=list(op.writes);writes[1]=writes[0]
        ops[index]=dataclasses.replace(op,writes=tuple(writes))
        with self.assertRaises(AssertionError):validate_request_regions(dataclasses.replace(plan,operators=tuple(ops)))
        ops=list(plan.operators)
        index=next(i for i,op in enumerate(ops) if op.name=='kv_append' and op.phase=='decode_1')
        op=ops[index];writes=list(op.writes);writes[0]=dataclasses.replace(writes[0],address=writes[0].address-512)
        ops[index]=dataclasses.replace(op,writes=tuple(writes))
        with self.assertRaises(AssertionError):validate_request_regions(dataclasses.replace(plan,operators=tuple(ops)))
    def test_batch_semantic_inputs_are_counted_once(self):
        plan,_=self.plan(4)
        for name in ['attention_score','attention_value','lm_head']:
            op=next(op for op in plan.operators if op.phase=='prefill' and op.name==name)
            recovered={'operands':{'read':{'A':120,'B':240,'C':0},'write':{'A':0,'B':0,'C':360}}}
            ledger={'path':'OFFICIAL_BATCHED_MATMUL' if name.startswith('attention') else 'OFFICIAL_MATMUL',
                    'known_read_bytes':360,'known_write_bytes':360,'unclassified_io_bytes':0}
            categories,_=semantic.allocate_operator(op,ledger,recovered)
            self.assertEqual(semantic.category_total(categories,'read_bytes'),360)
            self.assertEqual(semantic.category_total(categories,'write_bytes'),360)
    def test_invalid_batch_is_rejected(self):
        for value in [0,-1,1.5,True]:
            with self.assertRaises(ValueError):self.plan(value)


if __name__=='__main__':unittest.main()
