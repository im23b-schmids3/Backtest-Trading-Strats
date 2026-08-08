import pytest
from research_pipeline.cme_orderflow_absorption_v1.engine import CausalMBOBook, BookStateError

def apply(b, a, s, p, z, oid, seq=1, ts=1): return b.apply(action=a,side=s,price=p,size=z,order_id=oid,sequence=seq,ts_recv=ts)
def test_add_cancel_modify_multiple_orders_queue_restoration_best_spread():
 b=CausalMBOBook(); apply(b,'A','B',100,3,1); apply(b,'A','B',100,2,2,2); apply(b,'A','A',101,4,3,3)
 assert b.depth['B'][100]==5 and b.best_bid()==100 and b.best_ask()==101 and b.spread()==1
 apply(b,'C','B',100,1,1,4); apply(b,'M','B',99,5,2,5); apply(b,'A','B',100,1,4,6)
 assert b.depth['B'][100]==3 and b.depth['B'][99]==5
def test_partial_and_full_fills():
 b=CausalMBOBook(); apply(b,'A','A',101,5,1); assert apply(b,'F','A',101,2,1,2).executed
 assert b.orders[1].size==3; apply(b,'F','A',101,3,1,3); assert 1 not in b.orders
def test_negative_depth_and_invalid_state_fail_closed():
 b=CausalMBOBook()
 with pytest.raises(BookStateError): apply(b,'C','B',100,1,1)
 apply(b,'A','B',100,1,1,2)
 with pytest.raises(BookStateError): apply(b,'F','B',100,2,1,3)
def test_sequence_ordering_fails_closed():
 b=CausalMBOBook(); apply(b,'A','B',100,1,1,2,10)
 with pytest.raises(BookStateError): apply(b,'A','A',101,1,2,1,10)
def test_replenishment_feature_raw_sequence():
 b=CausalMBOBook(); apply(b,'A','A',101,2,1); assert apply(b,'F','A',101,1,1,2).executed; apply(b,'A','A',101,1,2,3)
 assert b.depth['A'][101]==2
