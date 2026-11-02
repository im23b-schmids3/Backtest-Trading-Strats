import json

from research_pipeline.cme_orderflow_absorption_l2_v1.es_four_mechanism_construction_analysis_v1 import (
    _date_bootstrap, _episode_annotation, _holm,
)


def _event(day, mechanism, ts, ordinal, direction=1, **features):
    return {"date":day,"mechanism":mechanism,"ts_recv":ts,"source_ordinal":ordinal,
            "event_id":f"{day}:{mechanism}:{ts}:{ordinal}","direction":direction,
            "features":json.dumps(features)}


def test_episode_grouping_is_deterministic_and_outcome_independent():
    events=[
        _event("2025-03-03","M1",100,1,session=0),
        _event("2025-03-03","M1",2_000_000_100,2,session=0),
        _event("2025-03-03","M1",5_000_000_100,3,session=0),
        _event("2025-03-03","M3",10,4,boundary=100.0),
        _event("2025-03-03","M3",1_000_000_010,5,boundary=100.0),
        _event("2025-03-03","M4",20,6),
        _event("2025-03-03","M4",21,7),
    ]
    ledger,rules,mapping=_episode_annotation(events)
    assert sum(x["is_episode_representative"] for x in ledger if x["mechanism"]=="M1")==2
    assert sum(x["is_episode_representative"] for x in ledger if x["mechanism"]=="M3")==1
    assert sum(x["is_episode_representative"] for x in ledger if x["mechanism"]=="M4")==2
    assert rules["M1"]["window_seconds"]==2
    assert mapping==_episode_annotation(events)[2]


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted=_holm({"a":.01,"b":.04,"c":.03,"d":None})
    assert adjusted["a"]<=adjusted["c"]<=adjusted["b"]
    assert adjusted["d"] is None
    assert all(v is None or 0<=v<=1 for v in adjusted.values())


def test_date_cluster_bootstrap_is_deterministic():
    rows={"d1":[{"r":1.0},{"r":2.0}],"d2":[{"r":-1.0}],"d3":[]}
    a=_date_bootstrap(rows,["d1","d2","d3"],"r")
    b=_date_bootstrap(rows,["d1","d2","d3"],"r")
    assert a==b
    assert a["n_dates"]==2 and a["seed"]==20261028
