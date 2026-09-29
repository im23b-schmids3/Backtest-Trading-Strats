from __future__ import annotations

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_candidate_tape as candidate_tape


def test_candidate_tape_atomic_write_is_readable_and_hash_bound(tmp_path):
    metadata = {
        "tape_version": candidate_tape.TAPE_VERSION,
        "bbo_path_complete": True,
        "source_sha256": "source-sha",
        "semantic_sha256": "semantic-sha",
        "event_count": 0,
        "feature_names": list(candidate_tape.FEATURE_NAMES),
    }
    tape = candidate_tape.CandidateTape(
        metadata=metadata,
        candidates=(),
        events=np.zeros(0, dtype=candidate_tape.EVENT_DTYPE),
    )
    path = tmp_path / "candidate-tape.npz"

    candidate_tape.write_tape(path, tape)

    assert path.is_file()
    loaded = candidate_tape.load_tape(
        path, source_sha256="source-sha", semantic_sha256="semantic-sha",
    )
    assert loaded.metadata["source_sha256"] == "source-sha"
    assert len(loaded.candidates) == 0
    assert len(loaded.events) == 0
