from __future__ import annotations

import itertools

from .models import PortfolioCandidate, PortfolioMember, PortfolioSpec
from .utils import stable_hash


def generate_candidates(spec: PortfolioSpec, eligible_members: list[PortfolioMember], *, maximum: int = 50, maximum_strategies: int | None = None) -> list[PortfolioCandidate]:
    if len(eligible_members) < 2: return []
    members = sorted(eligible_members, key=lambda item: item.strategy_id)
    result: list[PortfolioCandidate] = []
    max_size = min(len(members), maximum_strategies or spec.budget.maximum_strategies)
    for size in range(2, max_size + 1):
        for combination in itertools.combinations(members, size):
            ids = [item.strategy_id for item in combination]
            hashes = {item.strategy_id: item.candidate_hash for item in combination}
            candidate_id = f"{spec.portfolio_id}-" + "-".join(ids)
            candidate_hash = stable_hash({"portfolio_id": spec.portfolio_id, "version": spec.version, "members": ids, "candidate_hashes": hashes})
            result.append(PortfolioCandidate(candidate_id=candidate_id, portfolio_id=spec.portfolio_id, member_strategy_ids=ids, member_candidate_hashes=hashes, candidate_hash=candidate_hash))
            if len(result) >= maximum: return result
    return result
