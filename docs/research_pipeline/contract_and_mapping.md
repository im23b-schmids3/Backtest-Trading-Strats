# Contract and mapping policy

Default mappings are BTCUSDT -> MBT, ETHUSDT -> MET, PAXGUSDT -> MGC,
XAGUSDT -> SIL, SPX -> MES, and QQQ -> MNQ. BTC/ETH cross-mapping is rejected.
Proxy mappings must carry limitations and cannot be labelled native.

Duplicate-exposure groups are validated before simulation. Different contracts
may coexist only when the portfolio explicitly declares how shared exposure is
consolidated. No mapping is silently treated as native futures evidence.

The mapping list and contract registry are hashed and stored in the prop
registry. A mapping or dataset change requires a new research version or an
explicit invalidation outside this Phase D fixture.
