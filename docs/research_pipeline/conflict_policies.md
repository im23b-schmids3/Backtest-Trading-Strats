# Conflict policies

Supported policies are `FIRST_SIGNAL_WINS`, `HIGHEST_CONFIDENCE`,
`STRATEGY_PRIORITY`, `SKIP_CONFLICT`, `NET_EXPOSURE`, and
`ALLOW_INDEPENDENT`. Policy choice is explicit in the portfolio specification.
No policy silently treats two signals as independent when they share an
economic exposure group; the resulting conflict counts are persisted.
