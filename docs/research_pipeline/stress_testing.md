# Stress testing

Stress tests run after a passing holdout and use the frozen candidate. They
cover fees, slippage, worse entries/exits, removed trades, and missing fills.
Stress results are diagnostic: they cannot change parameters or reopen
research. Execution-sensitive outcomes are classified separately from an
ordinary no-edge result.
