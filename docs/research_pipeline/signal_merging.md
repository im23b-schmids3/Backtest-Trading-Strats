# Signal merging

Adapters provide standardized signal events with source hashes, causal entry
and exit timestamps, economic exposure groups, and provenance. Events are
sorted chronologically and merged without future information. Exact duplicates,
same-direction overlaps, opposite conflicts, and exposure skips are counted.
The configured conflict policy determines which events enter the shared replay;
the raw and accepted streams are both auditable.
