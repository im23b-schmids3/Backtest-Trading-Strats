# Refined interaction-level diagnostics

This is a bounded descriptive study, not a trading strategy. Interactions preserve the completed causal MBO reconstruction, interaction grouping, structural levels, prior-RTH profile, +/-4 ES-tick vicinity, 60-second timeout, features, and tiers without threshold changes or optimization.

## Emitted machine-readable results

```json
{
  "bounded_actual_es_price_examples": [
    {
      "date": "2026-07-20",
      "end_price_es": 7504.0,
      "label": "ABSORPTION_INTERACTION",
      "level": "CURRENT_RTH_LOW_SWEEP",
      "level_price_es": 7502.75,
      "termination": "VICINITY_EXIT_RESET"
    },
    {
      "date": "2026-07-24",
      "end_price_es": 7446.25,
      "label": "ABSORPTION_INTERACTION",
      "level": "CURRENT_RTH_HIGH_SWEEP",
      "level_price_es": 7448.75,
      "termination": "VICINITY_EXIT_RESET"
    },
    {
      "date": "2026-07-24",
      "end_price_es": 7439.25,
      "label": "ABSORPTION_INTERACTION",
      "level": "CURRENT_RTH_LOW_SWEEP",
      "level_price_es": 7437.25,
      "termination": "VICINITY_EXIT_RESET"
    },
    {
      "date": "2026-07-24",
      "end_price_es": 7479.5,
      "label": "ABSORPTION_INTERACTION",
      "level": "CURRENT_RTH_HIGH_SWEEP",
      "level_price_es": 7481.25,
      "termination": "VICINITY_EXIT_RESET"
    },
    {
      "date": "2026-07-27",
      "end_price_es": 7491.75,
      "label": "ABSORPTION_INTERACTION",
      "level": "CURRENT_RTH_LOW_SWEEP",
      "level_price_es": 7490.25,
      "termination": "VICINITY_EXIT_RESET"
    }
  ],
  "causal_signed_response_ticks": {
    "ABSORPTION_PLUS_REPLENISHMENT": {
      "CURRENT_RTH_HIGH_SWEEP": {
        "120s": {
          "count": 8,
          "max": 27,
          "mean": 0.375,
          "median": 8.0,
          "p25": -26,
          "p75": 12,
          "p90": 27,
          "p95": 27,
          "p99": 27,
          "trimmed_mean_1pct": 0.375
        },
        "15s": {
          "count": 8,
          "max": 15,
          "mean": 0.5,
          "median": -0.5,
          "p25": -5,
          "p75": 4,
          "p90": 15,
          "p95": 15,
          "p99": 15,
          "trimmed_mean_1pct": 0.5
        },
        "30s": {
          "count": 8,
          "max": 11,
          "mean": -0.125,
          "median": 2.0,
          "p25": -9,
          "p75": 5,
          "p90": 11,
          "p95": 11,
          "p99": 11,
          "trimmed_mean_1pct": -0.125
        },
        "5s": {
          "count": 8,
          "max": 2,
          "mean": -2,
          "median": -1.5,
          "p25": -5,
          "p75": -1,
          "p90": 2,
          "p95": 2,
          "p99": 2,
          "trimmed_mean_1pct": -2
        },
        "60s": {
          "count": 8,
          "max": 36,
          "mean": -2.625,
          "median": -1.5,
          "p25": -19,
          "p75": 3,
          "p90": 36,
          "p95": 36,
          "p99": 36,
          "trimmed_mean_1pct": -2.625
        }
      },
      "CURRENT_RTH_LOW_SWEEP": {
        "120s": {
          "count": 23,
          "max": 36,
          "mean": -4.739130434782608,
          "median": -7,
          "p25": -15,
          "p75": 8,
          "p90": 13,
          "p95": 20,
          "p99": 36,
          "trimmed_mean_1pct": -4.739130434782608
        },
        "15s": {
          "count": 23,
          "max": 17,
          "mean": -1.3478260869565217,
          "median": -3,
          "p25": -8,
          "p75": 4,
          "p90": 12,
          "p95": 13,
          "p99": 17,
          "trimmed_mean_1pct": -1.3478260869565217
        },
        "30s": {
          "count": 23,
          "max": 21,
          "mean": -2.4782608695652173,
          "median": -3,
          "p25": -9,
          "p75": 5,
          "p90": 8,
          "p95": 12,
          "p99": 21,
          "trimmed_mean_1pct": -2.4782608695652173
        },
        "5s": {
          "count": 23,
          "max": 10,
          "mean": 0.2608695652173913,
          "median": 0,
          "p25": -3,
          "p75": 2,
          "p90": 9,
          "p95": 10,
          "p99": 10,
          "trimmed_mean_1pct": 0.2608695652173913
        },
        "60s": {
          "count": 23,
          "max": 6,
          "mean": -5.782608695652174,
          "median": -5,
          "p25": -12,
          "p75": 4,
          "p90": 5,
          "p95": 5,
          "p99": 6,
          "trimmed_mean_1pct": -5.782608695652174
        }
      },
      "PRIOR_RTH_HIGH": {
        "120s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "15s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "30s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "5s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "60s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        }
      },
      "PRIOR_RTH_LOW": {
        "120s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "15s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "30s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "5s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "60s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        }
      },
      "PRIOR_RTH_POC": {
        "120s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "15s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "30s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "5s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "60s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        }
      },
      "PRIOR_RTH_VAH": {
        "120s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "15s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "30s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "5s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "60s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        }
      },
      "PRIOR_RTH_VAL": {
        "120s": {
          "count": 2,
          "max": 27,
          "mean": 4,
          "median": 4.0,
          "p25": -19,
          "p75": 27,
          "p90": 27,
          "p95": 27,
          "p99": 27,
          "trimmed_mean_1pct": 4
        },
        "15s": {
          "count": 2,
          "max": 16,
          "mean": 10.5,
          "median": 10.5,
          "p25": 5,
          "p75": 16,
          "p90": 16,
          "p95": 16,
          "p99": 16,
          "trimmed_mean_1pct": 10.5
        },
        "30s": {
          "count": 2,
          "max": 18,
          "mean": 13.5,
          "median": 13.5,
          "p25": 9,
          "p75": 18,
          "p90": 18,
          "p95": 18,
          "p99": 18,
          "trimmed_mean_1pct": 13.5
        },
        "5s": {
          "count": 2,
          "max": 4,
          "mean": 1,
          "median": 1.0,
          "p25": -2,
          "p75": 4,
          "p90": 4,
          "p95": 4,
          "p99": 4,
          "trimmed_mean_1pct": 1
        },
        "60s": {
          "count": 2,
          "max": 35,
          "mean": 17.5,
          "median": 17.5,
          "p25": 0,
          "p75": 35,
          "p90": 35,
          "p95": 35,
          "p99": 35,
          "trimmed_mean_1pct": 17.5
        }
      }
    },
    "HIGH_ABSORPTION": {
      "CURRENT_RTH_HIGH_SWEEP": {
        "120s": {
          "count": 71,
          "max": 47,
          "mean": -2.5211267605633805,
          "median": -2,
          "p25": -14,
          "p75": 9,
          "p90": 16,
          "p95": 27,
          "p99": 47,
          "trimmed_mean_1pct": -2.5211267605633805
        },
        "15s": {
          "count": 71,
          "max": 19,
          "mean": 0.5352112676056338,
          "median": 1,
          "p25": -3,
          "p75": 4,
          "p90": 6,
          "p95": 10,
          "p99": 19,
          "trimmed_mean_1pct": 0.5352112676056338
        },
        "30s": {
          "count": 71,
          "max": 20,
          "mean": -0.7605633802816901,
          "median": 0,
          "p25": -6,
          "p75": 4,
          "p90": 8,
          "p95": 12,
          "p99": 20,
          "trimmed_mean_1pct": -0.7605633802816901
        },
        "5s": {
          "count": 71,
          "max": 9,
          "mean": -0.30985915492957744,
          "median": 0,
          "p25": -2,
          "p75": 2,
          "p90": 3,
          "p95": 6,
          "p99": 9,
          "trimmed_mean_1pct": -0.30985915492957744
        },
        "60s": {
          "count": 71,
          "max": 36,
          "mean": -2.704225352112676,
          "median": -2,
          "p25": -11,
          "p75": 5,
          "p90": 13,
          "p95": 24,
          "p99": 36,
          "trimmed_mean_1pct": -2.704225352112676
        }
      },
      "CURRENT_RTH_LOW_SWEEP": {
        "120s": {
          "count": 68,
          "max": 267,
          "mean": 5.323529411764706,
          "median": -1.5,
          "p25": -12,
          "p75": 13,
          "p90": 36,
          "p95": 53,
          "p99": 267,
          "trimmed_mean_1pct": 5.323529411764706
        },
        "15s": {
          "count": 68,
          "max": 18,
          "mean": 0.014705882352941176,
          "median": 0.0,
          "p25": -4,
          "p75": 4,
          "p90": 13,
          "p95": 15,
          "p99": 18,
          "trimmed_mean_1pct": 0.014705882352941176
        },
        "30s": {
          "count": 68,
          "max": 22,
          "mean": 0.014705882352941176,
          "median": 1.0,
          "p25": -6,
          "p75": 7,
          "p90": 12,
          "p95": 17,
          "p99": 22,
          "trimmed_mean_1pct": 0.014705882352941176
        },
        "5s": {
          "count": 68,
          "max": 10,
          "mean": 0.39705882352941174,
          "median": 0.0,
          "p25": -3,
          "p75": 3,
          "p90": 7,
          "p95": 8,
          "p99": 10,
          "trimmed_mean_1pct": 0.39705882352941174
        },
        "60s": {
          "count": 68,
          "max": 267,
          "mean": 3.4411764705882355,
          "median": 1.0,
          "p25": -9,
          "p75": 8,
          "p90": 22,
          "p95": 36,
          "p99": 267,
          "trimmed_mean_1pct": 3.4411764705882355
        }
      },
      "PRIOR_RTH_HIGH": {
        "120s": {
          "count": 3,
          "max": 10,
          "mean": 3,
          "median": 5,
          "p25": -6,
          "p75": 10,
          "p90": 10,
          "p95": 10,
          "p99": 10,
          "trimmed_mean_1pct": 3
        },
        "15s": {
          "count": 3,
          "max": 7,
          "mean": -0.3333333333333333,
          "median": -3,
          "p25": -5,
          "p75": 7,
          "p90": 7,
          "p95": 7,
          "p99": 7,
          "trimmed_mean_1pct": -0.3333333333333333
        },
        "30s": {
          "count": 3,
          "max": 10,
          "mean": 4,
          "median": 3,
          "p25": -1,
          "p75": 10,
          "p90": 10,
          "p95": 10,
          "p99": 10,
          "trimmed_mean_1pct": 4
        },
        "5s": {
          "count": 3,
          "max": 1,
          "mean": -0.3333333333333333,
          "median": -1,
          "p25": -1,
          "p75": 1,
          "p90": 1,
          "p95": 1,
          "p99": 1,
          "trimmed_mean_1pct": -0.3333333333333333
        },
        "60s": {
          "count": 3,
          "max": 11,
          "mean": 6,
          "median": 4,
          "p25": 3,
          "p75": 11,
          "p90": 11,
          "p95": 11,
          "p99": 11,
          "trimmed_mean_1pct": 6
        }
      },
      "PRIOR_RTH_LOW": {
        "120s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "15s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "30s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "5s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        },
        "60s": {
          "count": 0,
          "max": null,
          "mean": null,
          "median": null,
          "p25": null,
          "p75": null,
          "p90": null,
          "p95": null,
          "p99": null,
          "trimmed_mean_1pct": null
        }
      },
      "PRIOR_RTH_POC": {
        "120s": {
          "count": 5,
          "max": 13,
          "mean": -7.2,
          "median": 2,
          "p25": -26,
          "p75": 4,
          "p90": 13,
          "p95": 13,
          "p99": 13,
          "trimmed_mean_1pct": -7.2
        },
        "15s": {
          "count": 5,
          "max": 4,
          "mean": -2.4,
          "median": 0,
          "p25": -6,
          "p75": 2,
          "p90": 4,
          "p95": 4,
          "p99": 4,
          "trimmed_mean_1pct": -2.4
        },
        "30s": {
          "count": 5,
          "max": 6,
          "mean": 1.8,
          "median": 3,
          "p25": 2,
          "p75": 4,
          "p90": 6,
          "p95": 6,
          "p99": 6,
          "trimmed_mean_1pct": 1.8
        },
        "5s": {
          "count": 5,
          "max": 7,
          "mean": -0.8,
          "median": 1,
          "p25": -6,
          "p75": 5,
          "p90": 7,
          "p95": 7,
          "p99": 7,
          "trimmed_mean_1pct": -0.8
        },
        "60s": {
          "count": 5,
          "max": 8,
          "mean": -1.4,
          "median": 1,
          "p25": -2,
          "p75": 4,
          "p90": 8,
          "p95": 8,
          "p99": 8,
          "trimmed_mean_1pct": -1.4
        }
      },
      "PRIOR_RTH_VAH": {
        "120s": {
          "count": 6,
          "max": 5,
          "mean": -2.5,
          "median": -3.0,
          "p25": -6,
          "p75": 1,
          "p90": 5,
          "p95": 5,
          "p99": 5,
          "trimmed_mean_1pct": -2.5
        },
        "15s": {
          "count": 6,
          "max": 4,
          "mean": 1,
          "median": 0.5,
          "p25": 0,
          "p75": 2,
          "p90": 4,
          "p95": 4,
          "p99": 4,
          "trimmed_mean_1pct": 1
        },
        "30s": {
          "count": 6,
          "max": 3,
          "mean": -1,
          "median": -1.0,
          "p25": -3,
          "p75": -1,
          "p90": 3,
          "p95": 3,
          "p99": 3,
          "trimmed_mean_1pct": -1
        },
        "5s": {
          "count": 6,
          "max": 1,
          "mean": 0.16666666666666666,
          "median": 0.5,
          "p25": -1,
          "p75": 1,
          "p90": 1,
          "p95": 1,
          "p99": 1,
          "trimmed_mean_1pct": 0.16666666666666666
        },
        "60s": {
          "count": 6,
          "max": 1,
          "mean": -2,
          "median": -2.0,
          "p25": -3,
          "p75": -2,
          "p90": 1,
          "p95": 1,
          "p99": 1,
          "trimmed_mean_1pct": -2
        }
      },
      "PRIOR_RTH_VAL": {
        "120s": {
          "count": 2,
          "max": 27,
          "mean": 4,
          "median": 4.0,
          "p25": -19,
          "p75": 27,
          "p90": 27,
          "p95": 27,
          "p99": 27,
          "trimmed_mean_1pct": 4
        },
        "15s": {
          "count": 2,
          "max": 16,
          "mean": 10.5,
          "median": 10.5,
          "p25": 5,
          "p75": 16,
          "p90": 16,
          "p95": 16,
          "p99": 16,
          "trimmed_mean_1pct": 10.5
        },
        "30s": {
          "count": 2,
          "max": 18,
          "mean": 13.5,
          "median": 13.5,
          "p25": 9,
          "p75": 18,
          "p90": 18,
          "p95": 18,
          "p99": 18,
          "trimmed_mean_1pct": 13.5
        },
        "5s": {
          "count": 2,
          "max": 4,
          "mean": 1,
          "median": 1.0,
          "p25": -2,
          "p75": 4,
          "p90": 4,
          "p95": 4,
          "p99": 4,
          "trimmed_mean_1pct": 1
        },
        "60s": {
          "count": 2,
          "max": 35,
          "mean": 17.5,
          "median": 17.5,
          "p25": 0,
          "p75": 35,
          "p90": 35,
          "p95": 35,
          "p99": 35,
          "trimmed_mean_1pct": 17.5
        }
      }
    },
    "RAW_INTERACTION": {
      "CURRENT_RTH_HIGH_SWEEP": {
        "120s": {
          "count": 744,
          "max": 54,
          "mean": -0.46236559139784944,
          "median": -1.0,
          "p25": -7,
          "p75": 8,
          "p90": 18,
          "p95": 25,
          "p99": 44,
          "trimmed_mean_1pct": -0.3273972602739726
        },
        "15s": {
          "count": 744,
          "max": 28,
          "mean": -0.15456989247311828,
          "median": 0.0,
          "p25": -3,
          "p75": 2,
          "p90": 6,
          "p95": 10,
          "p99": 15,
          "trimmed_mean_1pct": -0.16027397260273973
        },
        "30s": {
          "count": 744,
          "max": 32,
          "mean": -0.24193548387096775,
          "median": 0.0,
          "p25": -4,
          "p75": 3,
          "p90": 9,
          "p95": 12,
          "p99": 21,
          "trimmed_mean_1pct": -0.19041095890410958
        },
        "5s": {
          "count": 744,
          "max": 20,
          "mean": -0.1881720430107527,
          "median": 0.0,
          "p25": -2,
          "p75": 1,
          "p90": 3,
          "p95": 5,
          "p99": 9,
          "trimmed_mean_1pct": -0.20273972602739726
        },
        "60s": {
          "count": 744,
          "max": 43,
          "mean": 0.04032258064516129,
          "median": 0.0,
          "p25": -5,
          "p75": 6,
          "p90": 12,
          "p95": 20,
          "p99": 33,
          "trimmed_mean_1pct": 0.03424657534246575
        }
      },
      "CURRENT_RTH_LOW_SWEEP": {
        "120s": {
          "count": 716,
          "max": 267,
          "mean": 2.9259776536312847,
          "median": 0.0,
          "p25": -11,
          "p75": 13,
          "p90": 24,
          "p95": 37,
          "p99": 58,
          "trimmed_mean_1pct": 1.1780626780626782
        },
        "15s": {
          "count": 716,
          "max": 24,
          "mean": -0.12569832402234637,
          "median": 0.0,
          "p25": -4,
          "p75": 4,
          "p90": 8,
          "p95": 11,
          "p99": 18,
          "trimmed_mean_1pct": -0.12108262108262108
        },
        "30s": {
          "count": 716,
          "max": 29,
          "mean": -0.3058659217877095,
          "median": 0.0,
          "p25": -7,
          "p75": 6,
          "p90": 12,
          "p95": 16,
          "p99": 21,
          "trimmed_mean_1pct": -0.2706552706552707
        },
        "5s": {
          "count": 716,
          "max": 16,
          "mean": -0.13268156424581007,
          "median": 0.0,
          "p25": -3,
          "p75": 2,
          "p90": 5,
          "p95": 7,
          "p99": 11,
          "trimmed_mean_1pct": -0.14957264957264957
        },
        "60s": {
          "count": 716,
          "max": 267,
          "mean": 0.9329608938547486,
          "median": 0.0,
          "p25": -9,
          "p75": 8,
          "p90": 18,
          "p95": 24,
          "p99": 38,
          "trimmed_mean_1pct": -0.07692307692307693
        }
      },
      "PRIOR_RTH_HIGH": {
        "120s": {
          "count": 320,
          "max": 22,
          "mean": -3.103125,
          "median": -1.0,
          "p25": -8,
          "p75": 4,
          "p90": 11,
          "p95": 13,
          "p99": 21,
          "trimmed_mean_1pct": -2.9076433121019107
        },
        "15s": {
          "count": 320,
          "max": 15,
          "mean": -0.984375,
          "median": 0.0,
          "p25": -3,
          "p75": 2,
          "p90": 4,
          "p95": 6,
          "p99": 11,
          "trimmed_mean_1pct": -0.9267515923566879
        },
        "30s": {
          "count": 320,
          "max": 13,
          "mean": -1.590625,
          "median": 0.0,
          "p25": -4,
          "p75": 3,
          "p90": 5,
          "p95": 8,
          "p99": 11,
          "trimmed_mean_1pct": -1.515923566878981
        },
        "5s": {
          "count": 320,
          "max": 11,
          "mean": -0.371875,
          "median": 0.0,
          "p25": -2,
          "p75": 1,
          "p90": 3,
          "p95": 5,
          "p99": 7,
          "trimmed_mean_1pct": -0.33121019108280253
        },
        "60s": {
          "count": 320,
          "max": 20,
          "mean": -2.525,
          "median": -1.0,
          "p25": -7,
          "p75": 4,
          "p90": 8,
          "p95": 10,
          "p99": 16,
          "trimmed_mean_1pct": -2.4012738853503186
        }
      },
      "PRIOR_RTH_LOW": {
        "120s": {
          "count": 248,
          "max": 125,
          "mean": 0.6451612903225806,
          "median": 1.0,
          "p25": -13,
          "p75": 8,
          "p90": 20,
          "p95": 30,
          "p99": 120,
          "trimmed_mean_1pct": 0.06147540983606557
        },
        "15s": {
          "count": 248,
          "max": 77,
          "mean": 0.29838709677419356,
          "median": 0.0,
          "p25": -4,
          "p75": 3,
          "p90": 7,
          "p95": 12,
          "p99": 19,
          "trimmed_mean_1pct": 0.06147540983606557
        },
        "30s": {
          "count": 248,
          "max": 90,
          "mean": 0.43951612903225806,
          "median": 0.0,
          "p25": -5,
          "p75": 5,
          "p90": 9,
          "p95": 13,
          "p99": 66,
          "trimmed_mean_1pct": -0.0860655737704918
        },
        "5s": {
          "count": 248,
          "max": 20,
          "mean": -0.004032258064516129,
          "median": 0.0,
          "p25": -2,
          "p75": 2,
          "p90": 4,
          "p95": 5,
          "p99": 12,
          "trimmed_mean_1pct": -0.040983606557377046
        },
        "60s": {
          "count": 248,
          "max": 95,
          "mean": 0.40725806451612906,
          "median": 0.0,
          "p25": -6,
          "p75": 6,
          "p90": 14,
          "p95": 19,
          "p99": 83,
          "trimmed_mean_1pct": -0.06147540983606557
        }
      },
      "PRIOR_RTH_POC": {
        "120s": {
          "count": 433,
          "max": 52,
          "mean": -0.648960739030023,
          "median": 2,
          "p25": -8,
          "p75": 11,
          "p90": 24,
          "p95": 32,
          "p99": 47,
          "trimmed_mean_1pct": 1.5623529411764705
        },
        "15s": {
          "count": 433,
          "max": 26,
          "mean": -0.14318706697459585,
          "median": 0,
          "p25": -4,
          "p75": 3,
          "p90": 8,
          "p95": 14,
          "p99": 21,
          "trimmed_mean_1pct": -0.16705882352941176
        },
        "30s": {
          "count": 433,
          "max": 38,
          "mean": -0.39260969976905313,
          "median": 1,
          "p25": -4,
          "p75": 5,
          "p90": 12,
          "p95": 16,
          "p99": 30,
          "trimmed_mean_1pct": 0.33647058823529413
        },
        "5s": {
          "count": 433,
          "max": 16,
          "mean": -0.19399538106235567,
          "median": 0,
          "p25": -2,
          "p75": 2,
          "p90": 5,
          "p95": 6,
          "p99": 10,
          "trimmed_mean_1pct": -0.18352941176470589
        },
        "60s": {
          "count": 433,
          "max": 43,
          "mean": -0.7967667436489607,
          "median": 1,
          "p25": -7,
          "p75": 8,
          "p90": 17,
          "p95": 24,
          "p99": 35,
          "trimmed_mean_1pct": 0.7082352941176471
        }
      },
      "PRIOR_RTH_VAH": {
        "120s": {
          "count": 316,
          "max": 59,
          "mean": -0.5506329113924051,
          "median": 0.0,
          "p25": -9,
          "p75": 7,
          "p90": 18,
          "p95": 28,
          "p99": 51,
          "trimmed_mean_1pct": -0.5451612903225806
        },
        "15s": {
          "count": 316,
          "max": 16,
          "mean": -0.8069620253164557,
          "median": 0.0,
          "p25": -3,
          "p75": 2,
          "p90": 4,
          "p95": 6,
          "p99": 13,
          "trimmed_mean_1pct": -0.6935483870967742
        },
        "30s": {
          "count": 316,
          "max": 44,
          "mean": -1.4493670886075949,
          "median": -1.0,
          "p25": -5,
          "p75": 3,
          "p90": 7,
          "p95": 10,
          "p99": 21,
          "trimmed_mean_1pct": -1.5548387096774194
        },
        "5s": {
          "count": 316,
          "max": 9,
          "mean": -0.4430379746835443,
          "median": 0.0,
          "p25": -1,
          "p75": 1,
          "p90": 2,
          "p95": 3,
          "p99": 7,
          "trimmed_mean_1pct": -0.4161290322580645
        },
        "60s": {
          "count": 316,
          "max": 71,
          "mean": -0.9683544303797469,
          "median": -1.0,
          "p25": -6,
          "p75": 4,
          "p90": 9,
          "p95": 16,
          "p99": 42,
          "trimmed_mean_1pct": -1.2129032258064516
        }
      },
      "PRIOR_RTH_VAL": {
        "120s": {
          "count": 312,
          "max": 84,
          "mean": 2.28525641025641,
          "median": 0.0,
          "p25": -9,
          "p75": 12,
          "p90": 29,
          "p95": 38,
          "p99": 63,
          "trimmed_mean_1pct": 2.150326797385621
        },
        "15s": {
          "count": 312,
          "max": 38,
          "mean": -0.016025641025641024,
          "median": 0.0,
          "p25": -5,
          "p75": 4,
          "p90": 10,
          "p95": 13,
          "p99": 22,
          "trimmed_mean_1pct": -0.08823529411764706
        },
        "30s": {
          "count": 312,
          "max": 43,
          "mean": 0.14102564102564102,
          "median": -1.0,
          "p25": -6,
          "p75": 4,
          "p90": 16,
          "p95": 20,
          "p99": 38,
          "trimmed_mean_1pct": -0.0196078431372549
        },
        "5s": {
          "count": 312,
          "max": 19,
          "mean": -0.15384615384615385,
          "median": -1.0,
          "p25": -3,
          "p75": 2,
          "p90": 5,
          "p95": 7,
          "p99": 14,
          "trimmed_mean_1pct": -0.16666666666666666
        },
        "60s": {
          "count": 312,
          "max": 56,
          "mean": 1.0384615384615385,
          "median": -1.0,
          "p25": -9,
          "p75": 9,
          "p90": 21,
          "p95": 31,
          "p99": 41,
          "trimmed_mean_1pct": 0.9444444444444444
        }
      }
    },
    "STRONG_REPLENISHMENT": {
      "CURRENT_RTH_HIGH_SWEEP": {
        "120s": {
          "count": 13,
          "max": 27,
          "mean": -0.9230769230769231,
          "median": 5,
          "p25": -7,
          "p75": 13,
          "p90": 22,
          "p95": 27,
          "p99": 27,
          "trimmed_mean_1pct": -0.9230769230769231
        },
        "15s": {
          "count": 13,
          "max": 15,
          "mean": -0.6153846153846154,
          "median": 0,
          "p25": -5,
          "p75": 4,
          "p90": 5,
          "p95": 15,
          "p99": 15,
          "trimmed_mean_1pct": -0.6153846153846154
        },
        "30s": {
          "count": 13,
          "max": 12,
          "mean": -3.076923076923077,
          "median": -3,
          "p25": -11,
          "p75": 5,
          "p90": 11,
          "p95": 12,
          "p99": 12,
          "trimmed_mean_1pct": -3.076923076923077
        },
        "5s": {
          "count": 13,
          "max": 5,
          "mean": -1.1538461538461537,
          "median": -1,
          "p25": -2,
          "p75": -1,
          "p90": 2,
          "p95": 5,
          "p99": 5,
          "trimmed_mean_1pct": -1.1538461538461537
        },
        "60s": {
          "count": 13,
          "max": 36,
          "mean": -2.6153846153846154,
          "median": -1,
          "p25": -11,
          "p75": 5,
          "p90": 11,
          "p95": 36,
          "p99": 36,
          "trimmed_mean_1pct": -2.6153846153846154
        }
      },
      "CURRENT_RTH_LOW_SWEEP": {
        "120s": {
          "count": 45,
          "max": 36,
          "mean": -2.8,
          "median": -6,
          "p25": -14,
          "p75": 9,
          "p90": 19,
          "p95": 21,
          "p99": 36,
          "trimmed_mean_1pct": -2.8
        },
        "15s": {
          "count": 45,
          "max": 17,
          "mean": -1.0666666666666667,
          "median": -3,
          "p25": -6,
          "p75": 4,
          "p90": 9,
          "p95": 12,
          "p99": 17,
          "trimmed_mean_1pct": -1.0666666666666667
        },
        "30s": {
          "count": 45,
          "max": 21,
          "mean": -1.2,
          "median": -1,
          "p25": -7,
          "p75": 7,
          "p90": 12,
          "p95": 15,
          "p99": 21,
          "trimmed_mean_1pct": -1.2
        },
        "5s": {
          "count": 45,
          "max": 10,
          "mean": -0.4888888888888889,
          "median": -1,
          "p25": -3,
          "p75": 2,
          "p90": 6,
          "p95": 9,
          "p99": 10,
          "trimmed_mean_1pct": -0.4888888888888889
        },
        "60s": {
          "count": 45,
          "max": 19,
          "mean": -2.488888888888889,
          "median": -1,
          "p25": -9,
          "p75": 5,
          "p90": 12,
          "p95": 17,
          "p99": 19,
          "trimmed_mean_1pct": -2.488888888888889
        }
      },
      "PRIOR_RTH_HIGH": {
        "120s": {
          "count": 20,
          "max": 21,
          "mean": -7.1,
          "median": -9.5,
          "p25": -14,
          "p75": 3,
          "p90": 8,
          "p95": 16,
          "p99": 21,
          "trimmed_mean_1pct": -7.1
        },
        "15s": {
          "count": 20,
          "max": 7,
          "mean": -3.15,
          "median": -3.5,
          "p25": -7,
          "p75": -1,
          "p90": 4,
          "p95": 4,
          "p99": 7,
          "trimmed_mean_1pct": -3.15
        },
        "30s": {
          "count": 20,
          "max": 8,
          "mean": -4,
          "median": -4.0,
          "p25": -9,
          "p75": 0,
          "p90": 4,
          "p95": 7,
          "p99": 8,
          "trimmed_mean_1pct": -4
        },
        "5s": {
          "count": 20,
          "max": 7,
          "mean": -0.95,
          "median": -0.5,
          "p25": -4,
          "p75": 1,
          "p90": 2,
          "p95": 4,
          "p99": 7,
          "trimmed_mean_1pct": -0.95
        },
        "60s": {
          "count": 20,
          "max": 12,
          "mean": -3.4,
          "median": -4.0,
          "p25": -9,
          "p75": 3,
          "p90": 9,
          "p95": 10,
          "p99": 12,
          "trimmed_mean_1pct": -3.4
        }
      },
      "PRIOR_RTH_LOW": {
        "120s": {
          "count": 12,
          "max": 23,
          "mean": 3.0833333333333335,
          "median": 2.5,
          "p25": -12,
          "p75": 20,
          "p90": 22,
          "p95": 23,
          "p99": 23,
          "trimmed_mean_1pct": 3.0833333333333335
        },
        "15s": {
          "count": 12,
          "max": 15,
          "mean": 0.3333333333333333,
          "median": 2.0,
          "p25": -6,
          "p75": 3,
          "p90": 13,
          "p95": 15,
          "p99": 15,
          "trimmed_mean_1pct": 0.3333333333333333
        },
        "30s": {
          "count": 12,
          "max": 14,
          "mean": 1.75,
          "median": 0.5,
          "p25": -6,
          "p75": 8,
          "p90": 14,
          "p95": 14,
          "p99": 14,
          "trimmed_mean_1pct": 1.75
        },
        "5s": {
          "count": 12,
          "max": 10,
          "mean": 1.0833333333333333,
          "median": 2.5,
          "p25": -5,
          "p75": 5,
          "p90": 7,
          "p95": 10,
          "p99": 10,
          "trimmed_mean_1pct": 1.0833333333333333
        },
        "60s": {
          "count": 12,
          "max": 23,
          "mean": 4.25,
          "median": 3.5,
          "p25": -6,
          "p75": 17,
          "p90": 20,
          "p95": 23,
          "p99": 23,
          "trimmed_mean_1pct": 4.25
        }
      },
      "PRIOR_RTH_POC": {
        "120s": {
          "count": 23,
          "max": 51,
          "mean": 5.304347826086956,
          "median": 1,
          "p25": -13,
          "p75": 24,
          "p90": 35,
          "p95": 46,
          "p99": 51,
          "trimmed_mean_1pct": 5.304347826086956
        },
        "15s": {
          "count": 23,
          "max": 22,
          "mean": 3.1739130434782608,
          "median": 2,
          "p25": -5,
          "p75": 12,
          "p90": 18,
          "p95": 21,
          "p99": 22,
          "trimmed_mean_1pct": 3.1739130434782608
        },
        "30s": {
          "count": 23,
          "max": 30,
          "mean": 3.0434782608695654,
          "median": 6,
          "p25": -9,
          "p75": 13,
          "p90": 17,
          "p95": 27,
          "p99": 30,
          "trimmed_mean_1pct": 3.0434782608695654
        },
        "5s": {
          "count": 23,
          "max": 16,
          "mean": 1.826086956521739,
          "median": 2,
          "p25": -3,
          "p75": 6,
          "p90": 9,
          "p95": 10,
          "p99": 16,
          "trimmed_mean_1pct": 1.826086956521739
        },
        "60s": {
          "count": 23,
          "max": 43,
          "mean": 1,
          "median": 3,
          "p25": -12,
          "p75": 12,
          "p90": 23,
          "p95": 30,
          "p99": 43,
          "trimmed_mean_1pct": 1
        }
      },
      "PRIOR_RTH_VAH": {
        "120s": {
          "count": 6,
          "max": 11,
          "mean": -2.6666666666666665,
          "median": 1.5,
          "p25": -11,
          "p75": 8,
          "p90": 11,
          "p95": 11,
          "p99": 11,
          "trimmed_mean_1pct": -2.6666666666666665
        },
        "15s": {
          "count": 6,
          "max": 9,
          "mean": -0.8333333333333334,
          "median": -2.0,
          "p25": -3,
          "p75": 1,
          "p90": 9,
          "p95": 9,
          "p99": 9,
          "trimmed_mean_1pct": -0.8333333333333334
        },
        "30s": {
          "count": 6,
          "max": 9,
          "mean": -2.1666666666666665,
          "median": -1.0,
          "p25": -12,
          "p75": 9,
          "p90": 9,
          "p95": 9,
          "p99": 9,
          "trimmed_mean_1pct": -2.1666666666666665
        },
        "5s": {
          "count": 6,
          "max": 3,
          "mean": 0.16666666666666666,
          "median": 0.5,
          "p25": -1,
          "p75": 1,
          "p90": 3,
          "p95": 3,
          "p99": 3,
          "trimmed_mean_1pct": 0.16666666666666666
        },
        "60s": {
          "count": 6,
          "max": 0,
          "mean": -5.833333333333333,
          "median": -4.5,
          "p25": -10,
          "p75": -4,
          "p90": 0,
          "p95": 0,
          "p99": 0,
          "trimmed_mean_1pct": -5.833333333333333
        }
      },
      "PRIOR_RTH_VAL": {
        "120s": {
          "count": 36,
          "max": 34,
          "mean": 1.4722222222222223,
          "median": 1.5,
          "p25": -10,
          "p75": 18,
          "p90": 27,
          "p95": 31,
          "p99": 34,
          "trimmed_mean_1pct": 1.4722222222222223
        },
        "15s": {
          "count": 36,
          "max": 24,
          "mean": 1.1388888888888888,
          "median": 1.0,
          "p25": -5,
          "p75": 6,
          "p90": 13,
          "p95": 22,
          "p99": 24,
          "trimmed_mean_1pct": 1.1388888888888888
        },
        "30s": {
          "count": 36,
          "max": 29,
          "mean": -0.4166666666666667,
          "median": 0.0,
          "p25": -8,
          "p75": 5,
          "p90": 18,
          "p95": 24,
          "p99": 29,
          "trimmed_mean_1pct": -0.4166666666666667
        },
        "5s": {
          "count": 36,
          "max": 19,
          "mean": -0.6111111111111112,
          "median": -1.0,
          "p25": -4,
          "p75": 3,
          "p90": 6,
          "p95": 8,
          "p99": 19,
          "trimmed_mean_1pct": -0.6111111111111112
        },
        "60s": {
          "count": 36,
          "max": 43,
          "mean": 1.0555555555555556,
          "median": -2.0,
          "p25": -13,
          "p75": 19,
          "p90": 31,
          "p95": 41,
          "p99": 43,
          "trimmed_mean_1pct": 1.0555555555555556
        }
      }
    }
  },
  "counts_by_day": {
    "2026-07-20": {
      "ABSORPTION_PLUS_REPLENISHMENT": 1,
      "HIGH_ABSORPTION": 11,
      "RAW_INTERACTION": 186,
      "STRONG_REPLENISHMENT": 2
    },
    "2026-07-21": {
      "ABSORPTION_PLUS_REPLENISHMENT": 0,
      "HIGH_ABSORPTION": 11,
      "RAW_INTERACTION": 220,
      "STRONG_REPLENISHMENT": 1
    },
    "2026-07-22": {
      "ABSORPTION_PLUS_REPLENISHMENT": 0,
      "HIGH_ABSORPTION": 16,
      "RAW_INTERACTION": 407,
      "STRONG_REPLENISHMENT": 0
    },
    "2026-07-23": {
      "ABSORPTION_PLUS_REPLENISHMENT": 0,
      "HIGH_ABSORPTION": 14,
      "RAW_INTERACTION": 93,
      "STRONG_REPLENISHMENT": 0
    },
    "2026-07-24": {
      "ABSORPTION_PLUS_REPLENISHMENT": 3,
      "HIGH_ABSORPTION": 12,
      "RAW_INTERACTION": 278,
      "STRONG_REPLENISHMENT": 8
    },
    "2026-07-27": {
      "ABSORPTION_PLUS_REPLENISHMENT": 8,
      "HIGH_ABSORPTION": 21,
      "RAW_INTERACTION": 627,
      "STRONG_REPLENISHMENT": 35
    },
    "2026-07-28": {
      "ABSORPTION_PLUS_REPLENISHMENT": 8,
      "HIGH_ABSORPTION": 16,
      "RAW_INTERACTION": 267,
      "STRONG_REPLENISHMENT": 30
    },
    "2026-07-29": {
      "ABSORPTION_PLUS_REPLENISHMENT": 1,
      "HIGH_ABSORPTION": 17,
      "RAW_INTERACTION": 341,
      "STRONG_REPLENISHMENT": 30
    },
    "2026-07-30": {
      "ABSORPTION_PLUS_REPLENISHMENT": 2,
      "HIGH_ABSORPTION": 15,
      "RAW_INTERACTION": 353,
      "STRONG_REPLENISHMENT": 7
    },
    "2026-07-31": {
      "ABSORPTION_PLUS_REPLENISHMENT": 10,
      "HIGH_ABSORPTION": 22,
      "RAW_INTERACTION": 317,
      "STRONG_REPLENISHMENT": 42
    }
  },
  "counts_by_level": {
    "CURRENT_RTH_HIGH_SWEEP": 744,
    "CURRENT_RTH_LOW_SWEEP": 716,
    "PRIOR_RTH_HIGH": 320,
    "PRIOR_RTH_LOW": 248,
    "PRIOR_RTH_POC": 433,
    "PRIOR_RTH_VAH": 316,
    "PRIOR_RTH_VAL": 312
  },
  "dbn_sha256": "CBA9630FF44DEAB139A5D66B7886197435FA5388EE16B5E999E26CF4DB8B8B7C",
  "direction_counts": {
    "ABSORPTION_PLUS_REPLENISHMENT": {
      "BUYER_ABSORPTION": 24,
      "SELLER_ABSORPTION": 9
    },
    "HIGH_ABSORPTION": {
      "BUYER_ABSORPTION": 76,
      "SELLER_ABSORPTION": 79
    }
  },
  "distributions": {
    "adds": {
      "count": 3089,
      "max": 36106,
      "mean": 2540.027193266429,
      "median": 951,
      "p25": 384,
      "p75": 3218,
      "p90": 6955,
      "p95": 10179,
      "p99": 16843,
      "trimmed_mean_1pct": 2382.890723010895
    },
    "aggressive_imbalance": {
      "count": 3089,
      "max": 2736,
      "mean": 4.755260602136614,
      "median": 1,
      "p25": -18,
      "p75": 24,
      "p90": 101,
      "p95": 232,
      "p99": 722,
      "trimmed_mean_1pct": 2.4404093760316936
    },
    "buy_aggressor_volume": {
      "count": 3089,
      "max": 5140,
      "mean": 300.1663968921981,
      "median": 66,
      "p25": 9,
      "p75": 337,
      "p90": 950,
      "p95": 1376,
      "p99": 2679,
      "trimmed_mean_1pct": 271.62958071970945
    },
    "cancel_replace_ambiguity": {
      "count": 3089,
      "max": 43974,
      "mean": 3158.213661379087,
      "median": 1186,
      "p25": 475,
      "p75": 4037,
      "p90": 8670,
      "p95": 12632,
      "p99": 20748,
      "trimmed_mean_1pct": 2965.7279630241005
    },
    "cancels": {
      "count": 3089,
      "max": 35693,
      "mean": 2538.208481709291,
      "median": 939,
      "p25": 384,
      "p75": 3245,
      "p90": 6929,
      "p95": 10206,
      "p99": 16821,
      "trimmed_mean_1pct": 2381.7065037966327
    },
    "events": {
      "count": 3089,
      "max": 91185,
      "mean": 6352.928779540304,
      "median": 2284,
      "p25": 890,
      "p75": 8074,
      "p90": 17574,
      "p95": 25888,
      "p99": 43908,
      "trimmed_mean_1pct": 5955.1191812479365
    },
    "execution_volume": {
      "count": 3089,
      "max": 19737,
      "mean": 1183.7782453868565,
      "median": 258,
      "p25": 40,
      "p75": 1344,
      "p90": 3624,
      "p95": 5360,
      "p99": 10052,
      "trimmed_mean_1pct": 1076.9947177286233
    },
    "executions": {
      "count": 3089,
      "max": 11105,
      "mean": 654.687924894788,
      "median": 151,
      "p25": 23,
      "p75": 781,
      "p90": 2002,
      "p95": 3010,
      "p99": 5235,
      "trimmed_mean_1pct": 599.4255529877847
    },
    "modifies": {
      "count": 3089,
      "max": 8281,
      "mean": 620.005179669796,
      "median": 238,
      "p25": 93,
      "p75": 789,
      "p90": 1735,
      "p95": 2451,
      "p99": 4189,
      "trimmed_mean_1pct": 582.7134367778144
    },
    "replenished_volume": {
      "count": 3089,
      "max": 3224,
      "mean": 197.3564260278407,
      "median": 44,
      "p25": 6,
      "p75": 232,
      "p90": 618,
      "p95": 909,
      "p99": 1644,
      "trimmed_mean_1pct": 180.64311654011226
    },
    "replenishment_count": {
      "count": 3089,
      "max": 2062,
      "mean": 133.153447717708,
      "median": 30,
      "p25": 4,
      "p75": 161,
      "p90": 411,
      "p95": 614,
      "p99": 1042,
      "trimmed_mean_1pct": 122.88907230108947
    },
    "sell_aggressor_volume": {
      "count": 3089,
      "max": 5500,
      "mean": 295.4111362900615,
      "median": 64,
      "p25": 8,
      "p75": 329,
      "p90": 918,
      "p95": 1354,
      "p99": 2683,
      "trimmed_mean_1pct": 267.09805216242984
    },
    "spread_median_ticks": {
      "count": 3089,
      "max": 4.0,
      "mean": 1.0029135642602784,
      "median": 1.0,
      "p25": 1.0,
      "p75": 1.0,
      "p90": 1.0,
      "p95": 1.0,
      "p99": 1.0,
      "trimmed_mean_1pct": 1.0
    },
    "spread_min_ticks": {
      "count": 3089,
      "max": 1,
      "mean": 0.3790870831984461,
      "median": 0,
      "p25": 0,
      "p75": 1,
      "p90": 1,
      "p95": 1,
      "p99": 1,
      "trimmed_mean_1pct": 0.4232419940574447
    },
    "unknown_aggressor_volume": {
      "count": 3089,
      "max": 9827,
      "mean": 588.2007122045969,
      "median": 129,
      "p25": 20,
      "p75": 669,
      "p90": 1797,
      "p95": 2669,
      "p99": 5007,
      "trimmed_mean_1pct": 535.2436447672499
    }
  },
  "label_counts": {
    "ABSORPTION_INTERACTION": 2646,
    "PROBABLE_REPLENISHMENT_INTERACTION": 440,
    "UNLABELED_INTERACTION": 3
  },
  "legacy_plus_was_union": true,
  "per_rth_practical_counts": {
    "ABSORPTION_PLUS_REPLENISHMENT": 3.3,
    "HIGH_ABSORPTION": 15.5,
    "RAW_INTERACTION": 308.9,
    "STRONG_REPLENISHMENT": 15.5
  },
  "plus_counts_by_structural_level": {
    "CURRENT_RTH_HIGH_SWEEP": 8,
    "CURRENT_RTH_LOW_SWEEP": 23,
    "PRIOR_RTH_HIGH": 0,
    "PRIOR_RTH_LOW": 0,
    "PRIOR_RTH_POC": 0,
    "PRIOR_RTH_VAH": 0,
    "PRIOR_RTH_VAL": 2
  },
  "pnl_calculated": false,
  "pre_selectivity_provenance": {
    "counts_not_current_results": {
      "ABSORPTION_PLUS_REPLENISHMENT": 3086,
      "HIGH_ABSORPTION": 2646,
      "RAW_INTERACTION": 3089
    },
    "historical_only": true,
    "source": "committed pre-selectivity summary"
  },
  "previous_rth_monday_2026_07_27": {
    "pass": true,
    "source_date": "2026-07-24"
  },
  "read_only": true,
  "response_integrity": {
    "excluded_from_descriptive_distributions": true,
    "pass": true,
    "sanity_violation_count": 0
  },
  "run_id": "run-1786169438290",
  "score_construction": {
    "freeze_order": "scores and p95 thresholds are computed before any response distribution",
    "percentile": "nearest-rank p95 over full pilot interaction distribution",
    "response_inputs": "none",
    "weights": "equal 0.20"
  },
  "score_thresholds": {
    "absorption_p95": 0.7977986403366786,
    "replenishment_p95": 0.7785691162188411
  },
  "selectivity_rule": "READY requires response integrity, HIGH and STRONG <=100/RTH, PLUS <=30/RTH, and PLUS total >=10",
  "status": "READY_FOR_SMALL_BACKTEST_DESIGN",
  "study": "CMEOrderflowAbsorption.ES_V1_PILOT",
  "subset_checks": {
    "global_pass": true,
    "pass": true,
    "per_day_pass": {
      "2026-07-20": true,
      "2026-07-21": true,
      "2026-07-22": true,
      "2026-07-23": true,
      "2026-07-24": true,
      "2026-07-27": true,
      "2026-07-28": true,
      "2026-07-29": true,
      "2026-07-30": true,
      "2026-07-31": true
    }
  },
  "tier_counts": {
    "ABSORPTION_PLUS_REPLENISHMENT": 33,
    "HIGH_ABSORPTION": 155,
    "RAW_INTERACTION": 3089,
    "STRONG_REPLENISHMENT": 155
  },
  "trading_strategy_executed": false
}
```

Root cause repaired: the report writer retained the obsolete `passive_side` example schema after causal interaction examples changed shape, raising `KeyError`; it also referenced `Counter` without importing it. The failure occurred after reconstruction and before refined artifacts were emitted.

Monday 2026-07-27 prior-RTH levels were built from Friday 2026-07-24: PASS.

No entries, stops, targets, trading strategy, or PnL were calculated.
