# Interpretation and limitations

The final application population consists of Stage-1-positive candidate reviews. The 400 Stage-1-negative Gold audit records are retained for auditing but excluded from the primary Stage-2 population summaries.

Stage 2 is a 14-output multi-label task: 13 actionable issue domains plus `NO_ISSUE`. Issue labels may co-occur. `NO_ISSUE` is mutually exclusive with issue domains; a review with no output above threshold is marked unassigned.

Current Stage-2 F1 is approximately 0.7. Therefore:
- individual model predictions are uncertain;
- aggregate distributions should be interpreted as model-based estimates;
- paired-model agreement is a robustness signal, not a substitute for human verification;
- model-derived case examples are illustrative and should not be treated as proof of population-level prevalence.

The event-window diagnostic describes temporal concentration and label-share changes. It does not identify a causal effect of any external event.
