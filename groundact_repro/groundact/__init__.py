"""Reference implementation of the GroundAct inference logic (Section 3 of the paper).

No GroundAct checkpoint is public, so the learned pieces (entity queries, pointer projections,
reference-feedback projection, set encoder, residual adapter, forecaster) are implemented as
modules that can be trained or loaded, and the training-free parts (reference resolution, the
interaction features, the risk latch, the flow-time schedule, the corrected Euler integration) are
exact. For an end-to-end runnable experiment on top of an Alpamayo action expert, `correction`
offers a training-free residual that mirrors the paper's teacher construction (Appendix C).

entities    prefill-time grounding readout: K addressable entity states with class/box heads.
pointer     Token-as-Reference: gated pointer next-token distribution (Eq. 1) and reference
            resolution / consumption (Eq. 2).
selection   which entities enter the correction pathway (reasoning references or the Table 4(a)
            heuristics).
forecast    entity forecasts S_i over the planning horizon.
interaction Interact(P, S_i): clearance, relative motion, validity, predicted overlap.
correction  selective residual correction of the flow field and the corrected sampler.
"""
