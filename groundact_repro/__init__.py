"""Inference-side reproduction of GroundAct (arXiv:2610.07521) on Alpamayo checkpoints.

Modules
-------
data        PAI-AV sample lists (Normal / OOD / challenging), sample loading, actor tracks.
metrics     Open-loop metrics of Appendix G: ADE, corner distance, collision rate.
discrete    Discrete action-token decoding ("D" mode of Table 1) for Alpamayo VLMs.
backends    Unified wrapper around Alpamayo-R1 and Alpamayo 1.5 for both generation modes.
groundact   Reference implementation of the GroundAct inference logic (Sections 3.2-3.4).
"""
