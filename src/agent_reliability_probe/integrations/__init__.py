"""Adapters that ground the probe on real benchmark tasks and graders.

Each integration turns a real benchmark's episodes into the same `ScenarioRun`
objects the reliability profile consumes — so the benchmark supplies the tasks
*and* the ground-truth grader, and the probe only does the k-repeat sweep and the
reliability aggregation. The probe never grades itself on real-task runs.
"""
