"""Baseline RAG methods evaluated against FedCondGraphRAG for comparison.

Each subpackage wraps a third-party method's own code (vendored or pip-
installed, never reimplemented here) behind a thin per-client runner so it
can be evaluated on the same federated corpus partitions and question sets
as the main method. Currently: `hipporag`.
"""
