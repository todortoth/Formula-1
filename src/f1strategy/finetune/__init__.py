"""Supervised fine-tuning of the pre-trained temporal BERT (``f1strategy.seqbert``) on pit-stop questions.

Modules:  ``labels`` (ground truth from the Parquet data), ``scrub`` (input leak control), ``dataset`` (labelled
windows), ``model`` (backbone + new head, freeze modes), ``metrics`` / ``baselines`` (numpy only), ``train``, ``evaluate``.
"""
