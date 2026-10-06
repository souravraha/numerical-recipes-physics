"""Runnable, verification-first atlas case studies.

Each module is a small, self-contained script that drives one flagship
application in ``numerical_recipes.atlas``, prints a key physical result, and
saves a figure to ``output/``.  They run headless (matplotlib Agg backend) and
reuse the shared library -- nothing is re-implemented here.

Run one:  ``python -m examples.bh_shadow``  (with ``PYTHONPATH=src``).
Run all:  ``./examples/run_examples.sh``
"""
