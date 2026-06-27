"""irsys — shared core library for the IR System project.

All heavy logic (preprocessing, representations, indexing, retrieval,
refinement, evaluation, bonus features) lives here. The FastAPI services
under ``services/`` and the Streamlit UI under ``ui/`` are thin wrappers
that import from this package, so every service can also be run/tested
standalone.
"""

__version__ = "0.1.0"
