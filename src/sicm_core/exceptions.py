"""
Project exceptions.
"""

class SICMError(Exception):
    """Base exception."""


class ValidationError(SICMError):
    """Validation error."""


class SolverError(SICMError):
    """Solver error."""


class ModelNotFoundError(SICMError):
    """Unknown model."""
