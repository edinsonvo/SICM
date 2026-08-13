class SICMError(Exception): pass
class ValidationError(SICMError): pass
class SolverError(SICMError): pass
class ModelNotFoundError(SICMError): pass
