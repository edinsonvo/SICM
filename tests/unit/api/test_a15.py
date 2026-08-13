from sicm_core import create_engine

def test_public_factory():
    engine = create_engine()
    assert engine is not None
