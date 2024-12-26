from physnetjax.models.model import EF


def test_ef():
    model = EF()
    assert model is not None
    assert model.apply is not None
    assert model.init is not None
