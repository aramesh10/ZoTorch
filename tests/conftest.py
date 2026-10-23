import pytest
import torch

DEVICES = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param
