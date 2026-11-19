import pytest

from testing.ascend.gemm_features._support import Options


def pytest_addoption(parser):
    group = parser.getgroup("gemm feature acceptance")
    group.addoption("--gemm-mode", choices=["hardware", "compile", "simulator"], default="hardware")
    group.addoption("--gemm-device", type=int, default=0)
    group.addoption("--gemm-failure-dir", default=None)


@pytest.fixture
def gemm_options(request):
    return Options(
        request.config.getoption("--gemm-mode"), request.config.getoption("--gemm-device"), request.config.getoption("--gemm-failure-dir")
    )
