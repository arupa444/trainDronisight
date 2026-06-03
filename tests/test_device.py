from unittest import mock
from shared import device

def test_prefers_cuda():
    with mock.patch.object(device, "_cuda_available", return_value=True), \
         mock.patch.object(device, "_mps_available", return_value=True):
        assert device.select_device() == "cuda"

def test_falls_back_to_mps():
    with mock.patch.object(device, "_cuda_available", return_value=False), \
         mock.patch.object(device, "_mps_available", return_value=True):
        assert device.select_device() == "mps"

def test_falls_back_to_cpu():
    with mock.patch.object(device, "_cuda_available", return_value=False), \
         mock.patch.object(device, "_mps_available", return_value=False):
        assert device.select_device() == "cpu"
