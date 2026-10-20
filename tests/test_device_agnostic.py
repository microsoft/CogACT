"""Device selection tests. The mocked tests need no accelerator; the parity test needs an XPU."""

import contextlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla import device as dev  # noqa: E402
from vla.cogactvla import CogACT  # noqa: E402


@contextlib.contextmanager
def fake_host(xpu=False, cuda=False):
    with mock.patch.object(dev, "is_available", lambda t: {"xpu": xpu, "cuda": cuda}.get(t, t == "cpu")):
        yield


class DeviceModuleTest(unittest.TestCase):
    def test_detection_order(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
        with fake_host(cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")
        with fake_host():
            self.assertEqual(dev.default_device_type(), "cpu")

    def test_resolve_default(self):
        with fake_host(xpu=True):
            self.assertEqual(dev.resolve_device(), torch.device("xpu", 0))
        with fake_host(cuda=True):
            self.assertEqual(dev.resolve_device(None), torch.device("cuda", 0))
        with fake_host():
            self.assertEqual(dev.resolve_device(""), torch.device("cpu"))

    def test_explicit_device_wins(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda", 1))
            self.assertEqual(dev.resolve_device(torch.device("xpu", 1)), torch.device("xpu", 1))


class AutocastDispatchTest(unittest.TestCase):
    def call(self, device, mixed=True):
        stub = SimpleNamespace(vlm=SimpleNamespace(device=torch.device(device), enable_mixed_precision_training=mixed))
        with mock.patch("torch.autocast") as autocast:
            CogACT._autocast(stub, torch.bfloat16)
        return autocast.call_args

    def test_follows_the_model_device(self):
        for name in ("xpu:0", "cuda:0"):
            args, kwargs = self.call(name)
            self.assertEqual(args, (name.split(":")[0],))
            self.assertEqual(kwargs, {"dtype": torch.bfloat16, "enabled": True})

    def test_cpu_stays_without_autocast(self):
        self.assertFalse(self.call("cpu")[1]["enabled"])

    def test_mixed_precision_flag_is_honoured(self):
        self.assertFalse(self.call("xpu:0", mixed=False)[1]["enabled"])


@unittest.skipUnless(dev.is_available("xpu"), "needs an Intel XPU")
class XpuParityTest(unittest.TestCase):
    def test_ddim_sampler_matches_cpu(self):
        from action_model.action_model import ActionModel

        torch.manual_seed(0)
        model = ActionModel(token_size=64, model_type="DiT-S", in_channels=7, future_action_window_size=3, past_action_window_size=0)
        with torch.no_grad():  # zero-initialised layers would make any two devices agree
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.05)
        model.eval()
        model.create_ddim(ddim_step=3)
        noise = torch.randn(2, 4, 7)
        z = torch.randn(2, 1, 64)

        def sample(device):
            net = model.net.to(device)
            with torch.no_grad():
                return model.ddim_diffusion.ddim_sample_loop(
                    net.forward, noise.shape, noise.to(device), clip_denoised=False,
                    model_kwargs=dict(z=z.to(device)), progress=False, device=device, eta=0.0,
                ).cpu()

        cpu, xpu = sample("cpu"), sample("xpu:0")
        self.assertGreater(cpu.std().item(), 1e-3)
        torch.testing.assert_close(xpu, cpu, atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
