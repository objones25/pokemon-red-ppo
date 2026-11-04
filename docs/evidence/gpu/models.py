"""Nature CNN (SB3 layout) and IMPALA CNN at width multipliers, for 3x72x80 observations."""
from __future__ import annotations

import torch
import torch.nn as nn


class _Res(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b(torch.relu(self.a(torch.relu(x))))


def _impala_body(width: int, in_ch: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = []
    c0 = in_ch
    for c in (16 * width, 32 * width, 32 * width):
        layers += [nn.Conv2d(c0, c, 3, padding=1), nn.MaxPool2d(3, 2, 1), _Res(c), _Res(c)]
        c0 = c
    layers += [nn.ReLU(), nn.Flatten()]
    return nn.Sequential(*layers)


class Policy(nn.Module):
    def __init__(self, name: str, n_actions: int = 7, shape: tuple[int, int, int] = (3, 72, 80)) -> None:
        super().__init__()
        if name == "nature":
            self.body = nn.Sequential(
                nn.Conv2d(3, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
            )
            h = 512
        elif name.startswith("impala"):
            self.body = _impala_body(int(name.removeprefix("impala")))
            h = 256
        else:
            raise ValueError(name)
        with torch.no_grad():
            f = self.body(torch.zeros(1, *shape)).shape[1]
        self.fc = nn.Sequential(nn.Linear(f, h), nn.ReLU())
        self.pi = nn.Linear(h, n_actions)
        self.vf = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.fc(self.body(x))
        return self.pi(z), self.vf(z).squeeze(1)


if __name__ == "__main__":
    from torch.utils.flop_counter import FlopCounterMode

    for n in ("nature", "impala1", "impala2", "impala4"):
        m = Policy(n)
        with FlopCounterMode(m, display=False) as fc:
            m(torch.zeros(1, 3, 72, 80))
        print(n, "params", sum(p.numel() for p in m.parameters()), "fwd MFLOP", round(fc.get_total_flops() / 1e6, 1))
