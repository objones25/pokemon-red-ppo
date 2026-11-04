"""Ask W&B whether a key works. setup.sh runs this so a bad key fails at the prompt: wandb.init would log offline instead.

The key comes from the environment, never argv, where any process could read it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import wandb
from wandb.errors import Error as WandbError

from pokered.negative_space import require

KEY_VARIABLE = "WANDB_API_KEY"


def key_works(key: str, login: Callable[..., bool] = wandb.login) -> bool:
    """True when W&B accepts the key. It can't tell a refused key from an unreachable server: both raise the same error."""
    require(bool(key), "there is no key to check")
    try:
        return login(key=key, relogin=True, verify=True, prompt=False)
    except WandbError as error:
        print(f"W&B did not accept this key: {error}", file=sys.stderr)
        return False


def main() -> int:  # nsp: ignore NSP001 a thin CLI; key_works checks its input and a missing key is an operating error
    key = os.environ.get(KEY_VARIABLE)
    if not key:
        print(f"{KEY_VARIABLE} is not set", file=sys.stderr)
        return 1
    return 0 if key_works(key) else 1


if __name__ == "__main__":
    sys.exit(main())
