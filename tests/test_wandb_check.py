"""The key check setup.sh runs, so a bad key is caught at the prompt rather than becoming a silently offline run."""

from __future__ import annotations

from wandb.errors import AuthenticationError

from pokered import wandb_check


def test_key_works_when_wandb_accepts_it():
    assert wandb_check.key_works("good", login=lambda **_: True) is True


def test_key_fails_when_wandb_refuses_it():
    def refuse(**_: object) -> bool:
        raise AuthenticationError("API key must be 40 characters long")

    assert wandb_check.key_works("bad", login=refuse) is False


def test_key_is_verified_with_the_server_without_any_prompt():
    seen: dict[str, object] = {}

    def record(**options: object) -> bool:
        seen.update(options)
        return True

    wandb_check.key_works("good", login=record)
    assert seen == {"key": "good", "relogin": True, "verify": True, "prompt": False}


def test_main_reports_a_missing_key(monkeypatch, capsys):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert wandb_check.main() == 1
    assert "WANDB_API_KEY is not set" in capsys.readouterr().err


def test_main_exits_zero_for_a_good_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "good")
    monkeypatch.setattr(wandb_check, "key_works", lambda key: key == "good")
    assert wandb_check.main() == 0


def test_main_exits_one_for_a_bad_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "bad")
    monkeypatch.setattr(wandb_check, "key_works", lambda key: key == "good")
    assert wandb_check.main() == 1


def test_a_refusal_never_prints_the_key(monkeypatch, capsys):
    def refuse(**_: object) -> bool:
        raise AuthenticationError("refused")

    monkeypatch.setenv("WANDB_API_KEY", "secret-key")
    monkeypatch.setattr(wandb_check.wandb, "login", refuse)
    assert wandb_check.main() == 1
    assert "secret-key" not in capsys.readouterr().err
