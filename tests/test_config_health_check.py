import pytest

from lazyrouter.config import HealthCheckConfig


def test_health_check_idle_after_seconds_must_be_positive():
    with pytest.raises(ValueError, match="health_check.idle_after_seconds must be > 0"):
        HealthCheckConfig(idle_after_seconds=0)


def test_health_check_mode_accepts_periodical_and_on_start():
    assert HealthCheckConfig(mode="periodical").mode == "periodical"
    assert HealthCheckConfig(mode="on-start").mode == "on-start"


def test_health_check_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        HealthCheckConfig(mode="manual")


def test_probe_models_by_provider_accepts_single_string_and_list():
    cfg = HealthCheckConfig(
        probe_models_by_provider={
            "anthropic": "claude-sonnet-4-5",
            "github-copilot": ["copilot-gpt-4o", "copilot-claude-sonnet"],
        }
    )

    assert cfg.probe_models_by_provider["anthropic"] == ["claude-sonnet-4-5"]
    assert cfg.probe_models_by_provider["github-copilot"] == [
        "copilot-gpt-4o",
        "copilot-claude-sonnet",
    ]


def test_probe_models_by_provider_rejects_non_mapping():
    with pytest.raises(ValueError, match="probe_models_by_provider"):
        HealthCheckConfig(probe_models_by_provider=["m1"])  # type: ignore[arg-type]
