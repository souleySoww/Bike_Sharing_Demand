from __future__ import annotations

from pathlib import Path

import yaml


WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def load(name: str) -> dict:
    content = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    # YAML reads the key "on" as True.
    content["on"] = content.pop(True, content.get("on"))
    return content


def run_commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_ci_checks_pull_requests_into_develop_and_main() -> None:
    triggers = load("ci.yml")["on"]

    assert set(triggers["pull_request"]["branches"]) == {"develop", "main"}


def test_release_runs_on_merge_into_main() -> None:
    workflow = load("release.yml")

    assert workflow["on"]["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "write"}


def test_release_waits_for_lint_and_tests() -> None:
    jobs = load("release.yml")["jobs"]
    checks = run_commands(jobs["quality"])

    assert "flake8" in checks
    assert "pytest" in checks
    assert "docker compose config" in checks
    assert jobs["release"]["needs"] == "quality"
    assert "gh release create" in run_commands(jobs["release"])


def test_manual_runs_outside_main_are_prereleases() -> None:
    script = run_commands(load("release.yml")["jobs"]["release"])

    assert "--prerelease" in script
    assert "refs/heads/main" in str(load("release.yml")["jobs"]["release"])


def test_deploy_follows_the_release_on_main_only() -> None:
    deploy = load("release.yml")["jobs"]["deploy"]

    assert deploy["needs"] == "release"
    assert deploy["if"] == "github.ref == 'refs/heads/main'"
    commands = run_commands(deploy)
    assert "git archive" in commands
    assert "deploy/deploy.sh" in commands


def test_deploy_scripts_keep_server_state() -> None:
    root = WORKFLOWS.parents[1] / "deploy"
    deploy = (root / "deploy.sh").read_text(encoding="utf-8")

    for kept in (".env", "data", "mlflow-artifacts"):
        assert f"! -name {kept}" in deploy
    assert "retrain_baseline" in deploy
    assert "set -euo pipefail" in (root / "setup_vm.sh").read_text(
        encoding="utf-8"
    )
