from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "grafana" / "dashboards"
ALERTING_DIR = PROJECT_ROOT / "grafana" / "provisioning" / "alerting"
DATASOURCE_UID = "dsp-postgres"
TEAMS = {"ml", "data-quality"}

DASHBOARDS = sorted(DASHBOARD_DIR.rglob("*.json"))
RULE_FILES = [
    path
    for path in sorted(ALERTING_DIR.glob("*.yml"))
    if "groups" in yaml.safe_load(path.read_text(encoding="utf-8"))
]


def load_dashboard(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def all_rules() -> list[tuple[dict, dict]]:
    rules = []
    for path in RULE_FILES:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        for group in content["groups"]:
            rules.extend((group, rule) for rule in group["rules"])
    return rules


def refresh_seconds(value: str) -> int:
    number, unit = re.fullmatch(r"(\d+)([smh])", value).groups()
    return int(number) * {"s": 1, "m": 60, "h": 3600}[unit]


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda path: path.name)
def test_dashboard_meets_the_brief(path: Path) -> None:
    dashboard = load_dashboard(path)

    assert dashboard["id"] is None
    assert dashboard["uid"]
    assert len(dashboard["panels"]) >= 4
    assert refresh_seconds(dashboard["refresh"]) <= 30
    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] == DATASOURCE_UID
        for target in panel["targets"]:
            assert target["datasource"]["uid"] == DATASOURCE_UID
            assert "$__time" in target["rawSql"] or "v_baseline" in (
                target["rawSql"]
            )
        steps = panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
        assert steps


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda path: path.name)
def test_dashboard_has_thresholds_on_most_panels(path: Path) -> None:
    panels = load_dashboard(path)["panels"]
    with_thresholds = [
        panel
        for panel in panels
        if len(panel["fieldConfig"]["defaults"]["thresholds"]["steps"]) >= 3
    ]
    assert len(with_thresholds) >= 4


def test_dashboard_uids_are_unique() -> None:
    uids = [load_dashboard(path)["uid"] for path in DASHBOARDS]
    assert len(uids) == len(set(uids))


def test_every_alert_rule_is_routed_and_quiet_when_idle() -> None:
    folders = {path.parent.name for path in DASHBOARDS}
    rules = all_rules()
    assert rules
    for group, rule in rules:
        assert group["folder"] in folders
        assert rule["labels"]["team"] in TEAMS
        assert rule["noDataState"] == "OK"
        assert rule["execErrState"] == "Error"
        assert rule["condition"] in {query["refId"] for query in rule["data"]}
        for query in rule["data"]:
            assert query["datasourceUid"] in {DATASOURCE_UID, "__expr__"}


def test_rule_templates_use_plain_template_variables() -> None:
    # Grafana 12.1 keeps "$$" as is in alert annotations, which breaks the
    # template ("bad character U+0024"), so templates use $labels directly.
    for path in RULE_FILES:
        assert "$$" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("folder", "team"),
    [("ml-monitoring", "ml"), ("data-quality", "data-quality")],
)
def test_each_dashboard_folder_has_two_alerts(folder: str, team: str) -> None:
    rules = [
        rule
        for group, rule in all_rules()
        if group["folder"] == folder and rule["labels"]["team"] == team
    ]
    assert len(rules) >= 2


def test_alert_rule_uids_are_unique() -> None:
    uids = [rule["uid"] for _, rule in all_rules()]
    assert len(uids) == len(set(uids))
