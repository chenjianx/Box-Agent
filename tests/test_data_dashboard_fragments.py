"""Regression coverage for deterministic dashboard fragment assembly."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


SKILL_DIR = Path(__file__).resolve().parents[1] / "box_agent" / "skills" / "data-dashboard"
MERGER = SKILL_DIR / "scripts" / "merge_dashboard_fragments.js"
TEMPLATE = SKILL_DIR / "assets" / "template.html"
SKILL_INSTRUCTIONS = SKILL_DIR / "SKILL.md"
NODE = os.environ.get("BOX_AGENT_NODE") or shutil.which("node")


def test_dashboard_skill_routes_large_html_to_write_file_chunks() -> None:
    instructions = SKILL_INSTRUCTIONS.read_text(encoding="utf-8")

    assert "write_file" in instructions
    assert "chunk_index=0, final=false" in instructions
    assert "staged_file_write" not in instructions
    assert "expected_chunks" not in instructions
    assert "禁止把整份 HTML" in instructions
    assert "禁止用 `bash` heredoc" in instructions


def _run_merger(tmp_path: Path, contract: dict, fragments: list[dict]) -> subprocess.CompletedProcess[str]:
    if NODE is None:
        pytest.skip("Node.js is required to test the dashboard fragment merger")

    contract_path = tmp_path / "dashboard-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    fragment_paths = []
    for index, fragment in enumerate(fragments):
        fragment_path = tmp_path / f"fragment-{index}.json"
        fragment_path.write_text(json.dumps(fragment), encoding="utf-8")
        fragment_paths.append(str(fragment_path))

    return subprocess.run(
        [
            str(NODE),
            str(MERGER),
            "--template",
            str(TEMPLATE),
            "--contract",
            str(contract_path),
            "--out",
            str(tmp_path / "dashboard.html"),
            *fragment_paths,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_dashboard_fragments_merge_in_contract_order(tmp_path: Path) -> None:
    contract = {
        "title": "经营总览",
        "brand": "Northstar",
        "initialPage": "P2",
        "note": "模拟数据",
        "pages": [
            {"id": "P1", "label": "总览", "group": "概览"},
            {"id": "P2", "label": "趋势", "group": "经营"},
        ],
    }
    fragments = [
        {
            "pageId": "P2",
            "data": {"series": [1, 2]},
            "html": '<section class="page" id="page-P2"><div id="P2-chart" class="chart"></div></section>',
            "renderer": "() => { chart('P2-chart', { series: [] }); }",
            "css": "#page-P2 .chart { height: 420px; }",
        },
        {
            "pageId": "P1",
            "data": {"kpis": [{"label": "GMV", "value": 12}]},
            "html": '<section class="page" id="page-P1"><div id="P1-kpis" class="kpi-grid"></div></section>',
            "renderer": "() => { ui.kpis('P1-kpis', D.P1.kpis); }",
            "css": "#page-P1 .kpi-grid { margin-top: 8px; }",
        },
    ]

    result = _run_merger(tmp_path, contract, fragments)

    assert result.returncode == 0, result.stderr
    output = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "<title>经营总览</title>" in output
    assert "<div class=\"logo\">Northstar" in output
    assert output.index('id="page-P1"') < output.index('id="page-P2"')
    assert output.index('data-page="P1"') < output.index('data-page="P2"')
    assert "P1: {\n  \"kpis\"" in output
    assert "P2: {\n  \"series\"" in output
    assert 'showPage("P2");' in output
    assert "showPage('P1');" not in output
    assert "#page-P1 .kpi-grid" in output
    assert "#page-P2 .chart" in output

    inline_script = output.rsplit("<script>", 1)[1].rsplit("</script>", 1)[0]
    generated_script = tmp_path / "dashboard.js"
    generated_script.write_text(inline_script, encoding="utf-8")
    syntax_check = subprocess.run(
        [str(NODE), "--check", str(generated_script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax_check.returncode == 0, syntax_check.stderr


def test_dashboard_fragments_reject_shared_css(tmp_path: Path) -> None:
    contract = {
        "title": "经营总览",
        "pages": [{"id": "P1", "label": "总览"}],
    }
    fragments = [
        {
            "pageId": "P1",
            "data": {},
            "html": '<section class="page" id="page-P1"></section>',
            "renderer": "() => {}",
            "css": "body { color: red; }",
        }
    ]

    result = _run_merger(tmp_path, contract, fragments)

    assert result.returncode == 1
    assert "must start with #page-P1 and cannot change shared selectors" in result.stderr


@pytest.mark.parametrize(
    ("html", "renderer", "expected_error"),
    [
        (
            '<section class="page" id="page-P1"><div id="P1-chart"></div><div id="P1-chart"></div></section>',
            "() => {}",
            'contains duplicate id "P1-chart"',
        ),
        (
            '<section class="page" id="page-P1"></section>',
            "() => { D.P1 = {}; }",
            "must not declare shared dashboard state",
        ),
    ],
)
def test_dashboard_fragments_reject_non_isolated_page_content(
    tmp_path: Path,
    html: str,
    renderer: str,
    expected_error: str,
) -> None:
    contract = {"title": "经营总览", "pages": [{"id": "P1", "label": "总览"}]}
    fragments = [
        {
            "pageId": "P1",
            "data": {},
            "html": html,
            "renderer": renderer,
        }
    ]

    result = _run_merger(tmp_path, contract, fragments)

    assert result.returncode == 1
    assert expected_error in result.stderr
