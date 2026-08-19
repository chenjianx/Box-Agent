from pathlib import Path

from box_agent.tools.skill_loader import SKILL_SLOT_SENTINEL, move_skill_slot_to_end


def test_default_system_prompt_keeps_skill_metadata_slot_at_tail() -> None:
    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "box_agent"
        / "config"
        / "system_prompt.md"
    )
    prompt = prompt_path.read_text(encoding="utf-8")

    assert prompt.count("{SKILLS_METADATA}") == 1
    assert prompt.index("{SANDBOX_INFO}") < prompt.index("{SKILLS_METADATA}")
    assert prompt.index("</output_constraints>") < prompt.index("{SKILLS_METADATA}")
    assert prompt.index("## Attention") < prompt.index("{SKILLS_METADATA}")


def test_move_skill_slot_to_end_preserves_prefix_and_single_slot() -> None:
    prompt = f"prefix\n\n{SKILL_SLOT_SENTINEL}\n\nsuffix\n"

    relocated = move_skill_slot_to_end(prompt)

    assert relocated.count(SKILL_SLOT_SENTINEL) == 1
    assert relocated.endswith(SKILL_SLOT_SENTINEL)
    assert relocated.index("prefix") < relocated.index("suffix")
    assert relocated.index("suffix") < relocated.index(SKILL_SLOT_SENTINEL)


def test_pptx_skill_makes_scaffold_and_framework_fallback_executable() -> None:
    skill_path = (
        Path(__file__).resolve().parents[1]
        / "box_agent"
        / "skills"
        / "document-skills"
        / "pptx"
        / "SKILL.md"
    )
    skill = skill_path.read_text(encoding="utf-8")

    assert (
        "Use `cd <artifact-root> && ${BOX_AGENT_NODE:-node}` on that same line"
        in skill
    )
    assert "do not split `cd` and the inspector across lines" in skill
    assert (
        "Removing numbers or rewriting a claim as qualitative prose does not verify it"
        in skill
    )
    assert (
        "the exact unavailable-data placeholder must appear in `message` or `bullets`"
        in skill
    )
