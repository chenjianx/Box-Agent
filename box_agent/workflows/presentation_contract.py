"""Stable identifiers shared by the controlled-presentation workflow modules."""

import re
from typing import Final


WORKFLOW_KIND: Final[str] = "controlled_presentation"
PRESENTATION_DELIVERY_KEYWORDS: Final[tuple[str, ...]] = (
    "ppt",
    "pptx",
    "powerpoint",
    "演示文稿",
    "幻灯片",
    "slide deck",
    "slides",
    "presentation",
    "融资bp",
    "商业计划书",
    "pitch deck",
)
CHECKPOINT_MARKER: Final[str] = "CONTROLLED_PRESENTATION_STAGE="
RESEARCH_MODE_OPTION: Final[str] = "research_mode"
RESEARCH_ROUND_LIMIT_OPTION: Final[str] = "research_round_limit"
IMAGE_GENERATION_POLICY_OPTION: Final[str] = "image_generation_policy"
IMAGE_GENERATION_AUTO: Final[str] = "auto"
IMAGE_GENERATION_FORBIDDEN: Final[str] = "forbidden_by_user"
IMAGE_GENERATION_EXPLICIT_RETRY: Final[str] = "explicit_retry"

_IMAGE_FORBIDDEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:"
    r"(?:不用|不要|无需|不需要|禁止|别)(?:再)?(?:生成|使用|添加|放|要)?"
    r"(?:任何)?(?:图片|图像|配图|生图|图(?!表))"
    r"|(?:没|没有)(?:图片|图像|配图|图(?!表))(?:也)?(?:行|可以|没关系)"
    r"|无图(?:版|版本)?|纯文字(?:版|版本)?"
    r"|\b(?:no\s+(?:generated\s+)?images?|without\s+images?|text[- ]only)\b"
    r")",
    re.IGNORECASE,
)
_IMAGE_RETRY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:"
    r"(?=[^。；;\n]{0,80}(?:图片|图像|配图|生图))"
    r"(?=[^。；;\n]{0,80}(?:恢复|重新|再试|重试|再生成|继续生成|可以用图))"
    r"[^。；;\n]{1,120}"
    r"|(?=[^.;\n]{0,100}\b(?:image|images|image\s+service)\b)"
    r"(?=[^.;\n]{0,100}\b(?:restored|retry|try\s+again|generate\s+again)\b)"
    r"[^.;\n]{1,140}"
    r")",
    re.IGNORECASE,
)


def image_generation_policy_update(user_text: str) -> str | None:
    """Return an explicit per-turn image-policy update, if one is present."""
    normalized = " ".join(user_text.split()).strip()
    if not normalized:
        return None
    if _IMAGE_FORBIDDEN_RE.search(normalized):
        return IMAGE_GENERATION_FORBIDDEN
    if _IMAGE_RETRY_RE.search(normalized):
        return IMAGE_GENERATION_EXPLICIT_RETRY
    return None
