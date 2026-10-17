"""生成离线人工复核页面，并导出兼容决策工具的 JSONL。"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import html
import json
import mimetypes
from pathlib import Path
from typing import Any

from parksight_vlm.assessment import (
    ParkingAssessment,
    SemanticIssue,
    audit_assessment_semantics,
)

try:
    from scripts.data.build_label_review_sheets import _resolve_image
except ImportError:  # direct execution: python scripts/data/build_review_html.py
    from build_label_review_sheets import _resolve_image


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("error review must contain a non-empty items array")
    return items


def _load_reference_annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    references: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"reference annotation must be an object: {path}:{line_number}")
        case_id = value.get("case_id")
        assessment = value.get("assessment")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"reference case_id must be a non-blank string: {path}:{line_number}")
        if case_id in references:
            raise ValueError(f"duplicate reference case_id: {case_id}")
        if not isinstance(assessment, dict):
            raise ValueError(f"reference assessment must be an object: {case_id}")
        references[case_id] = assessment
    if not references:
        raise ValueError(f"reference annotations must not be empty: {path}")
    return references


def _image_data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _checks(
    options: list[str],
    selected: list[str],
    field: str,
    labels: dict[str, str] | None = None,
) -> str:
    selected_values = set(selected)
    return "".join(
        '<label><input type="checkbox" data-field="{}" value="{}"{}> {}</label>'.format(
            _escape(field),
            _escape(option),
            " checked" if option in selected_values else "",
            _escape(
                option
                if labels is None or option not in labels
                else f"{option} — {labels[option]}"
            ),
        )
        for option in options
    )


_SEMANTIC_ISSUE_LABELS = {
    SemanticIssue.LOW_RISK_WITH_PREPARE_TO_STOP: "low 风险使用 prepare_to_stop",
    SemanticIssue.HIGH_RISK_WITHOUT_IMMEDIATE_RESPONSE: "high 风险缺少 yield 或 prepare_to_stop",
    SemanticIssue.PATH_CONFLICT_WITHOUT_IMMEDIATE_RESPONSE: "行人/车辆近路径事件缺少 yield 或 prepare_to_stop",
    SemanticIssue.NON_LOW_RISK_WITHOUT_EVENT: "非 low 风险没有事件依据",
}


def _semantic_audit_text(
    candidate: dict[str, Any],
) -> tuple[str, tuple[SemanticIssue, ...]]:
    assessment = ParkingAssessment.from_mapping(candidate)
    audit = audit_assessment_semantics(assessment)
    if audit.is_consistent:
        return "语义审计：当前候选未发现已定义告警", audit.issues
    labels = "；".join(_SEMANTIC_ISSUE_LABELS[issue] for issue in audit.issues)
    return f"语义审计告警：{labels}", audit.issues


def build_review_html(
    *,
    error_review_path: Path,
    image_root: Path,
    output_path: Path,
    reference_annotations_path: Path | None = None,
) -> dict[str, Any]:
    """Build a self-contained offline HTML review page."""
    raw_items = _load_items(error_review_path)
    references = _load_reference_annotations(reference_annotations_path)
    case_ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    cards: list[str] = []
    semantic_issue_counts: Counter[str] = Counter()
    events = [
        "vru_near_maneuver_path",
        "vehicle_near_maneuver_path",
        "fixed_obstacle_near_path",
        "narrow_passage",
        "visibility_occlusion",
        "parking_space_conflict",
    ]
    advice = [
        "maintain_observation",
        "slow_down",
        "yield",
        "prepare_to_stop",
        "change_maneuver_when_safe",
    ]
    advice_labels = {
        "maintain_observation": "保持观察：没有明确需要立即减速、让行或停车的目标",
        "slow_down": "减速：空间受限或风险可控，但应降低速度",
        "yield": "让行：行人或车辆优先，应让行",
        "prepare_to_stop": "准备停车：存在明显近场障碍或路径冲突，应做好立即停车准备",
        "change_maneuver_when_safe": "安全时改变操作：当前路径不适合继续，安全时调整动作",
    }
    risks = ["low", "medium", "high"]

    for index, item in enumerate(raw_items, start=1):
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("error review case_id must be a non-blank string")
        if case_id in case_ids:
            raise ValueError(f"duplicate error review case_id: {case_id}")
        case_ids.add(case_id)
        image_path = _resolve_image({"image_ref": item.get("image_ref")}, image_root)
        candidate = item.get("candidate_assessment")
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate assessment must be an object: {case_id}")
        semantic_audit_text, semantic_issues = _semantic_audit_text(candidate)
        semantic_issue_counts.update(issue.value for issue in semantic_issues)
        cases.append({"case_id": case_id, "candidate_assessment": candidate})
        model = item.get("model_assessment")
        model_text = (
            "model: FAIL | " + str((item.get("failure") or {}).get("category", "unknown"))
            if model is None
            else "model: {} | {}".format(
                model.get("risk_level", "unknown"),
                ",".join(model.get("events", [])) or "no_event",
            )
        )
        candidate_text = "candidate: {} | {}".format(
            candidate.get("risk_level", "unknown"),
            ",".join(candidate.get("events", [])) or "no_event",
        )
        reference = references.get(case_id)
        reference_text = ""
        if reference is not None:
            reference_text = "reference: {} | {}".format(
                reference.get("risk_level", "unknown"),
                ",".join(reference.get("events", [])) or "no_event",
            )
        risk_options = "".join(
            '<option value="{}"{}>{}</option>'.format(
                _escape(risk),
                " selected" if risk == candidate.get("risk_level") else "",
                risk,
            )
            for risk in risks
        )
        raw_output = item.get("raw_output") or ""
        cards.append(
            _CARD_TEMPLATE.format(
                index=index,
                case_id=_escape(case_id),
                priority=_escape(item.get("review_priority", "unknown")),
                priority_class=_escape(item.get("review_priority", "unknown")),
                image_uri=_escape(_image_data_uri(image_path)),
                split=_escape(item.get("split")),
                source_group=_escape(item.get("source_group_id")),
                candidate_text=_escape(candidate_text),
                reference_text=_escape(reference_text),
                model_text=_escape(model_text),
                semantic_audit=_escape(semantic_audit_text),
                risk_options=risk_options,
                event_checks=_checks(events, candidate.get("events", []), "events"),
                advice_checks=_checks(
                    advice,
                    candidate.get("driver_advice", []),
                    "advice",
                    advice_labels,
                ),
                evidence=_escape("\n".join(candidate.get("evidence", []))),
                raw_output=_escape(raw_output),
            )
        )

    case_id_set = set(case_ids)
    if references and set(references) != case_id_set:
        missing = sorted(case_id_set - set(references))
        unexpected = sorted(set(references) - case_id_set)
        raise ValueError(
            f"reference annotations must cover error review exactly; missing={missing}, unexpected={unexpected}"
        )
    weak_supervision = any(
        str(item.get("label_source", "")).startswith(
            "qwen3_vl_2b_base_weak_supervision"
        )
        for item in raw_items
    )
    review_warning = (
        "弱监督候选：以下结果由基础模型生成，当前分布可能发生模式坍缩；请逐图核对，"
        "不要批量确认，也不要在人工定稿前用于训练。"
        if weak_supervision
        else ""
    )
    document = (
        _PAGE_TEMPLATE.replace("__CARDS__", "\n".join(cards))
        .replace("__CASES__", _json_for_script(cases))
        .replace("__CASE_COUNT__", str(len(cases)))
        .replace("__REVIEW_WARNING__", _escape(review_warning))
        .replace(
            "__STORAGE_KEY__",
            f"parksight-vlm-review-{output_path.stem}",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "sample_count": len(cases),
        "embedded_image_count": len(cases),
        "reference_count": len(references),
        "semantic_issue_counts": dict(sorted(semantic_issue_counts.items())),
        "ready_for_review": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error-review", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reference-annotations",
        type=Path,
        help="可选的只读对照 annotation JSONL，例如原始 teacher 结果",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_review_html(
                error_review_path=args.error_review,
                image_root=args.image_root,
                output_path=args.output,
                reference_annotations_path=args.reference_annotations,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


_CARD_TEMPLATE = """<article class=\"card {priority_class}\" data-case-id=\"{case_id}\" data-priority=\"{priority}\" data-split=\"{split}\">
<div class=\"card-head\"><strong>{index}. {case_id}</strong><span>priority: {priority}</span></div>
<div class=\"card-body\">
<div><img class=\"scene\" src=\"{image_uri}\" alt=\"{case_id}\"></div>
<div>
<div class=\"context\">split: {split}
source_group: {source_group}
{candidate_text}
{reference_text}
{model_text}</div>
<div class=\"semantic-audit\" data-semantic-audit>{semantic_audit}</div>
<label>复核状态 <select data-field=\"status\"><option value=\"pending\">pending</option><option value=\"confirmed\">confirmed</option><option value=\"corrected\">corrected</option></select></label>
<fieldset><legend>人工确认风险等级</legend><select data-field=\"risk\">{risk_options}</select></fieldset>
<fieldset><legend>人工确认事件</legend><div class=\"checks\">{event_checks}</div></fieldset>
<fieldset><legend>人工确认驾驶建议</legend><div class=\"checks\">{advice_checks}</div></fieldset>
<p class=\"review-hint\">复核建议：请先判断 risk_level 和 events，再选择匹配的 driver_advice。若人工判断近场车辆、行人或障碍已经影响当前机动路径，请同步检查 risk_level、events、evidence 与 driver_advice 的一致性；调整任一字段后选择 corrected，并填写复核说明。</p>
<label>可见证据（每行一条）<textarea data-field=\"evidence\">{evidence}</textarea></label>
<label>复核说明<textarea data-field=\"note\" placeholder=\"corrected 必填；confirmed 可填写确认依据\"></textarea></label>
<details><summary>查看模型原始输出</summary><pre>{raw_output}</pre></details>
</div>
</div>
</article>"""


_PAGE_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ParkSight-VLM 人工复核</title>
<style>
body { margin: 0; background: #f3f4f6; color: #17202a; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
header { position: sticky; top: 0; z-index: 2; padding: 12px 18px; background: #17202a; color: white; box-shadow: 0 2px 8px #0003; }
h1 { margin: 0 0 8px; font-size: 20px; }
.review-warning { margin: 0 0 8px; padding: 8px 10px; border-left: 4px solid #c0392b; background: #fff0ee; color: #8f2419; font-size: 13px; line-height: 1.45; }
.coverage { margin: 0 0 8px; padding: 8px 10px; border-left: 4px solid #2e75b6; background: #edf6ff; color: #173b5c; font-size: 12px; line-height: 1.5; white-space: pre-wrap; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
button, select { border: 1px solid #aab4bd; border-radius: 5px; padding: 6px 9px; background: white; font: inherit; }
button { cursor: pointer; }
button.primary { background: #2e75b6; color: white; border-color: #2e75b6; }
#summary { margin-left: auto; font-size: 14px; }
main { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 14px; padding: 16px; }
.card { background: white; border: 1px solid #d5dbe0; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px #0001; }
.card.high { border: 2px solid #c0392b; }
.card-head { padding: 9px 12px; display: flex; justify-content: space-between; gap: 10px; background: #eef2f5; }
.card-head strong { overflow-wrap: anywhere; }
.card-head span { font-weight: 700; }
.card.high .card-head span { color: #b42318; }
.card-body { display: grid; grid-template-columns: minmax(230px, 42%) 1fr; gap: 12px; padding: 12px; }
.scene { width: 100%; max-height: 330px; object-fit: contain; background: #111; }
.context { font-size: 13px; color: #4d5963; margin: 4px 0 10px; white-space: pre-wrap; overflow-wrap: anywhere; }
.semantic-audit { margin: 7px 0; padding: 7px 8px; border-left: 3px solid #4b8f29; background: #f1f8ed; color: #2f5d1d; font-size: 12px; line-height: 1.45; }
.semantic-audit.warning { border-left-color: #c0392b; background: #fff0ee; color: #8f2419; }
fieldset { border: 1px solid #d5dbe0; border-radius: 5px; margin: 7px 0; padding: 7px; }
legend { font-size: 12px; color: #53606b; }
label { display: block; font-size: 13px; margin: 5px 0; }
.checks { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
.review-hint { margin: 8px 0; padding: 8px; border-left: 3px solid #d28b00; background: #fff8e1; color: #5f4500; font-size: 12px; line-height: 1.5; }
textarea { box-sizing: border-box; width: 100%; min-height: 62px; resize: vertical; font: 13px/1.35 "Segoe UI", "Microsoft YaHei", sans-serif; }
details { margin-top: 7px; font-size: 12px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 180px; overflow: auto; background: #f5f6f7; padding: 6px; }
.hidden { display: none; }
@media (max-width: 700px) { main { grid-template-columns: 1fr; padding: 8px; } .card-body { grid-template-columns: 1fr; } #summary { margin-left: 0; } }
</style>
</head>
<body>
<header>
<h1>ParkSight-VLM __CASE_COUNT__ 条候选标注人工复核</h1>
<div class="review-warning">__REVIEW_WARNING__</div>
<div id="coverage" class="coverage"></div>
<div class="toolbar">
<label>筛选优先级 <select id="priority-filter"><option value="all">全部</option><option value="high">high</option><option value="medium">medium</option><option value="low">low</option></select></label>
<label>数据集分片 <select id="split-filter"><option value="all">全部</option><option value="train">train</option><option value="validation">validation</option></select></label>
<button id="confirm-visible">将当前显示项标为 confirmed</button>
<button class="primary" id="download">下载已完成决策 JSONL</button>
<span id="draft-status">草稿自动保存于当前浏览器</span>
<span id="summary"></span>
</div>
</header>
<main id="cases">__CARDS__</main>
<script type="application/json" id="review-data">__CASES__</script>
<script>
const CASES = JSON.parse(document.getElementById("review-data").textContent);
const STORAGE_KEY = "__STORAGE_KEY__";
const EVENT_NAMES = [
  "vru_near_maneuver_path",
  "vehicle_near_maneuver_path",
  "fixed_obstacle_near_path",
  "narrow_passage",
  "visibility_occlusion",
  "parking_space_conflict"
];
const EVENT_LABELS = {
  vru_near_maneuver_path: "行人/非机动车近路径",
  vehicle_near_maneuver_path: "车辆近路径",
  fixed_obstacle_near_path: "固定障碍近路径",
  narrow_passage: "狭窄通道",
  visibility_occlusion: "视野遮挡",
  parking_space_conflict: "车位冲突"
};
const SEMANTIC_LABELS = {
  low_risk_with_prepare_to_stop: "low 风险使用 prepare_to_stop",
  high_risk_without_immediate_response: "high 风险缺少 yield 或 prepare_to_stop",
  path_conflict_without_immediate_response: "行人/车辆近路径事件缺少 yield 或 prepare_to_stop",
  non_low_risk_without_event: "非 low 风险没有事件依据"
};
function visibleCards() { return [...document.querySelectorAll(".card:not(.hidden)")]; }
function updateSummary() {
  const counts = {pending: 0, confirmed: 0, corrected: 0};
  document.querySelectorAll(".card").forEach(card => counts[card.querySelector('[data-field="status"]').value]++);
  document.getElementById("summary").textContent = `共 ${CASES.length} 条 | pending ${counts.pending} | confirmed ${counts.confirmed} | corrected ${counts.corrected}`;
}
function updateCoverageSummary() {
  const countsBySplit = {};
  document.querySelectorAll(".card").forEach(card => {
    const split = card.dataset.split || "unknown";
    if (!countsBySplit[split]) countsBySplit[split] = Object.fromEntries(EVENT_NAMES.map(event => [event, 0]));
    card.querySelectorAll('[data-field="events"]:checked').forEach(input => countsBySplit[split][input.value]++);
  });
  const lines = ["事件覆盖（当前页面选择；最终以人工定稿 JSONL 为准）"];
  Object.entries(countsBySplit).forEach(([split, counts]) => {
    const missing = EVENT_NAMES.filter(event => counts[event] < (split === "train" ? 3 : split === "validation" ? 1 : 0));
    const summary = EVENT_NAMES.map(event => `${EVENT_LABELS[event]} ${counts[event]}`).join(" | ");
    lines.push(`${split}: ${summary}`);
    if (missing.length) lines.push(`  待补齐: ${missing.map(event => EVENT_LABELS[event]).join("、")}`);
  });
  lines.push("目标：train 每类至少 3 条；validation 每类至少 1 条；calibration 从已定稿 train 中另选并逐类覆盖。处于 pending 的选择也会计入提示，不能直接用于训练。");
  document.getElementById("coverage").textContent = lines.join("\n");
}
function readDraft(card) {
  return {
    status: card.querySelector('[data-field="status"]').value,
    risk: card.querySelector('[data-field="risk"]').value,
    events: [...card.querySelectorAll('[data-field="events"]:checked')].map(input => input.value),
    advice: [...card.querySelectorAll('[data-field="advice"]:checked')].map(input => input.value),
    evidence: card.querySelector('[data-field="evidence"]').value,
    note: card.querySelector('[data-field="note"]').value
  };
}
function semanticIssues(card) {
  const risk = card.querySelector('[data-field="risk"]').value;
  const events = new Set([...card.querySelectorAll('[data-field="events"]:checked')].map(input => input.value));
  const advice = new Set([...card.querySelectorAll('[data-field="advice"]:checked')].map(input => input.value));
  const immediate = advice.has("yield") || advice.has("prepare_to_stop");
  const issues = [];
  if (risk === "low" && advice.has("prepare_to_stop")) issues.push("low_risk_with_prepare_to_stop");
  if (risk === "high" && !immediate) issues.push("high_risk_without_immediate_response");
  if ((events.has("vru_near_maneuver_path") || events.has("vehicle_near_maneuver_path")) && !immediate) {
    issues.push("path_conflict_without_immediate_response");
  }
  if (risk !== "low" && events.size === 0) issues.push("non_low_risk_without_event");
  return issues;
}
function updateSemanticAudit(card) {
  const element = card.querySelector('[data-semantic-audit]');
  const issues = semanticIssues(card);
  element.classList.toggle("warning", issues.length > 0);
  element.textContent = issues.length
    ? "语义审计告警：" + issues.map(issue => SEMANTIC_LABELS[issue]).join("；")
    : "语义审计：当前选择未发现已定义告警";
}
function saveDraft() {
  try {
    const draft = {};
    document.querySelectorAll(".card").forEach(card => { draft[card.dataset.caseId] = readDraft(card); });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(draft));
    document.getElementById("draft-status").textContent = "草稿已自动保存";
  } catch (error) {
    document.getElementById("draft-status").textContent = "浏览器未提供草稿存储";
  }
}
function restoreDraft() {
  try {
    const draft = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    document.querySelectorAll(".card").forEach(card => {
      const state = draft[card.dataset.caseId];
      if (!state) return;
      if (["pending", "confirmed", "corrected"].includes(state.status)) {
        card.querySelector('[data-field="status"]').value = state.status;
      }
      if (["low", "medium", "high"].includes(state.risk)) {
        card.querySelector('[data-field="risk"]').value = state.risk;
      }
      ["events", "advice"].forEach(field => {
        const values = new Set(Array.isArray(state[field]) ? state[field] : []);
        card.querySelectorAll(`[data-field="${field}"]`).forEach(input => { input.checked = values.has(input.value); });
      });
      if (typeof state.evidence === "string") card.querySelector('[data-field="evidence"]').value = state.evidence;
      if (typeof state.note === "string") card.querySelector('[data-field="note"]').value = state.note;
    });
    document.getElementById("draft-status").textContent = "已恢复本地草稿";
  } catch (error) {
    document.getElementById("draft-status").textContent = "无可恢复草稿";
  }
}
function applyFilters() {
  const priority = document.getElementById("priority-filter").value;
  const split = document.getElementById("split-filter").value;
  document.querySelectorAll(".card").forEach(card => card.classList.toggle(
    "hidden",
    (priority !== "all" && card.dataset.priority !== priority)
      || (split !== "all" && card.dataset.split !== split)
  ));
}
document.getElementById("priority-filter").addEventListener("change", applyFilters);
document.getElementById("split-filter").addEventListener("change", applyFilters);
document.getElementById("confirm-visible").addEventListener("click", () => {
  visibleCards().forEach(card => card.querySelector('[data-field="status"]').value = "confirmed");
  saveDraft();
  updateSummary();
});
document.querySelectorAll('[data-field="status"]').forEach(select => select.addEventListener("change", () => { updateSummary(); saveDraft(); }));
document.querySelectorAll('[data-field]:not([data-field="status"])').forEach(input => input.addEventListener("input", saveDraft));
document.querySelectorAll('[data-field]:not([data-field="status"])').forEach(input => input.addEventListener("change", saveDraft));
document.querySelectorAll('[data-field="risk"], [data-field="events"], [data-field="advice"]').forEach(input => input.addEventListener("change", () => {
  updateSemanticAudit(input.closest(".card"));
  updateCoverageSummary();
  saveDraft();
}));
function readAssessment(card) {
  return {
    schema_version: "parking_risk_v1",
    risk_level: card.querySelector('[data-field="risk"]').value,
    events: [...card.querySelectorAll('[data-field="events"]:checked')].map(input => input.value),
    evidence: card.querySelector('[data-field="evidence"]').value.split("\n").map(value => value.trim()).filter(Boolean),
    driver_advice: [...card.querySelectorAll('[data-field="advice"]:checked')].map(input => input.value)
  };
}
document.getElementById("download").addEventListener("click", () => {
  const output = [...document.querySelectorAll(".card")].filter(card => {
    const status = card.querySelector('[data-field="status"]').value;
    return status === "confirmed" || status === "corrected";
  }).map(card => ({
    case_id: card.dataset.caseId,
    review_status: card.querySelector('[data-field="status"]').value,
    human_assessment: readAssessment(card),
    review_note: card.querySelector('[data-field="note"]').value
  }));
  if (!output.length) { alert("还没有已完成的复核决策。"); return; }
  const blob = new Blob([output.map(row => JSON.stringify(row)).join("\n") + "\n"], {type: "application/x-ndjson;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "review_decisions_batch.jsonl";
  link.click();
  URL.revokeObjectURL(link.href);
});
restoreDraft();
document.querySelectorAll(".card").forEach(updateSemanticAudit);
updateSummary();
updateCoverageSummary();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main())
