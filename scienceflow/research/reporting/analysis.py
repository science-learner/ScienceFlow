"""Best-effort scholarly retrieval and evidence-bounded report synthesis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import urllib.parse
import urllib.request
from dataclasses import fields, replace
from html import unescape
from pathlib import Path
from typing import Any
from uuid import uuid4

from inquirycraft.adapters import AskLLMAdapter
from inquirycraft.events import JsonlEventSink
from inquirycraft.llm import CompletionRequest, LLMClient
from inquirycraft.memory import Message
from inquirycraft.runtime import AgentRuntime, RuntimeOptions

from scienceflow.foundation.config.llm.llm_factory import build_stage_llm
from scienceflow.foundation.config.llm.llm_http import aclose_llm_clients
from scienceflow.foundation.config.schema.settings import StageConfig
from scienceflow.runtime.observability.telemetry.agent.audit import (
    ScienceFlowProviderAudit,
)

from .models import ReportDocument

_MAX_REFERENCES = 6
_TEXT_LIMIT = 4000


def _clean_text(value: object, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _search_query(report: ReportDocument) -> str:
    name = re.sub(r"[-_]", " ", report.task_name)
    if not re.fullmatch(r"(?:run|task)[\s-]*\d+", name.strip(), flags=re.IGNORECASE):
        return _clean_text(name, 160)
    description = re.sub(r"[`#*_[\]{}()]", " ", report.task_description)
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]+", description)
    return _clean_text(" ".join(words[:10]), 160)


def _query_terms(value: str) -> set[str]:
    stop = {
        "about",
        "inside",
        "maximize",
        "minimize",
        "must",
        "objective",
        "problem",
        "the",
        "this",
        "using",
        "with",
    }
    terms = set()
    for term in re.findall(r"[a-z][a-z0-9]+", value.casefold()):
        if len(term) < 4 or term in stop:
            continue
        terms.add(term[:-1] if term.endswith("s") and len(term) > 5 else term)
    return terms


def search_related_works(
    report: ReportDocument,
    *,
    query: str = "",
    timeout: float = 8.0,
) -> list[dict[str, Any]]:
    """Retrieve a small public metadata set; network failure is non-fatal."""

    query = _clean_text(query, 280) or _search_query(report)
    if not query:
        return []
    params = urllib.parse.urlencode(
        {
            "query.bibliographic": query,
            "rows": _MAX_REFERENCES,
            "select": (
                "DOI,title,author,abstract,published,container-title,URL,"
                "is-referenced-by-count,score,type"
            ),
        }
    )
    request = urllib.request.Request(
        "https://api.crossref.org/works?" + params,
        headers={
            "Accept": "application/json",
            "User-Agent": "ScienceFlow/0.1 (research-report metadata retrieval)",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, TypeError):
        return []
    items = payload.get("message", {}).get("items", [])
    result = []
    query_terms = _query_terms(query)
    for item in items:
        if not isinstance(item, dict):
            continue
        titles = item.get("title") or []
        title = _clean_text(titles[0] if titles else "", 500)
        if not title:
            continue
        overlap = query_terms & _query_terms(title)
        if query_terms and len(overlap) < min(2, len(query_terms)):
            continue
        authors = []
        for author in item.get("author") or []:
            if not isinstance(author, dict):
                continue
            name = _clean_text(
                " ".join(
                    (str(author.get("given") or ""), str(author.get("family") or ""))
                ),
                100,
            )
            if name:
                authors.append(name)
        parts = item.get("published", {}).get("date-parts", [])
        try:
            year = int(parts[0][0])
        except (IndexError, TypeError, ValueError):
            year = None
        doi = _clean_text(item.get("DOI"), 240)
        if doi and not re.fullmatch(r"10\.\d{4,9}/\S+", doi, flags=re.IGNORECASE):
            doi = ""
        url = "https://doi.org/" + urllib.parse.quote(doi, safe="/:;()") if doi else ""
        try:
            citations = max(0, int(item.get("is-referenced-by-count") or 0))
        except (TypeError, ValueError):
            citations = 0
        try:
            score = float(item.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        result.append(
            {
                "id": f"R{len(result) + 1}",
                "title": title,
                "authors": authors[:6],
                "year": year,
                "venue": _clean_text((item.get("container-title") or [""])[0], 240),
                "doi": doi,
                "url": url,
                "citations": citations,
                "retrieval_score": score if math.isfinite(score) else 0.0,
                "type": _clean_text(item.get("type"), 80),
                "abstract": _clean_text(
                    unescape(re.sub(r"<[^>]+>", " ", str(item.get("abstract") or ""))),
                    1800,
                ),
            }
        )
    return result


def _task_payload(record: dict[str, Any], report: ReportDocument) -> dict[str, Any]:
    payload = record.get("manifest_payload")
    if not isinstance(payload, dict):
        return {}
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if str(task.get("run_id") or "") == report.run_id:
            return task
        if str(task.get("exp_id") or "") == report.task_name:
            return task
    return tasks[0] if tasks and isinstance(tasks[0], dict) else {}


def _feedback_stage(
    record: dict[str, Any], report: ReportDocument
) -> StageConfig | None:
    task = _task_payload(record, report)
    agent = task.get("agent") if isinstance(task.get("agent"), dict) else {}
    raw = agent.get("feedback") or agent.get("code")
    if not isinstance(raw, dict):
        return None
    stage = StageConfig()
    known = {field.name for field in fields(StageConfig)}
    for key, value in raw.items():
        if key in known:
            setattr(stage, key, value)
    return stage if stage.model_aliases or stage.api_keys or stage.api_key else None


def _evidence_payload(report: ReportDocument) -> dict[str, Any]:
    worker_best: dict[str, float] = {}
    for point in report.points:
        try:
            metric = float(point.get("metric"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(metric) or point.get("valid") is False:
            continue
        worker = str(point.get("worker_id") or "worker")
        current = worker_best.get(worker)
        if current is None or (
            metric < current if report.lower_is_better else metric > current
        ):
            worker_best[worker] = metric
    artifact: dict[str, Any] = {}
    if report.artifact_preview.get("kind") == "circle_packing":
        circles = report.artifact_preview.get("circles") or []
        radii = [float(circle[2]) for circle in circles]
        artifact = {
            "kind": "circle_packing",
            "circle_count": len(circles),
            "radii_sum": sum(radii),
            "minimum_radius": min(radii) if radii else None,
            "maximum_radius": max(radii) if radii else None,
            "circles": circles[:100],
        }
    return {
        "status": report.status,
        "metric": report.metric_name,
        "lower_is_better": report.lower_is_better,
        "best": report.best,
        "evaluated": report.evaluated,
        "valid": report.valid,
        "worker_best": worker_best,
        "finalists": [
            {
                "candidate": row.get("candidate_id"),
                "source": row.get("source"),
                "metric": row.get("metric"),
                "valid": row.get("valid"),
            }
            for row in report.finalists
        ],
        "estra": report.estra,
        "eec": report.eec,
        "artifact": artifact,
    }


def _json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Report agent did not return a JSON object")
    result = json.loads(value[start : end + 1])
    if not isinstance(result, dict):
        raise TypeError("Report agent output must be an object")
    return result


def _validated_analysis(value: dict[str, Any]) -> dict[str, Any]:
    limitations = value.get("limitations")
    if not isinstance(limitations, list):
        limitations = []
    return {
        "summary": _clean_text(value.get("summary"), _TEXT_LIMIT),
        "related_work": _clean_text(value.get("related_work"), _TEXT_LIMIT),
        "interpretation": _clean_text(value.get("interpretation"), _TEXT_LIMIT),
        "limitations": [
            _clean_text(item, 800) for item in limitations[:6] if _clean_text(item, 800)
        ],
    }


async def _run_report_model(
    llm: Any,
    *,
    prompt: str,
    system: str,
    timeout: float,
    temperature: float,
    run_id: str,
    trigger: str,
    audit_dir: Path | None,
) -> str:
    """Execute one report call through the same typed runtime and audit schema."""
    session_id = f"{run_id or 'report'}-report-{trigger}-{uuid4().hex[:12]}"
    workspace = audit_dir.parent if audit_dir is not None else Path.cwd()
    client = llm if isinstance(llm, LLMClient) else AskLLMAdapter(llm)
    options = RuntimeOptions(
        model=str(getattr(llm, "model", "") or "scienceflow-report"),
        workspace=workspace,
        timeout=timeout,
        temperature=temperature,
        session_id=session_id,
        run_id=run_id or "report",
        agent_id="report-agent",
        stream_llm=True,
        operation_log=(
            audit_dir / "agent_runtime_operations" / f"{session_id}.jsonl"
            if audit_dir is not None
            else None
        ),
    )
    runtime = AgentRuntime(
        llm=client,
        options=options,
        event_sink=(
            JsonlEventSink(audit_dir / "agent_runtime_events.jsonl")
            if audit_dir is not None
            else None
        ),
        provider_observer=(
            ScienceFlowProviderAudit(
                None,
                audit_dir,
                llm_override=llm,
                llm_role="feedback",
            ).observe
            if audit_dir is not None
            else None
        ),
    )
    system_message = Message.system_message(system)
    metadata = {
        "system_msgs": [system_message],
        "message_objects": True,
        "force_tool_stream": True,
        "call_kind": "ephemeral",
        "turn_kind": trigger,
        "route": trigger,
        "llm_role": "feedback",
    }
    runtime.context.metadata.update(metadata)
    result = await runtime.complete_ephemeral(
        CompletionRequest(
            messages=(Message.user_message(prompt).to_dict(),),
            model=options.model,
            temperature=temperature,
            timeout=timeout,
            metadata=metadata,
        )
    )
    return (
        str(result.content or "").strip() or str(result.reasoning_content or "").strip()
    )


async def _ask_report_agent(
    record: dict[str, Any],
    report: ReportDocument,
    *,
    audit_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    stage = _feedback_stage(record, report)
    if stage is None:
        related = await asyncio.to_thread(search_related_works, report)
        return {}, {"status": "not_configured", "model": ""}, related
    llm = build_stage_llm(stage, role="feedback")
    try:
        query_prompt = json.dumps(
            {
                "task_name": report.task_name,
                "task_description": report.task_description[:5000],
                "instruction": (
                    "Return a 2-8 word English scholarly search query that describes "
                    "the research problem, without run IDs, files, or implementation details."
                ),
            },
            ensure_ascii=False,
        )
        try:
            query_raw = await asyncio.wait_for(
                _run_report_model(
                    llm,
                    prompt=query_prompt,
                    system='Return JSON only: {"query":"scholarly search terms"}.',
                    timeout=25,
                    temperature=0.1,
                    run_id=report.run_id,
                    trigger="report_search_query",
                    audit_dir=audit_dir,
                ),
                timeout=35,
            )
            search_query = _clean_text(_json_object(query_raw).get("query"), 160)
        except Exception:  # noqa: BLE001 - deterministic retrieval fallback
            search_query = _search_query(report)
        related_works = await asyncio.to_thread(
            search_related_works,
            report,
            query=search_query,
        )
        prompt = json.dumps(
            {
                "task_name": report.task_name,
                "task_description": report.task_description[:8000],
                "run_evidence": _evidence_payload(report),
                "related_work_candidates": related_works,
                "required_output": {
                    "summary": "2-4 evidence-grounded sentences",
                    "related_work": "compare the task and results with only the supplied candidates",
                    "interpretation": "explain the observed result and research trajectory",
                    "limitations": ["specific limitation"],
                },
            },
            ensure_ascii=False,
        )
        system = (
            "You are ScienceFlow's final-report research analyst. Return one JSON object only. "
            "Use only supplied run evidence for experimental claims and only supplied metadata "
            "for related-work claims. Cite references by R1, R2, and so on. Never invent values, "
            "papers, methods, links, or causal explanations. State uncertainty when evidence is thin."
        )
        raw = await asyncio.wait_for(
            _run_report_model(
                llm,
                prompt=prompt,
                system=system,
                timeout=45,
                temperature=0.2,
                run_id=report.run_id,
                trigger="report_analysis",
                audit_dir=audit_dir,
            ),
            timeout=55,
        )
        analysis = _validated_analysis(_json_object(raw))
        model = ",".join(str(value) for value in stage.model_aliases) or str(
            stage.model
        )
        return (
            analysis,
            {"status": "ready", "model": model, "query": search_query},
            related_works,
        )
    finally:
        await aclose_llm_clients(llm)


def _cache_digest(report: ReportDocument) -> str:
    payload = json.dumps(
        {
            "run_id": report.run_id,
            "attempt": report.attempt,
            "status": report.status,
            "description": report.task_description,
            "evidence": _evidence_payload(report),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_cache(path: Path | None, digest: str) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("evidence_sha256") != digest:
        return None
    if not isinstance(value.get("analysis"), dict) or not isinstance(
        value.get("related_works"), list
    ):
        return None
    return value


def _write_cache(
    path: Path | None,
    *,
    digest: str,
    analysis: dict[str, Any],
    related_works: list[dict[str, Any]],
    status: dict[str, str],
) -> None:
    if path is None or status.get("status") != "ready":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_sha256": digest,
                "analysis": analysis,
                "related_works": related_works,
                "report_agent": status,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def enrich_report(
    record: dict[str, Any],
    report: ReportDocument,
    *,
    cache_path: str | Path | None = None,
) -> ReportDocument:
    """Attach retrieved literature and optional synthesis without risking the run."""

    if os.environ.get("SCIENCEFLOW_REPORT_AGENT", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return replace(report, report_agent={"status": "disabled", "model": ""})
    if not isinstance(record.get("manifest_payload"), dict):
        return replace(
            report, report_agent={"status": "local_evidence_only", "model": ""}
        )
    selected_cache = Path(cache_path) if cache_path is not None else None
    digest = _cache_digest(report)
    cached = _read_cache(selected_cache, digest)
    if cached is not None:
        status = dict(cached.get("report_agent") or {})
        status["status"] = "cached"
        return replace(
            report,
            related_works=list(cached["related_works"]),
            research_analysis=dict(cached["analysis"]),
            report_agent=status,
        )
    try:
        analysis, status, related = asyncio.run(
            _ask_report_agent(
                record,
                report,
                audit_dir=(
                    selected_cache.parent / "report_agent_runtime_audit"
                    if selected_cache is not None
                    else None
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001 - reporting must not change run outcome
        analysis = {}
        related = search_related_works(report)
        status = {"status": "degraded", "model": "", "reason": type(exc).__name__}
    _write_cache(
        selected_cache,
        digest=digest,
        analysis=analysis,
        related_works=related,
        status=status,
    )
    return replace(
        report,
        related_works=related,
        research_analysis=analysis,
        report_agent=status,
    )
