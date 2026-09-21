"""Optional TUI preparation agent; long-research execution stays in its owner."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from inspect import signature

from scienceflow.research.onboarding import OnboardingState
from scienceflow.research.onboarding.support.preparation import (
    parse_preparation,
    preparation_prompt,
)
from scienceflow.research.onboarding.support.preparation.discovery import draft_summary
from scienceflow.research.onboarding.support.preparation.registered import (
    registered_missing_prompt,
    registered_preparation_fields,
)
from scienceflow.research.onboarding.support.preparation.validation import (
    validate_evaluator,
)
from scienceflow.research.onboarding.support.task_defaults import (
    may_defer_metric_direction,
)

from .control.notices import research_notice


class ResearchPreparation:
    def __init__(self, interaction):
        self.owner = interaction
        self.task = None
        self.question = ''
        self.validation_error = ''
        self.validated_command = ''
        self.validation_samples = {}

    async def begin(self, host, answer: str = '') -> bool:
        session = self.owner.session
        if not callable(getattr(host, 'run_host_task', None)) or session is None:
            return False
        if self.task is not None and not self.task.done():
            await host.notice('Reading task materials and checking the evaluator. Use /cancel to stop preparation.')
            return True
        if host.busy:
            await host.notice('Finish or cancel the current chat request before preparing research.')
            return True
        session.state = OnboardingState.ASKING
        session.preflight_report = None
        self.owner.files = self.owner.preflight = None
        host.set_host_status('long research · preparing')
        if session.draft.registered_task:
            await research_notice(
                host,
                'Loading registered task and checking runtime settings…',
            )
        else:
            await research_notice(
                host,
                'Preparing task: reading descriptions and checking missing information…',
            )
        prepare = self._prepare_registered if session.draft.registered_task else self._prepare
        self.task = asyncio.create_task(prepare(host, session, answer))
        return True

    async def _prepare_registered(self, host, session, answer):
        """Prepare a trusted task without an LLM/tool-discovery round trip."""

        try:
            spec = session.task_package
            if spec is None:
                raise ValueError("Registered task package metadata is unavailable.")
            reserved_cpu_ids = self._reserved_cpu_ids()
            fields = registered_preparation_fields(
                spec,
                answer,
                workspace=self.owner.workspace,
                reserved_cpu_ids=reserved_cpu_ids,
            )
            session.revise(fields)
            self.owner.files = self.owner.preflight = None
            self.validation_error = ""
            self.question = registered_missing_prompt(session.draft)
            if self.question:
                session.state = OnboardingState.ASKING
                await research_notice(host, self.question)
            elif session.ready_for_preflight:
                await self._show_preflight(host, session)
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError) as exc:
            self.validation_error = str(exc)
            self.question = registered_missing_prompt(session.draft)
            await research_notice(
                host,
                f"配置未通过：{exc}\n"
                f"{self.question or '请修正后重试；/cancel 返回聊天。'}"
            )
        finally:
            if self.owner.session is session:
                self.owner._set_status(host)

    def _reserved_cpu_ids(self) -> frozenset[int]:
        managed = getattr(self.owner, "_managed", None)
        if managed is None or not hasattr(managed, "rows"):
            return frozenset()
        from scienceflow.runtime.parallel.control.resource_claims import claims_for

        reserved: set[int] = set()
        for row in managed.rows.values():
            if not row.get("alive"):
                continue
            claims = row.get("resource_claims") or claims_for(row.get("draft") or {})
            reserved.update(int(cpu_id) for cpu_id in claims.get("cpu", ()))
        return frozenset(reserved)

    async def _prepare(self, host, session, answer):
        try:
            for attempt in range(2):
                prompt = preparation_prompt(session.draft, answer, self.question, getattr(self.owner, "task_workspace", self.owner.workspace),
                                            validation_error=self.validation_error)
                if getattr(self.owner, "task_workspace", self.owner.workspace) != self.owner.workspace:
                    prompt += f"\nResolve user-supplied relative input paths against {self.owner.workspace}. Write new preparation artifacts only inside {self.owner.task_workspace}."
                managed = self.owner._managed if hasattr(self.owner, '_managed') else None
                if managed is not None and hasattr(managed, 'rows'):
                    pools = [{'workspace': r.get('control_workspace'), 'resources': r.get('resource_claims') or
                              {k: (r.get('draft') or {}).get(k) for k in ('cpu_list', 'gpu_list')}}
                             for r in managed.rows.values() if r.get('alive')]
                    prompt += f"\nOther active managed tasks reserve these pools: {pools!r}. Choose non-overlapping resources."
                options = ({'show_response': False}
                           if 'show_response' in signature(host.run_host_task).parameters else {})
                result = await host.run_host_task(prompt, **options)
                if self.owner.session is not session:
                    return
                try:
                    fields, question, summary, validation = parse_preparation(result, session.draft, getattr(self.owner, "task_workspace", self.owner.workspace))
                except ValueError as exc:
                    self.validation_error = str(exc)
                    if attempt:
                        raise
                    await host.notice('准备配置未通过校验，已将原因反馈给模型，正在自动纠正一次…')
                else:
                    self.validation_error = ''
                    break
            candidate = replace(session.draft, **fields)
            if not candidate.registered_task and candidate.artifact_command and (not question or validation):
                await host.notice('Checking evaluator with valid and invalid sample artifacts…')
                samples = validation or (self.validation_samples if candidate.artifact_command == self.validated_command else {})
                await validate_evaluator(candidate.artifact_command, samples, getattr(self.owner, "task_workspace", self.owner.workspace))
                self.validation_samples = samples
                self.validated_command = candidate.artifact_command
            session.revise(fields)
            if (
                question
                and session.ready_for_preflight
                and may_defer_metric_direction(session.draft)
            ):
                question = ''
            if not session.draft.registered_task and self.validated_command:
                # The prepared command has the JSON contract exercised above.
                session.draft.evaluator = {
                    'enabled': True, 'backend': 'artifact_command', 'task_profile': 'artifact_command',
                    'candidate': {'artifact': session.draft.artifact_path, 'artifact_kind': 'artifact'},
                    'metric': {'name': session.draft.metric_name, 'lower_is_better': session.draft.lower_is_better,
                               'type': 'exploratory', 'json_path': 'metric'},
                    'command': {'evaluator_command': session.draft.artifact_command},
                }
            self.owner.files = self.owner.preflight = None
            self.question = question
            if summary:
                await research_notice(host, summary)
            if question:
                # An unresolved scientific clarification blocks even structurally complete drafts.
                session.state = OnboardingState.ASKING
                await research_notice(host, question)
            elif session.ready_for_preflight:
                await self._show_preflight(host, session)
            else:
                await host.notice(await self.owner._decorate_prompt(session.next_prompt(), session.next_question()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.validation_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, ValueError) and 'required preparation JSON' in str(exc):
                await host.notice(result)
            await host.notice(f'Preparation could not complete: {type(exc).__name__}: {exc}\n'
                              'Provide missing information or ask to retry. /cancel returns to chat.')
        finally:
            if self.owner.session is session:
                self.owner._set_status(host)

    async def _show_preflight(self, host, session):
        summary = draft_summary(session.draft)
        preview = getattr(self.owner._managed, 'resource_preview', None)
        if callable(preview):
            allocation = preview(session.draft)
            if allocation:
                summary = '\n'.join((*summary.splitlines()[:2], allocation))
        await research_notice(host, summary)
        await self.owner._preflight(host)

    async def close(self):
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.task = None
        self.question = ''
        self.validation_error = ''
        self.validated_command = ''
        self.validation_samples = {}
