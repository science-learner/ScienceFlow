"""Optional display-only folding of the current research preparation."""

import logging

logger = logging.getLogger(__name__)


class ResearchHistory:
    def __init__(self) -> None:
        self.section = None

    async def begin(self, host) -> None:
        self.section = None
        begin = getattr(host, 'begin_transcript_section', None)
        fold = getattr(host, 'fold_transcript_section', None)
        if callable(begin) and callable(fold):
            try:
                self.section = await begin()
            except (RuntimeError, ValueError):
                logger.warning('Could not mark research configuration display', exc_info=True)

    async def fold(self, host, draft, label: str) -> None:
        section, self.section = self.section, None
        fold = getattr(host, 'fold_transcript_section', None)
        if section is None or not callable(fold):
            return
        code, feedback = ','.join(draft.code_models), ','.join(draft.feedback_models)
        models = code if code == feedback else f'Code {code} · Feedback {feedback}'
        summary = (
            f'{label} · Configuration · {draft.workers} workers · CPU {draft.cpu_list}'
            f' · GPU {draft.gpu_list} · {(draft.wall_clock_sec or 0) / 60:g} min'
            + (f' · {models}' if models else '')
        )
        try:
            await fold(section, summary)
        except (RuntimeError, ValueError):
            # A presentation failure must not turn a successful launch into a
            # failed task or encourage launching the same task again.
            logger.warning('Could not fold research configuration display', exc_info=True)
