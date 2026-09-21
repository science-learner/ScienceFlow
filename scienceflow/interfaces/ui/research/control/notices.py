"""Research-specific emphasis over the optional generic Rich notice port."""

from rich.text import Text


async def research_notice(host, message: str):
    """Render research notices with semantic emphasis when the host supports it."""
    render = getattr(host, 'notice_rich', None)
    if not callable(render):
        await host.notice(message)
        return
    body = Text(message, style='#DADCE7')
    body.highlight_regex(
        r'(?m)^(?:Task \d+|Proposed Task \d+|Loaded task description:|'
        r'Research models|Starting long research|请补充运行配置：)',
        'bold #E1C694',
    )
    body.highlight_regex(
        r'(?m)^(?:Loading|Preparing|Reading|Checking)\b[^\n]*',
        'bold #D8BE91',
    )
    body.highlight_regex(
        r'(?m)^(?:Default|Models|Workspace|Task|Artifact|Data|GPU|CPU|'
        r'Preflight|Manifest|Equivalent CLI)(?::| ·)',
        'bold #DADCE7',
    )
    body.highlight_regex(
        r'\b(?:Artifact|Data|GPU|CPU|Code|Feedback|Policy)(?::| |=)',
        '#8FA9C4',
    )
    body.highlight_regex(
        r'\b(?:models|feedback-models|model-policy|data|workers|cpu|gpu|duration)=',
        'bold #8FA9C4',
    )
    body.highlight_regex(r'(?<!\w)/(?:[^\s,]+/)*[^\s,]+', '#9297A3')
    body.highlight_regex(
        r'(?<!\w)/(?:long-research|research-usage|resources|resume|status|tasks|stop|cancel|models|web)\b',
        'bold #8FA9C4',
    )
    body.highlight_regex(r'\b\d+(?:-\d+)?(?:\.\d+)?\s*(?:workers?|min|分钟|个\s*(?:worker|CPU|核))\b', 'bold #8FA9C4')
    body.highlight_regex(r'\b(?:maximize|minimize)\s+\w+', 'bold #8FA9C4')
    body.highlight_regex(r'(?m)^(?:Manifest|Equivalent CLI):.*$', '#9297A3')
    body.highlight_regex(r'\b[\w_]+=ok\b', '#9297A3')
    body.highlight_regex(r'\b(?:ok|verified|started|available)\b', 'bold #8FA9C4')
    body.highlight_regex(r'\bexploratory\b', 'bold #D8BE91')
    body.highlight_regex(r'\b[\w_]+=failed\b|\bfailed\b', 'bold #E99AA3')
    body.highlight_regex(
        r'\b(?:Type run|Enter defaults|Fixed example)\b',
        'bold #8FA9C4',
    )
    await render(body)
