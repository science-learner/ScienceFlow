from __future__ import annotations

import pytest
from rich.console import Console

from scienceflow.interfaces.ui.research.control.notices import research_notice


class _RichHost:
    def __init__(self) -> None:
        self.content = None

    async def notice_rich(self, content) -> None:
        self.content = content


@pytest.mark.asyncio
async def test_research_notice_emphasizes_onboarding_terms_and_values():
    host = _RichHost()
    message = (
        '请补充运行配置：data=/path/to/dataset workers=2 cpu=8 '
        'gpu=cpu duration=20min 可在一行内填写'
    )

    await research_notice(host, message)

    assert host.content.plain == message
    console = Console()
    heading = host.content.get_style_at_offset(console, message.index('请补充'))
    setting = host.content.get_style_at_offset(console, message.index('workers='))
    path = host.content.get_style_at_offset(console, message.index('/path'))
    assert heading.bold
    assert setting.bold
    assert heading.color != setting.color
    assert path.color != setting.color


@pytest.mark.asyncio
async def test_research_notice_falls_back_to_plain_notice():
    messages = []

    class PlainHost:
        async def notice(self, message) -> None:
            messages.append(message)

    await research_notice(PlainHost(), 'Task 2 · /tmp/task-0002')

    assert messages == ['Task 2 · /tmp/task-0002']


@pytest.mark.asyncio
async def test_research_notice_emphasizes_models_and_launch_status():
    host = _RichHost()
    message = (
        'Research models · available: gpt-test\n'
        'Default · Code gpt-test · Feedback gpt-test · Policy auto\n'
        'Starting long research.\n'
        'Task 3 · circle-packing · CPU 0-7 · GPU CPU-only · 2 workers · started\n'
        'Workspace · /tmp/task-0003'
    )

    await research_notice(host, message)

    console = Console()
    assert host.content.get_style_at_offset(
        console, message.index('Research models')
    ).bold
    assert host.content.get_style_at_offset(console, message.index('Code')).color
    assert host.content.get_style_at_offset(
        console, message.index('started')
    ).bold
    assert host.content.get_style_at_offset(console, message.index('/tmp')).color
