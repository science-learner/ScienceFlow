"""Task-aware completion over already-loaded board/catalogue data."""

import shlex


def task_completions(manager, text):
    command, _, fragment = text.partition(' ')
    if command in {'/stop', '/resume', '/status', '/attach', '/select', '/research-usage'}:
        if ' ' in fragment.strip():
            return []
        matches = []
        for entry in manager.entries:
            row = manager.rows.get(entry.get('run_id'))
            if row is None:
                continue
            if command == '/stop' and (not row['alive'] or row['status'] in {'completed', 'failed', 'stopped'}):
                continue
            if command == '/resume' and (row['alive'] or row['status'] == 'completed'):
                continue
            number, name = str(entry['number']), entry['name']
            if fragment.casefold() not in number.casefold() and fragment.casefold() not in name.casefold():
                continue
            matches.append((f'{command} {number}', f"{name} · {row['status']}"))
        return matches
    if command in {'/long-research', '/long_research'}:
        try:
            words = shlex.split(fragment)
            selected = bool(words) and any(shlex.split(value) == words[:1] for value, _ in manager.task_catalog)
        except ValueError:
            selected = False
        if selected and (len(words) > 1 or fragment.endswith(' ')):
            return None
        if '=' in fragment:
            return None  # Preserve existing resource/path parameter completion.
        return [(command + ' ' + value + ' ', description)
                for value, description in manager.task_catalog
                if fragment.strip().casefold() in (value + ' ' + description).casefold()]
    return None
