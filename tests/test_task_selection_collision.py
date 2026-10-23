"""Task discovery keeps registered names usable after successful-run export."""

from types import SimpleNamespace

from scienceflow.research.onboarding.library import catalog


def test_builtin_name_and_completion_stay_short_with_same_name_local_task(monkeypatch, tmp_path):
    local = tmp_path / 'task_library' / 'circle-packing-version'
    builtin = tmp_path / 'builtins' / 'circle-packing'
    monkeypatch.setattr(catalog, 'local_packages', lambda _: [
        SimpleNamespace(source_dir=local, task_id='circle-packing'),
    ])
    monkeypatch.setattr(catalog, 'iter_task_packages', lambda: [
        SimpleNamespace(source_dir=builtin, task_id='circle-packing'),
    ])
    text, choices = catalog.task_selection_text(tmp_path)
    assert len(choices) == 2
    assert len({choice.task_id for choice in choices}) == 2
    assert text.count('circle-packing') == 2
    assert catalog.resolve_task_selection('circle-packing', choices) == 'circle-packing'
    assert catalog.resolve_task_selection('1', choices) == str(local)
    assert catalog.resolve_task_selection('2', choices) == 'circle-packing'
    assert catalog.resolve_task_selection(local.name, choices) == str(local)
    assert catalog.resolve_task_selection('circle-packing workers=2', choices) == (
        'circle-packing workers=2'
    )
    assert catalog.task_choices(tmp_path) == (
        (local.name, 'circle-packing · Local'),
        ('circle-packing', 'circle-packing · Built-in'),
    )
    assert catalog.resolve_task_selection(str(local), choices) == str(local)


def test_catalog_preserves_each_local_registration_with_same_display_name(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, 'iter_task_packages', lambda: [])
    monkeypatch.setattr(catalog, 'local_packages', lambda _: [
        SimpleNamespace(source_dir=tmp_path / 'one', task_id='packing'),
        SimpleNamespace(source_dir=tmp_path / 'two', task_id='packing'),
    ])
    choices = catalog.discover_tasks(tmp_path)
    assert len(choices) == 2
    assert catalog.resolve_task_selection('one', choices) == str(tmp_path / 'one')
    assert catalog.resolve_task_selection('two', choices) == str(tmp_path / 'two')
    assert catalog.task_choices(tmp_path) == (
        ('one', 'packing · Local'), ('two', 'packing · Local'),
    )


def test_builtin_resolution_is_not_shadowed_by_local_export(monkeypatch, tmp_path):
    from scienceflow.research.onboarding.support import task_defaults

    builtin = SimpleNamespace(task_id='circle-packing')
    local = SimpleNamespace(task_id='circle-packing')
    monkeypatch.setattr(task_defaults, 'find_task_package',
                        lambda value: builtin if value == 'circle-packing' else None)
    monkeypatch.setattr(catalog, 'resolve_local', lambda *_: local)
    assert task_defaults.resolve_task_package(
        '', 'circle-packing workers=2', workspace=tmp_path,
    ) is builtin
    assert task_defaults.resolve_task_package(
        '', '/tasks/local-copy', workspace=tmp_path,
    ) is local


def test_long_research_partial_input_completes_to_id_without_paths(monkeypatch, tmp_path):
    from scienceflow.interfaces.ui.research.control.tasks.completion import task_completions

    monkeypatch.setattr(catalog, 'local_packages', lambda _: [
        SimpleNamespace(source_dir=tmp_path / 'task_library' / 'circle-packing-copy',
                        task_id='circle-packing'),
    ])
    monkeypatch.setattr(catalog, 'iter_task_packages', lambda: [
        SimpleNamespace(source_dir=tmp_path / 'builtins' / 'circle-packing',
                        task_id='circle-packing'),
    ])
    manager = SimpleNamespace(task_catalog=catalog.task_choices(tmp_path))
    assert task_completions(manager, '/long-research cir') == [
        ('/long-research circle-packing-copy ', 'circle-packing · Local'),
        ('/long-research circle-packing ', 'circle-packing · Built-in'),
    ]


def test_quoted_local_ids_and_tabs_preserve_resource_arguments():
    choices = (
        catalog.TaskChoice('/tmp/task library/local copy', 'local copy', 'Local'),
        catalog.TaskChoice('circle-packing', 'circle-packing', 'Built-in'),
    )
    assert catalog.resolve_task_selection("'local copy'", choices) == choices[0].value
    assert catalog.resolve_task_selection("'local copy' workers=2 model='my model'", choices) == (
        "'/tmp/task library/local copy' workers=2 model='my model'"
    )
    assert catalog.resolve_task_selection('circle-packing\tworkers=2', choices) == (
        'circle-packing workers=2'
    )
    assert catalog.resolve_task_selection('1\tworkers=2', choices) == (
        "'/tmp/task library/local copy' workers=2"
    )
    assert catalog.resolve_task_selection('find a new packing', choices) == 'find a new packing'
