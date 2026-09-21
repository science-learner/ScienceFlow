"""Publish successful experiments as portable local task folders, atomically."""

import asyncio
import hashlib
import json
import math
import re
import shlex
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from scienceflow.runtime.parallel.control.registry import locked
from scienceflow.runtime.task_package import find_task_package, load_task_package

from ..support.preparation.validation import validate_evaluator
from .portable import clean_text, command_bundle, copy_sources, requirements


def valid_best(record):
    values = []
    for root in record.get('task_roots', ()):
        for path in Path(root).glob('workers/*/logs/lhr_state.json'):
            if path.stat().st_size > 4_000_000:
                continue
            state = json.loads(path.read_text())
            if not isinstance(state, dict):
                continue
            best = state.get('global_best') or {}
            if not isinstance(best, dict):
                continue
            score = best.get('metric_value')
            if best.get('validation_ok') is True and isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                values.append(score)
    if not values:
        raise ValueError('No evaluator-validated finite result was recorded.')
    return min(values) if record['draft']['lower_is_better'] else max(values)


def fingerprint(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.name != 'provenance.json' and '__pycache__' not in path.parts:
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_package(record, output):
    draft = record['draft']
    if not draft.get('task_text') or not draft.get('metric_name') or not isinstance(draft.get('lower_is_better'), bool):
        raise ValueError('Task description and explicit metric direction are required.')
    evaluator = draft.get('evaluator') or {}
    command = draft.get('artifact_command') or (evaluator.get('command') or {}).get('evaluator_command')
    source = evaluator.get('package_source')
    if command and source:
        spec = load_task_package(Path(source))
        copy_sources(spec.source_dir, output)
        config = dict(spec.config)
        normalized = config['evaluator']['command']
        samples = {key: str(output / value) for key, value in config.get('validation', {}).items()}
        rendered = normalized.replace('{python}', shlex.quote(sys.executable)).replace('{task_dir}', shlex.quote(str(output)))
        asyncio.run(validate_evaluator(rendered, samples, output.parent / 'checks'))
        description = (spec.source_dir / spec.description_relpath).read_text()
    elif command:
        workspace = Path(draft['workspace_base']).parent
        report = workspace / '.scienceflow/preparation/validation.json'
        validation = json.loads(report.read_text()) if report.is_file() else {}
        normalized, samples = command_bundle(command, validation, output)
        rendered = normalized.replace('{python}', shlex.quote(sys.executable)).replace('{task_dir}', shlex.quote(str(output)))
        asyncio.run(validate_evaluator(rendered, samples, output.parent / 'checks'))
        config = dict(id=draft.get('exp_id') or 'research', category='local', profile='artifact_command',
                      description='description.md', artifact={'path': draft['artifact_path'], 'kind': draft.get('artifact_kind') or 'artifact'},
                      metric={'name': draft['metric_name'], 'lower_is_better': draft['lower_is_better'], 'type': 'exploratory', 'authoritative': False},
                      evaluator={'command': normalized},
                      validation={key: str(Path(value).relative_to(output)) for key, value in samples.items()})
        description = draft['task_text']
    else:
        source = evaluator.get('package_source')
        spec = load_task_package(Path(source)) if source else find_task_package(draft.get('exp_id', ''))
        if spec is None:
            raise ValueError('No registered evaluator source is available.')
        copy_sources(spec.source_dir, output)
        config = dict(spec.config)
        description = (spec.source_dir / spec.description_relpath).read_text()
        config['description'] = 'description.md'
        entry = spec.evaluator_entrypoint
        if entry.startswith('../_shared/'):
            copy_sources(spec.source_dir.parent / '_shared', output / '_shared')
            config['evaluator'] = {**config['evaluator'], 'entrypoint': entry[3:]}
        elif '..' in entry.split(':')[0].split('/') or entry.startswith('/'):
            raise ValueError('Evaluator entrypoint depends on files outside its task package.')
    if (config.get('metric') or {}).get('lower_is_better') is None:
        config['metric'] = {**config.get('metric', {}), 'lower_is_better': draft['lower_is_better']}
    config['inputs'] = {'required': draft.get('input_data_dir') != 'none'}
    clean_text(json.dumps(config))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'description.md').write_text(clean_text(description))
    (output / 'task.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    if not (output / 'requirements.txt').exists():
        requirements(output)
    load_task_package(output)
    return config


def export_success(record):
    """No model calls. Deferred exports leave the successful experiment untouched."""
    if record.get('status') != 'completed' or not record.get('draft'):
        return {'status': 'skipped', 'reason': 'Only completed experiments can be registered.'}
    try:
        score = valid_best(record)
        workspace = Path(record.get('library_workspace') or record.get('control_workspace') or Path(record['draft']['workspace_base']).parent)
        library = workspace / 'task_library'
        library.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.export-', dir=library) as temporary:
            output = Path(temporary) / 'package'
            output.mkdir()
            config = build_package(record, output)
            identity = fingerprint(output)
            slug = re.sub(r'[^a-zA-Z0-9_-]+', '-', config['id']).strip('-')[:60] or 'research'
            target = library / f'{slug}-{identity[:12]}'
            with locked(library):
                provenance = {'schema_version': 1, 'fingerprint': identity, 'runs': [], 'best_metric': score}
                if target.exists():
                    if fingerprint(target) != identity:
                        raise ValueError('Existing task package was edited; it will not be overwritten.')
                    provenance = json.loads((target / 'provenance.json').read_text())
                if record['run_id'] not in {row['run_id'] for row in provenance['runs']}:
                    provenance['runs'].append({'run_id': record['run_id'], 'metric': score, 'finished_at': record.get('finished_at')})
                scores = [row['metric'] for row in provenance['runs']]
                provenance['best_metric'] = min(scores) if config['metric']['lower_is_better'] else max(scores)
                (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
                if not target.exists():
                    output.rename(target)
                else:
                    (output / 'provenance.json').replace(target / 'provenance.json')
            return {'status': 'registered', 'path': str(target), 'task_id': config['id'], 'best_metric': provenance['best_metric']}
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, yaml.YAMLError) as exc:
        return {'status': 'deferred', 'reason': str(exc)}
