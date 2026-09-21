"""Bounded, explicit evaluator files; never copy runtime/configuration directories."""

import ast
import importlib.metadata
import re
import shlex
import sys
from pathlib import Path

_SECRET = re.compile(r"""(?i)\b(?:api[_ -]?keys?|password|access_token)["']?\s*[=:]\s*\S+|\bsk-[A-Za-z0-9_-]{16,}""")


def clean_text(text):
    if _SECRET.search(text):
        raise ValueError('Credential-like text prevents automatic task export.')
    return text


def copy_file(source, target):
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 2_000_000:
        raise ValueError(f'File is not a portable small regular file: {source.name}')
    content = source.read_bytes()
    clean_text(content.decode('utf-8'))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def copy_sources(source, target):
    files = [p for p in source.rglob('*') if p.is_file() and p.suffix in {'.py', '.md', '.yaml', '.json', '.txt'}
             and not any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(source).parts)]
    if len(files) > 128 or sum(p.stat().st_size for p in files) > 8_000_000:
        raise ValueError('Task sources exceed the automatic export size limit.')
    for path in files:
        if any(part.lower() in {'dataset', 'data', 'logs', 'runs', 'workers', 'sessions'} for part in path.relative_to(source).parts):
            continue
        if path.name.lower() in {'config.json', 'config.yaml', 'credentials.json'}:
            raise ValueError('Separate evaluator dependencies from private configuration before export.')
        copy_file(path, target / path.relative_to(source))


def requirements(directory):
    modules = set()
    for path in directory.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules.add(node.module.split('.')[0])
    packages = importlib.metadata.packages_distributions()
    local = {p.stem for p in directory.rglob('*.py')} | {p.name for p in directory.rglob('*') if p.is_dir()}
    lines = []
    for module in sorted(modules - set(sys.stdlib_module_names) - local):
        distributions = packages.get(module, [])
        if not distributions:
            raise ValueError(f'Cannot identify evaluator dependency: {module}')
        for distribution in distributions:
            lines.append(f'{distribution}=={importlib.metadata.version(distribution)}')
    (directory / 'requirements.txt').write_text('\n'.join(sorted(set(lines))) + '\n')


def command_bundle(command, validation, output):
    tokens = shlex.split(command)
    if len(tokens) < 3 or not (Path(tokens[0]).name.startswith('python') or tokens[0] == '{python}'):
        raise ValueError('Automatic export currently requires a Python script evaluator or a registered task package.')
    script = Path(tokens[1])
    if not script.is_absolute() or script.suffix != '.py' or '{artifact_abs_path}' not in tokens:
        raise ValueError('Evaluator must reference an absolute Python script and an artifact argument.')
    if validation.get('command') != command or not validation.get('valid_passed') or not validation.get('invalid_rejected'):
        raise ValueError('Missing matching valid/invalid evaluator validation evidence.')
    # Copy only evaluator sources, never its surrounding experiment workspace.
    files = [script, *[p for p in script.parent.glob('*.py') if p != script]]
    if len(files) > 64:
        raise ValueError('Too many evaluator modules for automatic export.')
    for path in files:
        tree = ast.parse(path.read_text())
        if any(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('/') for n in ast.walk(tree)):
            raise ValueError('Evaluator contains absolute file dependencies; make them package-relative before export.')
        copy_file(path, output / 'evaluator' / path.name)
    args = []
    for token in tokens[2:]:
        if token == '{artifact_abs_path}':
            args.append(token)
        elif '/' in token or '\\' in token or any(c in token for c in ';|&$`'):
            raise ValueError('External paths or shell operations need an explicit portable evaluator adapter.')
        else:
            args.append(shlex.quote(token))
    samples = validation.get('samples') or []
    if len(samples) != 2:
        raise ValueError('Two validation samples are required.')
    for name, source in zip(('valid', 'invalid'), samples):
        path = Path(source)
        copy_file(path, output / 'examples' / (name + path.suffix))
    normalized = '{python} {task_dir}/evaluator/' + shlex.quote(script.name) + ' ' + ' '.join(args)
    return normalized, {'valid_artifact': str(next((output / 'examples').glob('valid.*'))),
                        'invalid_artifact': str(next((output / 'examples').glob('invalid.*')))}
