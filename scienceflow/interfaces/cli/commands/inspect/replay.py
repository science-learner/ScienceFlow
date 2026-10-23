"""Replay preparation command."""

import json

import click

@click.command("replay-prepare")
@click.option(
    "--source",
    "-s",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Saved run or bad-case directory to copy.",
)
@click.option(
    "--out",
    "-o",
    "output",
    required=True,
    type=click.Path(exists=False, file_okay=False),
    help="New isolated replay directory to create.",
)
@click.option(
    "--message-index",
    default=None,
    type=int,
    help="Zero-based memory record index containing the assistant tool call.",
)
@click.option(
    "--tool-call-id",
    default=None,
    help="Tool call id to replay. Mutually exclusive with --message-index.",
)
@click.option(
    "--memory",
    default=None,
    type=click.Path(exists=False, dir_okay=False),
    help="Optional short_term.json or long_term.jsonl source path, relative to copied root or absolute under source.",
)
@click.option(
    "--manifest-name",
    default="replay.yaml",
    show_default=True,
    help="Manifest filename under the copied root to patch.",
)
@click.option(
    "--patch-manifest/--no-patch-manifest",
    default=True,
    show_default=True,
    help="Patch workspace_base in the copied manifest to point at <out>/run.",
)
def replay_prepare_cmd(
    source,
    output,
    message_index,
    tool_call_id,
    memory,
    manifest_name,
    patch_manifest,
):
    """Copy a saved LNR run and truncate memory to replay one pending tool call."""
    from scienceflow.research.solver.lnr.transitions.replay_prepare import ReplayPrepareError, prepare_lnr_replay

    try:
        result = prepare_lnr_replay(
            source,
            output,
            message_index=message_index,
            tool_call_id=tool_call_id,
            memory_path=memory,
            manifest_name=manifest_name,
            patch_manifest=patch_manifest,
        )
    except ReplayPrepareError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
