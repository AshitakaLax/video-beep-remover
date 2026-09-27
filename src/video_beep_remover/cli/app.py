"""The `vbr` Typer application and its sub-command groups; the commands register themselves on them."""

from typing import Annotated

import typer

from video_beep_remover import __version__

app = typer.Typer(
    name="vbr",
    help="Mute a configurable list of words in video files.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
config_app = typer.Typer(help="Create, show and check configuration files.", no_args_is_help=True)
app.add_typer(config_app, name="config")
cache_app = typer.Typer(
    help="Inspect or clear the cache of downloaded subtitles and transcripts.", no_args_is_help=True
)
app.add_typer(cache_app, name="cache")


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"vbr {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=version_callback, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Mute a configurable list of words in video files."""
