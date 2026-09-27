"""vbr config: create, show and check configuration files (DESIGN.md §4)."""

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from video_beep_remover.cli.app import config_app
from video_beep_remover.cli.console import console, fail, load_or_exit
from video_beep_remover.cli.options import ConfigOpt
from video_beep_remover.config.loader import defaults_text, redact, to_toml, user_config_path
from video_beep_remover.detect.lexicon import compile_lexicon
from video_beep_remover.errors import UsageError, VbrError


@config_app.command("init")
def config_init(
    path: Annotated[
        Path | None, typer.Argument(help="Where to write it (default: the per-user config).")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing file.")] = False,
) -> None:
    """Write a starter config with every option and its default."""
    target = path or user_config_path()
    if target.exists() and not force:
        raise fail(UsageError(f"{target} already exists (use --force to replace it)"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(defaults_text(), "utf-8")
    console.print(f"Wrote {escape(str(target))}")


@config_app.command("show")
def config_show(config: ConfigOpt = None) -> None:
    """Print the effective configuration (defaults + your file), secrets redacted."""
    loaded = load_or_exit(config)
    source = loaded.source or "built-in defaults only"
    typer.echo(f"# effective configuration; source: {source}\n")
    typer.echo(to_toml(redact(loaded.config.model_dump(mode="json"))), nl=False)


@config_app.command("check")
def config_check(config: ConfigOpt = None) -> None:
    """Validate the configuration and compile the word list."""
    loaded = load_or_exit(config)
    try:
        lexicon = compile_lexicon(loaded.config.lexicon, base_dir=loaded.base_dir)
    except VbrError as exc:
        raise fail(exc) from exc
    console.print(f"config: {escape(str(loaded.source or 'built-in defaults only'))}")
    counts: dict[str, int] = {}
    for term in lexicon.terms:
        counts[term.category] = counts.get(term.category, 0) + 1
    listed = ", ".join(f"{name} ({counts.get(name, 0)} terms)" for name in lexicon.categories)
    console.print(f"enabled categories: {escape(listed) or 'none'}")
    for warning in lexicon.warnings:
        console.print(f"[yellow]warning:[/] {escape(warning)}")
    if not lexicon.terms and not lexicon.masked_patterns:
        raise fail(UsageError("the word list is empty: enable a category or add terms"))
    console.print("[green]✔[/] configuration is valid")
