"""Command line: ``hypercare seed | run | eval | maintain | review | serve``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from hypercare.config import PACKAGE_DIR, get_settings

app = typer.Typer(help="Multi-agent hypercare support system.", no_args_is_help=True)


@app.command()
def seed() -> None:
    """(Re)build the SQLite database from the synthetic data and the knowledge base."""
    from hypercare.seed import build_database

    s = get_settings()
    counts = build_database(s.db_path, s.seed_dir, s.kb_dir)
    typer.echo(f"Database written to {s.db_path}")
    for table, n in counts.items():
        typer.echo(f"  {table:24s} {n}")


@app.command()
def run(
    ticket_id: int,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full run as JSON.")] = False,
    record: Annotated[bool, typer.Option(help="Save live LLM calls for replay.")] = False,
) -> None:
    """Process one ticket through the agents."""
    from hypercare.services import open_services

    services = open_services(record=record)
    result = services.orchestrator().run(ticket_id)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"{result.ticket_ref}  {result.category}/{result.subcategory}")
    for step in result.steps:
        typer.echo(
            f"  {step.agent:26s} {step.action:20s} {step.status:7s} {step.duration_ms:>6d} ms"
            f"  {step.summary[:70]}"
        )
    if result.verification:
        for c in result.verification.checks:
            mark = "PASS" if c.passed else "FAIL"
            typer.echo(f"    [{mark}] {c.name}: {c.detail}")
    typer.echo(
        f"\nDecision: {result.final.decision}"
        + (
            f" (proposed {result.proposed.decision}, overridden)"
            if result.verification and result.verification.overridden and result.proposed
            else ""
        )
    )
    typer.echo(result.final.response)
    if result.final.sources:
        typer.echo(f"Sources: {', '.join(result.final.sources)}")
    typer.echo(
        f"\n{result.duration_ms / 1000:.1f} s, {result.input_tokens} in / "
        f"{result.output_tokens} out tokens, ${result.cost_usd:.4f}"
    )


@app.command("eval")
def evaluate(
    record: Annotated[bool, typer.Option(help="Save live LLM calls for replay.")] = False,
    workers: Annotated[int, typer.Option(help="Tickets processed in parallel.")] = 4,
    out: Annotated[Path, typer.Option(help="Directory for the reports.")] = Path("reports"),
    ticket: Annotated[
        list[int] | None, typer.Option(help="Only these ticket ids (repeatable).")
    ] = None,
    publish: Annotated[
        bool, typer.Option(help="Also save the report for the app's Evaluation tab.")
    ] = False,
) -> None:
    """Run the evaluation cases on a fresh copy of the seed database."""
    from hypercare.db import connect
    from hypercare.evaluation import load_cases, run_evaluation, to_markdown
    from hypercare.llm import build_llm
    from hypercare.seed import build_database
    from hypercare.services import load_index

    s = get_settings()
    cases = load_cases()
    if ticket:
        cases = [c for c in cases if c.ticket_id in set(ticket)]
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "seed.db"
        build_database(db, s.seed_dir, s.kb_dir)
        conn = connect(db)
        report = run_evaluation(
            conn,
            build_llm(s, record=record),
            load_index(conn),
            model=s.model if s.mode == "live" else f"{s.model} (replay)",
            cases=cases,
            workers=workers,
            recurring_min_tickets=s.recurring_min_tickets,
            top_k=s.retrieval_top_k,
        )
        conn.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(report.model_dump_json(indent=2) + "\n")
    (out / "evaluation.md").write_text(to_markdown(report))
    typer.echo(to_markdown(report))
    typer.echo(f"Reports written to {out}/")
    if publish:
        from hypercare.config import DATA_DIR

        (DATA_DIR / "eval" / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
        typer.echo("Report published for the app.")


@app.command()
def maintain(
    record: Annotated[bool, typer.Option(help="Save live LLM calls for replay.")] = False,
) -> None:
    """Detect recurring knowledge-base gaps and draft updates for review."""
    from hypercare.services import open_services

    report = open_services(record=record).maintenance().run()
    typer.echo(f"Recurring subcategories checked: {report.candidates}")
    typer.echo(f"Covered by the documentation: {report.covered or '-'}")
    typer.echo(f"Already have a proposal: {report.skipped_existing or '-'}")
    for p in report.proposals:
        typer.echo(f"\nProposal #{p.proposal_id} for {p.target_document}: {p.proposed_title}")
        for c in p.checks:
            typer.echo(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    typer.echo(f"\nCost: ${report.usage.cost_usd:.4f}")


@app.command()
def review(
    proposal_id: int,
    reviewer: Annotated[str, typer.Option(help="Name of the reviewer.")],
    approve: Annotated[bool, typer.Option("--approve/--reject")] = True,
    comment: str = "",
) -> None:
    """Approve or reject a knowledge-base proposal."""
    from hypercare.agents.maintenance import ReviewError, review_proposal
    from hypercare.db import connect
    from hypercare.services import ensure_database

    s = get_settings()
    ensure_database(s)
    try:
        result = review_proposal(
            connect(s.db_path),
            proposal_id,
            "APPROVED" if approve else "REJECTED",
            reviewer,
            comment,
        )
    except ReviewError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(json.dumps(result, indent=2))


@app.command()
def serve(
    port: Annotated[int, typer.Option(envvar="PORT")] = 8501,
) -> None:
    """Start the Streamlit app."""
    script = PACKAGE_DIR / "ui" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",  # noqa: S104 - container
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    raise typer.Exit(subprocess.call(cmd, env=os.environ.copy()))  # noqa: S603
