import json
import sys
import typer
import httpx

API_URL = "http://127.0.0.1:8000/chat"
cli = typer.Typer(add_completion=False)

@cli.command()
def main(
    repo: str = typer.Option(..., help="owner/repo"),
    question: str = typer.Argument(None, help="Your question. If omitted, enter interactive mode."),
):
    if question is None:
        typer.echo(f"Chatting with repo {repo}. Type 'exit' to quit.\n")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            _ask(repo, q)
    else:
        _ask(repo, question)

def _ask(repo: str, q: str):
    try:
        resp = httpx.post(API_URL, json={"repo": repo, "question": q})
        if resp.status_code >= 400:
            typer.secho(f"Error: {resp.status_code} {resp.text}", fg=typer.colors.RED)
            return
        data = resp.json()
        typer.secho(data["text"], fg=typer.colors.GREEN)
        if data.get("timestamps"):
            typer.echo("\nTimestamps:")
            for t in data["timestamps"]:
                typer.echo(f"  - {t}")
        if data.get("citations"):
            typer.echo("\nCitations:")
            for c in data["citations"]:
                typer.echo(f"  - {c}")
        typer.echo("")
    except Exception as e:
        typer.secho(f"Client error: {e}", fg=typer.colors.RED)

if __name__ == "__main__":
    cli()
