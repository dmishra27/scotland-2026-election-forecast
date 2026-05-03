"""CLI: run the end-to-end training pipeline."""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parents[1]))


@click.command()
@click.option("--data-path", default=None, help="Path to voters.parquet (generates fresh if absent)")
@click.option("--n-voters", default=12500, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--n-trials", default=50, show_default=True, help="Optuna trials per base learner")
@click.option("--model-dir", default="models", show_default=True)
@click.option("--tracking-uri", default=None, show_default=True, help="MLflow tracking URI (omit to use local file store)")
@click.option("--experiment", default="scotland-2026-forecast", show_default=True)
def main(data_path, n_voters, seed, n_trials, model_dir, tracking_uri, experiment):
    """Train stacking ensemble for Scotland 2026 election forecast."""
    import json

    click.echo("Starting training pipeline ...")
    click.echo(f"  n_voters={n_voters}, seed={seed}, n_trials={n_trials}")
    click.echo(f"  model_dir={model_dir}, tracking_uri={tracking_uri}")

    from src.orchestration.train_pipeline import run_pipeline

    metrics = run_pipeline(
        data_path=Path(data_path) if data_path else None,
        n_voters=n_voters,
        random_seed=seed,
        n_trials=n_trials,
        model_dir=Path(model_dir),
        tracking_uri=tracking_uri,
        experiment_name=experiment,
    )

    click.echo("\nFinal test metrics:")
    for k, v in metrics.items():
        click.echo(f"  {k:<30} {v}")

    click.echo(f"\nModels saved to {model_dir}/")
    click.echo("Done.")


if __name__ == "__main__":
    main()
