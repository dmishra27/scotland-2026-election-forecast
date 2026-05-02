"""CLI: generate synthetic voter data and save to data/raw/voters.parquet."""

import sys
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))


@click.command()
@click.option("--n-voters", default=None, type=int, help="Override n_voters from config")
@click.option("--seed", default=None, type=int, help="Override random seed")
@click.option("--out", default="data/raw/voters.parquet", show_default=True)
@click.option("--config", default="configs/config.yaml", show_default=True)
def main(n_voters, seed, out, config):
    """Generate synthetic Scotland 2026 voter micro-panel."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    n = n_voters or cfg["data"]["n_voters"]
    s = seed if seed is not None else cfg["data"]["random_seed"]
    conc = cfg["data"]["dirichlet_concentration"]

    from src.data.generate_voters import generate_voters, get_vote_share_summary

    click.echo(f"Generating {n:,} voters (seed={s}, concentration={conc}) ...")
    df = generate_voters(n_voters=n, random_seed=s, dirichlet_concentration=conc)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    summary = get_vote_share_summary(df)
    click.echo(f"Saved {len(df):,} rows → {out_path}")
    click.echo("Constituency vote shares:")
    for party, share in sorted(summary["constituency"].items(), key=lambda x: -x[1]):
        click.echo(f"  {party:<14} {share:.1%}")
    click.echo(f"Tactical rate: {summary['tactical_rate']:.1%}")


if __name__ == "__main__":
    main()
