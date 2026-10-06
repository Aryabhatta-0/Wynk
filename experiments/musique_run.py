"""Fixed vs random vs ACO on the frozen MuSiQue-Answerable subset with a REAL model backend.

    python -m experiments.musique_run prepare  --source <musique_ans_v1.0_dev.jsonl> [--freeze]
    python -m experiments.musique_run run      --source ... --out DIR [--seeds 0 1] [--env-file]
    python -m experiments.musique_run assemble --source ... --out DIR [--official-repo PATH]

* ``prepare`` rebuilds the subset from the official file and checks it against the committed
  manifest (``experiments/musique_frozen/manifest.json``); ``--freeze`` writes that manifest
  once, before any strategy has run.
* ``run`` executes ``run_strategy`` for each (seed, strategy) of the frozen protocol and writes
  each record to ``DIR/runs/<strategy>/seed-<n>.json`` as soon as it finishes (several ``run``
  processes may cover different seeds). It needs a configured OpenAI-compatible backend
  (``GEMMA_BASE_URL`` / ``GEMMA_MODEL`` / ``GEMMA_API_KEY``); without one it exits BLOCKED -
  there is no stand-in model. The client's model hash must equal the protocol's
  ``expected_model_hash`` before anything runs.
* ``assemble`` builds the canonical artifact from the per-run records (``assemble``), writes the
  report, the resource-axis learning curves, the champions' validation predictions in the
  official MuSiQue prediction format and - given the official repository - the official
  ``evaluate_v1.0.py`` answer F1 over the same predictions, as a cross-check only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.dataset import SplitRole
from core.experiment import ModelConfiguration
from experiments.budget_ledger import ExperimentBudget
from experiments.musique import check_manifest, prepare
from experiments.optimization_experiment import (
    ExperimentPlan,
    Strategy,
    assemble,
    dumps,
    resource_curves,
    run_strategy,
    searchable_evaluator,
)

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "musique_frozen" / "protocol.json"
MANIFEST = ROOT / "musique_frozen" / "manifest.json"
BLOCKED = 3  # exit code: no real backend configured


def load_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plan_from(protocol: Mapping[str, Any], manifest_hash: str) -> ExperimentPlan:
    return ExperimentPlan(
        model=ModelConfiguration(**protocol["model"]),
        expected_model_hash=protocol["expected_model_hash"],
        expected_prompt_version=protocol["expected_prompt_version"],
        dataset_manifest_hash=manifest_hash,
        budget=ExperimentBudget(**protocol["budget"]),
        seeds=tuple(protocol["seeds"]),
        strategies=tuple(Strategy(s) for s in protocol["strategies"]),
        trials=protocol["trials"],
        batch_size=protocol["batch_size"],
        lcb_z=protocol["lcb_z"],
    )


def frozen(source: Path, protocol: Mapping[str, Any]):
    contract, splits, data, manifest, rows = prepare(source, protocol)
    if not MANIFEST.is_file():
        raise SystemExit(f"no frozen manifest at {MANIFEST}: run `prepare --freeze` first")
    check_manifest(manifest, json.loads(MANIFEST.read_text(encoding="utf-8")))
    return contract, splits, data, manifest, rows


def cmd_prepare(args) -> None:
    protocol = load_protocol()
    contract, splits, data, manifest, _ = prepare(args.source, protocol)
    if args.freeze:
        if MANIFEST.is_file():
            raise SystemExit("manifest is already frozen; it is never rewritten")
        MANIFEST.write_text(dumps(manifest), encoding="utf-8")
    else:
        check_manifest(manifest, json.loads(MANIFEST.read_text(encoding="utf-8")))
    if args.write_subset:
        args.write_subset.parent.mkdir(parents=True, exist_ok=True)
        args.write_subset.write_bytes(data)
    counts = {s.role.value: len(s.row_ids) for s in splits.splits}
    print(f"manifest {manifest['manifest_hash']} rows {contract.dataset.row_count} splits {counts}")


def real_runner(env_file: Path | None):
    from api.chat import load_env_file
    from runtime.gemma_client import GemmaConfig, ModelUnavailableError, OpenAICompatibleClient
    from runtime.runner import WorkflowRunner

    try:
        config = GemmaConfig.from_env(load_env_file(env_file))
    except ModelUnavailableError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(BLOCKED) from exc
    client = OpenAICompatibleClient(config)
    return WorkflowRunner(model=client, benchmark_hash="inline"), client


def cmd_run(args) -> None:
    protocol = load_protocol()
    contract, splits, data, manifest, _ = frozen(args.source, protocol)
    plan = plan_from(protocol, manifest["manifest_hash"])
    runner, client = real_runner(args.env_file)
    if client.model_hash != plan.expected_model_hash:  # bound before ANY call
        raise SystemExit(
            f"model hash {client.model_hash} != frozen {plan.expected_model_hash}: refusing to run"
        )

    def run_workflow(genome, task, trial, seed):
        return runner.run_sync(genome, task, trial=trial, seed=seed)

    suite, evaluate, version = searchable_evaluator(contract, splits, data, run_workflow)
    seeds = args.seeds or list(plan.seeds)
    for seed in seeds:
        for strategy in plan.strategies:
            path = args.out / "runs" / strategy.value / f"seed-{seed}.json"
            if path.is_file():
                print(f"skip {path} (exists)")
                continue
            record = run_strategy(
                plan,
                strategy,
                seed,
                suite,
                evaluate,
                checker=runner.checker,
                evaluator_version=version,
                workers=protocol["execution"]["workers"],
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dumps({"evaluator_version": version, **record}), encoding="utf-8")
            champ = record["champion"]
            print(
                f"{strategy.value} seed {seed}: {record['usage']['candidate_evaluations']} "
                f"candidates, stop={record['stop_reason']}, champion F1="
                f"{champ['validation']['score_mean'] if champ else None}",
                flush=True,
            )


# -- assemble / report ---------------------------------------------------------------------------
def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def predictions_for(run: Mapping[str, Any], val_ids: Sequence[str]) -> list[dict[str, Any]]:
    """The champion's validation predictions (latest trial per row), official format."""
    champ = run["champion"]
    latest: dict[str, Any] = {}
    for cand in run["candidates"]:
        if cand["genome_hash"] != champ["genome_hash"]:
            continue
        for e in cand["runs"]:
            if e["split"] == SplitRole.VALIDATION.value and e["verdict"] is not None:
                latest[e["row_id"]] = e
    out = []
    for rid in val_ids:
        pred = (latest[rid]["prediction"] or {}).get("answer") if rid in latest else None
        out.append(
            {
                "id": rid,
                "predicted_answer": pred if isinstance(pred, str) else "",
                "predicted_support_idxs": [],
                "predicted_answerable": True,
                "wynk_token_f1": latest[rid]["score"] if rid in latest else None,
            }
        )
    return out


def official_answer_f1(
    repo: Path, preds: list[dict[str, Any]], gold_rows: Mapping[str, Mapping], work: Path
) -> dict[str, Any]:
    """Run the official ``evaluate_v1.0.py`` on exactly these predictions (cross-check)."""
    work.mkdir(parents=True, exist_ok=True)
    repo, work = repo.resolve(), work.resolve()  # the evaluator runs with cwd=repo
    pred_path, gold_path = work / "predictions.jsonl", work / "gold.jsonl"
    pred_path.write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    gold_path.write_text(
        "".join(json.dumps(gold_rows[p["id"]]) + "\n" for p in preds), encoding="utf-8"
    )
    code = (
        "import json,sys,importlib.util;"
        f"sys.path.insert(0,{str(repo)!r});"
        f"spec=importlib.util.spec_from_file_location('ev',{str(repo / 'evaluate_v1.0.py')!r});"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        f"print(json.dumps(m.evaluate({str(pred_path)!r},{str(gold_path)!r})))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=repo
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def plot_curves(artifact: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_titles = {
        "evaluation": "candidate evaluations",
        "cumulative_model_calls": "cumulative model calls",
        "cumulative_tokens": "cumulative tokens",
        "cumulative_execution_s": "cumulative execution time (s)",
        "cumulative_e2e_wall_s": "cumulative end-to-end wall time (s)",
        "cumulative_cost": "cumulative cost",
    }
    colors = {"fixed": "#7f7f7f", "random": "#1f77b4", "aco": "#d62728"}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (axis, title) in zip(axs.flat, axes_titles.items(), strict=True):
        drawn = False
        for run in artifact["runs"]:
            pts = resource_curves(run).get(axis)
            if not pts:
                continue
            xs, ys = zip(*pts, strict=True)
            ax.step(
                xs,
                ys,
                where="post",
                color=colors[run["strategy"]],
                alpha=0.7,
                marker="o",
                label=f"{run['strategy']} s{run['seed']}",
            )
            drawn = True
        ax.set_title(title if drawn else f"{title} (not available)")
        ax.set_xlabel(title)
        ax.set_ylabel("best-so-far validation token F1")
        ax.grid(alpha=0.3)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9, fontsize=8)
    fig.suptitle("MuSiQue-Ans dev (frozen subset): fixed vs random vs ACO, validation token F1")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def report(artifact: Mapping[str, Any], official: Mapping[str, Any]) -> str:
    s = artifact["summary"]["by_strategy"]
    prob = artifact["identity"]["problem"]
    lines = [
        "# MuSiQue-Answerable: fixed vs random vs ACO (real model)",
        "",
        f"- experiment_id `{artifact['experiment_id']}`",
        f"- model `{prob['model']['model']}` model_hash `{prob['model_hash']}`"
        f" prompts `{prob['prompt_template_version']}`",
        f"- manifest `{prob['dataset_manifest_hash']}` dataset `{prob['dataset_content_hash']}`"
        f" splits `{prob['splits_hash']}` contract `{prob['task_contract_hash']}`",
        f"- splits {artifact['splits']}; test rows executed: {artifact['test_runs']}",
        f"- evaluator `{prob['evaluator_run_version']}` config {prob['evaluator_config']}",
        "",
        "## Per seed",
        "",
        "| strategy | seed | stop | cand | val F1 | val pass | calls | prompt tok | compl tok |"
        " tokens | tok/cand | mean lat s | p95 lat s | exec s | E2E s | opt ovh s | eval ovh s |"
        " cost |",
        "|" + "---|" * 18,
    ]
    for strategy, v in s.items():
        for p in v["per_seed"]:
            lines.append(
                f"| {strategy} | {p['seed']} | {p['stop_reason']} | {p['candidate_evaluations']}"
                f" | {_fmt(p['champion_validation_score'])}"
                f" | {_fmt(p['champion_validation_pass_rate'])} | {p['model_calls']}"
                f" | {p['prompt_tokens']} | {p['completion_tokens']} | {p['tokens']}"
                f" | {_fmt(p['tokens_per_candidate'], 0)} | {_fmt(p['mean_latency_s'], 1)}"
                f" | {_fmt(p['p95_latency_s'], 1)} | {_fmt(p['execution_s'], 0)}"
                f" | {_fmt(p['e2e_wall_s'], 0)} | {_fmt(p['optimizer_overhead_s'], 2)}"
                f" | {_fmt(p['evaluator_overhead_s'], 3)} | {_fmt(p['cost'])} |"
            )
    lines += [
        "",
        "## Aggregate over seeds (mean / std / median / min / max)",
        "",
        "| strategy | metric | mean | std | median | min | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for strategy, v in s.items():
        for name in (
            "champion_validation_score",
            "champion_validation_pass_rate",
            "candidate_evaluations",
            "model_calls",
            "tokens",
            "mean_latency_s",
            "p95_latency_s",
            "e2e_wall_s",
            "cost",
        ):
            m = v["metrics"][name]
            lines.append(
                f"| {strategy} | {name} | {_fmt(m['mean'])} | {_fmt(m['std'])} |"
                f" {_fmt(m['median'])} | {_fmt(m['min'])} | {_fmt(m['max'])} |"
            )
    lines += [
        "",
        "Highest mean champion validation token F1: "
        + ", ".join(artifact["summary"]["highest_mean_champion_validation_score"]),
        "",
        "## Official MuSiQue answer F1 cross-check (champion validation predictions)",
        "",
        "| strategy | seed | Wynk token F1 (answer only) | official answer F1 (answer + aliases) |"
        " official EM |",
        "|---|---|---|---|---|",
    ]
    for (strategy, seed), o in sorted(official.items()):
        lines.append(
            f"| {strategy} | {seed} | {_fmt(o['wynk_token_f1'])} | {_fmt(o.get('answer_f1'))} |"
            f" {_fmt(o.get('answer_em'))} |"
        )
    return "\n".join(lines) + "\n"


def cmd_assemble(args) -> None:
    protocol = load_protocol()
    contract, splits, data, manifest, rows = frozen(args.source, protocol)
    plan = plan_from(protocol, manifest["manifest_hash"])
    records = []
    versions = set()
    for strategy in plan.strategies:
        for seed in plan.seeds:
            rec = json.loads(
                (args.out / "runs" / strategy.value / f"seed-{seed}.json").read_text("utf-8")
            )
            versions.add(rec.pop("evaluator_version"))
            records.append(rec)
    if len(versions) != 1:
        raise SystemExit(f"runs disagree on the evaluator version: {versions}")
    from experiments.contract_run import contract_suite

    suite, _ = contract_suite(contract, splits, data)
    artifact = assemble(
        plan,
        suite,
        records,
        evaluator_version=versions.pop(),
        synthetic=False,
        provenance=manifest,
    )
    (args.out / "experiment.json").write_text(dumps(artifact), encoding="utf-8")
    plot_curves(artifact, args.out / "learning_curves.png")

    val_ids = list(splits.split(SplitRole.VALIDATION).row_ids)
    gold = {r["id"]: r for r in rows}
    official: dict[tuple[str, int], dict[str, Any]] = {}
    for run in artifact["runs"]:
        if run["champion"] is None:
            continue
        preds = predictions_for(run, val_ids)
        key = (run["strategy"], run["seed"])
        pdir = args.out / "predictions" / run["strategy"] / f"seed-{run['seed']}"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "champion_validation_predictions.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8"
        )
        f1s = [p["wynk_token_f1"] or 0.0 for p in preds]
        official[key] = {"wynk_token_f1": sum(f1s) / len(f1s)}
        if args.official_repo:
            official[key] |= official_answer_f1(
                args.official_repo,
                [{k: v for k, v in p.items() if k != "wynk_token_f1"} for p in preds],
                gold,
                args.out / "official_eval" / run["strategy"] / f"seed-{run['seed']}",
            )
    (args.out / "official_crosscheck.json").write_text(
        dumps({f"{k[0]}/seed-{k[1]}": v for k, v in official.items()}), encoding="utf-8"
    )
    (args.out / "REPORT.md").write_text(report(artifact, official), encoding="utf-8")
    print((args.out / "REPORT.md").read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "run", "assemble"):
        sp = sub.add_parser(name)
        sp.add_argument("--source", type=Path, required=True)
        if name == "prepare":
            sp.add_argument("--freeze", action="store_true")
            sp.add_argument("--write-subset", type=Path)
        if name == "run":
            sp.add_argument("--env-file", type=Path, default=Path(".env"))
            sp.add_argument("--seeds", type=int, nargs="*")
        if name in ("run", "assemble"):
            sp.add_argument("--out", type=Path, required=True)
        if name == "assemble":
            sp.add_argument("--official-repo", type=Path)
    args = p.parse_args(argv)
    {"prepare": cmd_prepare, "run": cmd_run, "assemble": cmd_assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
