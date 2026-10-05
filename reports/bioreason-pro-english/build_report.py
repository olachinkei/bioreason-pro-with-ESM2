"""Build the English article with the W&B Reports API; reruns update the same report.

Uses the existing WANDB_API_KEY. Never starts training or changes source runs.
The report-data run stores immutable snapshots of four Weave calls and two evals.
"""
import json
import os
import re
from pathlib import Path

import requests
import wandb
import wandb_workspaces.reports.v2 as wr

ROOT = Path(__file__).resolve().parent
ENTITY = "wandb-healthcare"
PROJECT = "bioreasonpro-senpai"
BASE = f"https://wandb.ai/{ENTITY}/{PROJECT}"
STATE = ROOT / "report.json"
CALLS = [
    ("A0A8M9PEU8", "SFT", "01a0d5fa-9b33-7ba8-a39f-17bf79ad15dd"),
    ("A0A8M9PEU8", "RL (GRPO)", "01a0d4e1-48f7-768a-aca7-f849e61ce2f1"),
    ("A0A8M1PV90", "SFT", "01a0d5fa-68e8-736d-a177-3ebaae3ea3c4"),
    ("A0A8M1PV90", "RL (GRPO)", "01a0d4e1-1700-7412-b7c1-4193aea12cac"),
]


def read_call(call_id):
    response = requests.post(
        "https://trace.wandb.ai/call/read",
        auth=("api", os.environ["WANDB_API_KEY"]),
        json={"project_id": f"{ENTITY}/{PROJECT}", "id": call_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["call"]


def snapshot_tables(state):
    if state.get("artifact_version"):
        return
    rows = []
    parents = {}
    for protein, stage, call_id in CALLS:
        call = read_call(call_id)
        output = call["output"]
        scores = output["scores"]
        rows.append([
            protein, stage,
            scores["ia_f1_scorer"]["diag_ia_weighted_f1_per_sample"],
            scores["coverage_scorer"]["diag_n_terms"],
            output["output"],
            json.dumps(scores, ensure_ascii=False),
            f"{BASE}/weave/calls/{call_id}",
        ])
        parents[stage] = call["parent_id"]
    traces = wandb.Table(
        columns=["Protein", "Stage", "IA-weighted F1", "Predicted GO terms",
                 "Full generated output", "Recorded scores", "Source Weave call"],
        data=rows,
    )
    api = wandb.Api(timeout=60)
    eval_rows = []
    for stage, run_id, prefix in [
        ("SFT", "0fjxjs7x", "comparison/sft/"),
        ("RL (GRPO)", "u5axw1dw", "comparison/sft_to_grpo/"),
    ]:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        parent = read_call(parents[stage])
        metrics = dict(run.summary)
        eval_rows.append([
            stage, metrics["evaluated_proteins"],
            *[metrics[prefix + "weighted_fmax" + suffix] for suffix in ["", "_mf", "_bp", "_cc"]],
            json.dumps(parent.get("output"), ensure_ascii=False),
            f"{BASE}/weave/calls/{parents[stage]}", f"{BASE}/runs/{run_id}",
        ])
    evaluations = wandb.Table(
        columns=["Stage", "Proteins", "Weighted F_max", "MF", "BP", "CC",
                 "Weave evaluation output", "Source Weave evaluation", "Source evaluation run"],
        data=eval_rows,
    )
    artifact = wandb.Artifact(
        "bioreason-pro-english-report-evidence", type="report-data",
        description="Snapshots of the two original holdout evaluations and four Weave calls cited in the English report; no new model evaluation.",
        metadata={"source_article": "https://note.com/olachin/n/n71ccc2ca5972", "report_id": state["id"]},
    )
    artifact.add(traces, "case_studies")
    artifact.add(evaluations, "evaluations")
    with wandb.init(
        entity=ENTITY, project=PROJECT, job_type="report-data",
        name="english-article-report-evidence", tags=["report-data", "no-training"],
        dir="/tmp", settings=wandb.Settings(disable_git=True, disable_code=True),
    ) as run:
        logged = run.log_artifact(artifact)
        logged.wait()
        state["artifact_version"] = logged.version
        state["artifact_name"] = logged.name.split(":")[0]
        state["evidence_run"] = run.url
    STATE.write_text(json.dumps(state, indent=2))


def runset(name, ids):
    return wr.Runset(entity=ENTITY, project=PROJECT, name=name,
                     filters=f"ID in {json.dumps(ids)}",
                     custom_run_colors={ids[0]: "#2563EB", **({ids[1]: "#D97706"} if len(ids) > 1 else {})})


def native_blocks(state):
    training = [
        wr.H3(text="SFT training — loss"),
        wr.PanelGrid(hide_run_sets=True, runsets=[runset("SFT training: ktq03cwy", ["ktq03cwy"])], panels=[
            wr.LinePlot(title="SFT loss across 5,000 steps", x="sft/step", y=["sft/loss"],
                        title_x="SFT step", title_y="Loss", smoothing_factor=0.1,
                        smoothing_show_original=True, layout=wr.Layout(w=24, h=8))]),
        wr.H3(text="RL training — reward"),
        wr.PanelGrid(hide_run_sets=True, runsets=[runset("GRPO training: 85vrztci", ["85vrztci"])], panels=[
            wr.LinePlot(title="GRPO reward across 100 steps", x="rl/step", y=["rl/reward"],
                        title_x="RL step", title_y="Reward", smoothing_factor=0,
                        layout=wr.Layout(w=24, h=8))]),
    ]
    panels = []
    for index, (label, suffix) in enumerate([("Overall", ""), ("Molecular Function (MF)", "_mf"), ("Biological Process (BP)", "_bp"), ("Cellular Component (CC)", "_cc")]):
        metrics = [f"comparison/{stage}/weighted_fmax{suffix}" for stage in ["sft", "sft_to_grpo"]]
        panels.append(wr.BarPlot(
            title=f"{label} — weighted F_max", metrics=metrics, orientation="h",
            range_x=(0, 1), title_x="Weighted F_max", max_runs_to_show=2,
            line_titles={metrics[0]: "SFT", metrics[1]: "RL (GRPO)"},
            layout=wr.Layout(x=0, y=6 * index, w=24, h=6),
        ))
    artifact_args = dict(entity=ENTITY, project=PROJECT, artifact=state["artifact_name"], version=state["artifact_version"])
    evaluation = [
        wr.PanelGrid(hide_run_sets=True, runsets=[runset("Full holdout: SFT / RL (8,630 proteins)", ["0fjxjs7x", "u5axw1dw"])], panels=panels),
        wr.H3(text="Embedded evaluation records"),
        wr.P(text="Snapshot of the original Weave evaluation outputs and W&B metrics. Open the source evaluation from the table to inspect all per-protein records."),
        wr.WeaveBlockArtifactVersionedFile(**artifact_args, file="evaluations.table.json"),
    ]
    return {
        "architecture": [wr.Image(url="https://assets.st-note.com/img/1790324219-dLnmKYVA4vFGUtW3D2sZeiQ7.png", caption=["BioReason-Pro architecture and training overview. Source: ", wr.Link(text="BioReason-Pro paper", url="https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1")])],
        "training": training,
        "evaluation": evaluation,
        "traces": [wr.H3(text="Embedded Weave case studies"), wr.WeaveBlockArtifactVersionedFile(**artifact_args, file="case_studies.table.json")],
    }


def main():
    state = json.loads(STATE.read_text())
    snapshot_tables(state)
    inserts = native_blocks(state)
    markdown = (ROOT / "article.md").read_text()
    blocks = []
    for chunk in re.split(r"\n\n+", markdown):
        if chunk.startswith("# "):
            continue
        marker = re.fullmatch(r"<!-- EMBED:(\w+) -->", chunk.strip())
        if marker:
            blocks.extend(inserts[marker.group(1)])
        elif chunk.startswith("### "):
            blocks.append(wr.H3(text=chunk[4:]))
        elif chunk.startswith("## "):
            blocks.append(wr.H2(text=chunk[3:]))
        elif chunk.strip():
            blocks.append(wr.MarkdownBlock(text=chunk))
    report = wr.Report.from_url(state["url"])
    report.blocks = blocks
    report.width = "fixed"
    report.save(draft=True)
    state.update(url=report.url, id=report.id, blocks=len(blocks), status="draft")
    STATE.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
