#!/usr/bin/env python3
"""Run a paired CLAP prompt-suite checkpoint comparison and publish it to R2.

The destination follows the validation audio-probe shape: each ``audio/sample_*``
directory contains baseline and candidate WAV/CSV pairs, while aggregate and
per-prompt comparisons live under ``metrics/``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import click
import numpy as np
import torch
from sh import Command

from synth_setter.clap import (
    DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256,
    clap_checkpoint_sha256,
    resolve_clap_checkpoint,
)
from synth_setter.cli import clap_render
from synth_setter.cli.clap_render import summarize_cosine_distances, write_summary_csv
from synth_setter.conditioning import resolve_embedding_conditioning
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.meanaudio_generation import (
    MEANAUDIO_DURATION_SECONDS,
    MEANAUDIO_LATENT_SHAPE,
    MEANAUDIO_STEPS,
    load_meanaudio_s_full_generator,
    load_meanaudio_s_full_reencoded_generator,
    meanaudio_s_full_provenance,
    meanaudio_s_full_reencoded_provenance,
    validate_meanaudio_s_full_latent,
)
from synth_setter.pipeline.schemas.spec import RenderConfig

BASELINE_SUITE = "r2://experiments/clap-renders/suites/clap-suite-20260731T211818136757Z"
BASELINE_CHECKPOINT = (
    "r2://intermediate-data/checkpoints/flow_simple_440k_1m_clap/"
    "flow_simple_440k_1m_clap-20260730T215504153Z-"
    "588c02237a964b0aad982370cf347086/last.ckpt"
)
DEFAULT_MEANAUDIO_INVERSE_CHECKPOINT = (
    "r2://intermediate-data/checkpoints/flow_sketch_prelim_base/"
    "flow_sketch_prelim_base-20260901T230441570Z-"
    "1463d84a00744d739e598e01bdce4196/last.ckpt"
)
CANDIDATE_SOURCE_CLAP = "clap"
CANDIDATE_SOURCE_MEANAUDIO_S_FULL = "meanaudio-s-full"
CANDIDATE_SOURCE_MEANAUDIO_S_FULL_REENCODED = "meanaudio-s-full-reencoded"
type DeviceSetting = Literal["auto", "cpu", "cuda", "mps"]
SUMMARY_FIELDS = (
    "count",
    "mean",
    "std_population",
    "min",
    "p25",
    "median",
    "p75",
    "max",
)
PAIR_FIELDS = (
    "index",
    "prompt",
    "baseline_cosine_similarity",
    "baseline_cosine_distance",
    "candidate_cosine_similarity",
    "candidate_cosine_distance",
    "distance_delta_candidate_minus_baseline",
    "winner",
)


def build_candidate_identity(
    candidate_source: str,
    candidate_checkpoint: str,
    *,
    steps: int,
    duration_seconds: float,
    seed: int,
) -> dict[str, float | int | str]:
    """Build the complete identity that gates local candidate artifact reuse.

    :param candidate_source: Explicit candidate generation backend.
    :param candidate_checkpoint: Immutable inverse-checkpoint URI.
    :param steps: MeanAudio mean-flow steps; ignored for CLAP candidates.
    :param duration_seconds: MeanAudio target duration; ignored for CLAP candidates.
    :param seed: Candidate generation and inverse-flow seed.
    :returns: Stable source, generation, weight, and inverse identities.
    :raises ValueError: The candidate source is unsupported.
    """
    identity: dict[str, float | int | str] = {
        "candidate_source": candidate_source,
        "candidate_checkpoint": candidate_checkpoint,
        "candidate_checkpoint_identity": f"uri:{candidate_checkpoint}",
        "seed": seed,
    }
    if candidate_source == CANDIDATE_SOURCE_CLAP:
        return identity
    if candidate_source == CANDIDATE_SOURCE_MEANAUDIO_S_FULL:
        provenance = meanaudio_s_full_provenance()
    elif candidate_source == CANDIDATE_SOURCE_MEANAUDIO_S_FULL_REENCODED:
        provenance = meanaudio_s_full_reencoded_provenance()
    else:
        raise ValueError(f"unsupported candidate source {candidate_source!r}")
    return identity | {
        **provenance,
        "meanaudio_steps": steps,
        "meanaudio_duration_seconds": duration_seconds,
    }


def ensure_resume_identity(
    path: Path,
    identity: Mapping[str, float | int | str],
) -> None:
    """Create or verify the identity manifest that authorizes local cache reuse.

    :param path: Local candidate identity JSON path.
    :param identity: Complete identity for this invocation.
    :raises ValueError: Existing cache identity differs from this invocation.
    """
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached != identity:
            raise ValueError(f"cached candidate identity in {path} does not match this invocation")
        return
    candidate_cache = path.parent / "audio"
    if candidate_cache.is_dir() and next(candidate_cache.glob("*/candidate.*"), None) is not None:
        raise ValueError(f"candidate cache under {candidate_cache} has no identity")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_paired_row(
    index: int,
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
) -> dict[str, int | float | str]:
    """Build one paired metric row with lower cosine distance as the winner.

    :param index: Stable one-based prompt index.
    :param baseline: Baseline CLAP comparison row.
    :param candidate: Candidate CLAP comparison row.
    :returns: Paired metrics and winning arm.
    :raises ValueError: Prompt identities differ between arms.
    """
    if baseline["prompt"] != candidate["prompt"]:
        raise ValueError(f"prompt mismatch at index {index}")
    baseline_distance = float(baseline["cosine_distance"])
    candidate_distance = float(candidate["cosine_distance"])
    delta = round(candidate_distance - baseline_distance, 12)
    winner = "tie"
    if delta < 0:
        winner = "candidate"
    elif delta > 0:
        winner = "baseline"
    return {
        "index": index,
        "prompt": baseline["prompt"],
        "baseline_cosine_similarity": float(baseline["cosine_similarity"]),
        "baseline_cosine_distance": baseline_distance,
        "candidate_cosine_similarity": float(candidate["cosine_similarity"]),
        "candidate_cosine_distance": candidate_distance,
        "distance_delta_candidate_minus_baseline": delta,
        "winner": winner,
    }


def write_aggregate_comparison(
    path: Path,
    paired_rows: Sequence[Mapping[str, int | float | str]],
) -> None:
    """Write per-arm distance statistics and paired win counts.

    :param path: Aggregate CSV destination.
    :param paired_rows: Per-prompt paired comparison rows.
    """
    fieldnames: list[str] = ["arm", *SUMMARY_FIELDS, "wins", "ties"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for arm in ("baseline", "candidate"):
            distances = [float(row[f"{arm}_cosine_distance"]) for row in paired_rows]
            writer.writerow(
                {
                    "arm": arm,
                    **summarize_cosine_distances(distances),
                    "wins": sum(row["winner"] == arm for row in paired_rows),
                    "ties": sum(row["winner"] == "tie" for row in paired_rows),
                }
            )


def _render_candidate(*args: str) -> None:
    """Run one isolated CLAP render with streamed output.

    :param *args: Arguments forwarded to ``synth-setter-clap``.
    """
    Command("synth-setter-clap")(*args, _out=sys.stdout, _err=sys.stderr)


def _validate_meanaudio_inverse_model(
    model: VSTFlowMatchingModule,
    render: RenderConfig,
) -> None:
    """Require the four-second MeanAudio conditioning and active renderer width.

    :param model: Loaded candidate inverse model.
    :param render: Render configuration carrying parameter-spec identity.
    :raises ValueError: Conditioning, sketch controls, or output width is incompatible.
    """
    conditioning = resolve_embedding_conditioning(model.hparams["conditioning"])
    if (
        conditioning is None
        or conditioning.column != "meanaudio_16k"
        or conditioning.input_shape != MEANAUDIO_LATENT_SHAPE
    ):
        raise ValueError(
            "candidate inverse checkpoint must use meanaudio_16k conditioning "
            f"with input shape {MEANAUDIO_LATENT_SHAPE}"
        )
    if model.hparams["sketch_controls"] is not None:
        raise ValueError(
            "MeanAudio-only rendering does not support sketch-conditioned checkpoints"
        )
    param_spec_name = render.param_spec_name
    expected_width = len(param_specs[param_spec_name])
    checkpoint_width = model.hparams["num_params"]
    if checkpoint_width != expected_width:
        raise ValueError(
            f"checkpoint output width {checkpoint_width} does not match "
            f"{param_spec_name} width {expected_width}"
        )


def render_meanaudio_candidate(
    prompt: str,
    latent: np.ndarray,
    *,
    checkpoint: str,
    output: Path,
    wav_r2_uri: str,
    device: DeviceSetting,
    seed: int,
) -> dict[str, float | int | str]:
    """Drive one unnormalized MeanAudio latent through inverse, Surge, and CLAP.

    :param prompt: Natural-language target used only for CLAP evaluation.
    :param latent: Direct unnormalized S-Full latent shaped ``(1, 20, 125)``.
    :param checkpoint: Immutable MeanAudio-conditioned inverse checkpoint URI.
    :param output: Candidate WAV destination.
    :param wav_r2_uri: Published WAV identity recorded in the metric row.
    :param device: Torch inference device.
    :param seed: Inverse-flow sampling seed.
    :returns: Candidate CSV row with CLAP cosine similarity and distance.
    :raises RuntimeError: The pinned CLAP checkpoint fails identity validation.
    """
    selected_device = clap_render._resolve_device(device)
    settings = clap_render._load_settings()
    if r2_io.is_r2_uri(checkpoint) or r2_io.is_r2_uri(settings.clap_checkpoint):
        r2_io.ensure_r2_env_loaded()
    clap_checkpoint_dir = resolve_clap_checkpoint(settings.clap_checkpoint)
    if (
        clap_checkpoint_sha256(Path(clap_checkpoint_dir))
        != DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256
    ):
        raise RuntimeError("default CLAP checkpoint SHA-256 mismatch")
    text_embedding = clap_render._encode_text(prompt, clap_checkpoint_dir, selected_device)
    inverse_checkpoint = clap_render.resolve_inverse_checkpoint(checkpoint)
    render = clap_render._workspace_render_config(settings.render)
    model = VSTFlowMatchingModule.load_from_checkpoint(
        inverse_checkpoint,
        map_location=selected_device,
        weights_only=False,
    )
    _validate_meanaudio_inverse_model(model, render)
    model.to(selected_device).eval()
    conditioning = torch.from_numpy(validate_meanaudio_s_full_latent(latent)).to(selected_device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        prediction, _ = model.predict_step({"conditioning": conditioning}, 0)
    audio = clap_render._render_wav(prediction.detach().cpu(), render, output)
    audio_embedding = clap_render._encode_audio(
        audio,
        render.sample_rate,
        clap_checkpoint_dir,
        selected_device,
    )
    comparison = clap_render.compare_embeddings(
        text_embedding.detach().cpu().float().numpy(),
        audio_embedding,
    )
    csv_r2_uri = clap_render._csv_uri_for_wav(wav_r2_uri)
    row: dict[str, float | int | str] = {
        "prompt": prompt,
        "wav_r2_uri": wav_r2_uri,
        "csv_r2_uri": csv_r2_uri,
        "seed": seed,
        "text_embedding_norm": comparison.text_embedding_norm,
        "audio_embedding_norm": comparison.audio_embedding_norm,
        "cosine_similarity": comparison.cosine_similarity,
        "cosine_distance": comparison.cosine_distance,
    }
    clap_render.write_comparison_csv(output.with_suffix(".csv"), row)
    return row


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV as string-keyed rows.

    :param path: CSV source.
    :returns: Rows in source order.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write rows with a stable field order.

    :param path: CSV destination.
    :param fieldnames: Ordered column names.
    :param rows: Rows to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _publish(path: Path, destination: str) -> None:
    """Upload one local artifact to an exact R2 URI.

    :param path: Local source file.
    :param destination: Exact R2 object URI.
    """
    r2_io.upload(path, destination)


def _parse_args() -> argparse.Namespace:
    """Parse comparison sources and destinations.

    :returns: Validated command-line argument namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument(
        "--candidate-source",
        default=CANDIDATE_SOURCE_CLAP,
        choices=(
            CANDIDATE_SOURCE_CLAP,
            CANDIDATE_SOURCE_MEANAUDIO_S_FULL,
            CANDIDATE_SOURCE_MEANAUDIO_S_FULL_REENCODED,
        ),
    )
    parser.add_argument("--baseline-suite", default=BASELINE_SUITE)
    parser.add_argument("--baseline-checkpoint", default=BASELINE_CHECKPOINT)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--meanaudio-steps", type=int, default=MEANAUDIO_STEPS)
    return parser.parse_args()


def main() -> None:
    """Render and publish one paired CLAP checkpoint comparison.

    :raises ValueError: An input URI or baseline manifest is invalid.
    """
    args = _parse_args()
    for value in (
        args.baseline_checkpoint,
        args.baseline_suite,
        args.candidate_checkpoint,
        args.destination,
    ):
        if not r2_io.is_r2_uri(value):
            raise ValueError(f"expected r2:// URI, got {value}")

    if args.meanaudio_steps < 1:
        raise ValueError(f"MeanAudio steps must be positive, got {args.meanaudio_steps}")
    if (
        args.candidate_source
        in {
            CANDIDATE_SOURCE_MEANAUDIO_S_FULL,
            CANDIDATE_SOURCE_MEANAUDIO_S_FULL_REENCODED,
        }
        and args.candidate_checkpoint != DEFAULT_MEANAUDIO_INVERSE_CHECKPOINT
    ):
        raise ValueError(
            "MeanAudio-S-Full comparison requires candidate checkpoint "
            f"{DEFAULT_MEANAUDIO_INVERSE_CHECKPOINT}"
        )

    destination = args.destination.rstrip("/")
    baseline_suite = args.baseline_suite.rstrip("/")
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_identity = build_candidate_identity(
        args.candidate_source,
        args.candidate_checkpoint,
        steps=args.meanaudio_steps,
        duration_seconds=MEANAUDIO_DURATION_SECONDS,
        seed=args.seed,
    )
    identity_path = output_dir / "candidate-identity.json"
    ensure_resume_identity(identity_path, candidate_identity)

    source_manifest = output_dir / "source-manifest.csv"
    r2_io.download_to_path(f"{baseline_suite}/manifest.csv", source_manifest)
    baseline_rows = _read_csv(source_manifest)
    if not baseline_rows:
        raise ValueError("baseline manifest is empty")

    sample_names = []
    for index, baseline in enumerate(baseline_rows, start=1):
        source_stem = Path(baseline["wav_r2_uri"]).stem
        sample_names.append(f"sample_{index:03d}_{source_stem.partition('-')[2]}")

    if args.candidate_source in {
        CANDIDATE_SOURCE_MEANAUDIO_S_FULL,
        CANDIDATE_SOURCE_MEANAUDIO_S_FULL_REENCODED,
    }:
        missing_latents = []
        for baseline, sample_name in zip(baseline_rows, sample_names, strict=True):
            sample_dir = output_dir / "audio" / sample_name
            candidate_complete = (sample_dir / "candidate.wav").is_file() and (
                sample_dir / "candidate.csv"
            ).is_file()
            latent_path = sample_dir / "candidate-conditioning.npy"
            if not candidate_complete and not latent_path.is_file():
                missing_latents.append((baseline["prompt"], latent_path))
            elif latent_path.is_file():
                validate_meanaudio_s_full_latent(np.load(latent_path, allow_pickle=False))
        if missing_latents:
            generator_loader = (
                load_meanaudio_s_full_reencoded_generator
                if args.candidate_source == CANDIDATE_SOURCE_MEANAUDIO_S_FULL_REENCODED
                else load_meanaudio_s_full_generator
            )
            generator = generator_loader(
                steps=args.meanaudio_steps,
                duration_seconds=MEANAUDIO_DURATION_SECONDS,
                device=args.device,
            )
            try:
                for prompt, latent_path in missing_latents:
                    latent_path.parent.mkdir(parents=True, exist_ok=True)
                    staging = latent_path.with_suffix(".npy.tmp")
                    with staging.open("wb") as stream:
                        np.save(stream, generator(prompt, args.seed), allow_pickle=False)
                    staging.replace(latent_path)
                    click.echo(f"Generated MeanAudio latent: {prompt}", err=True)
            finally:
                del generator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    paired_rows: list[dict[str, int | float | str]] = []
    baseline_manifest: list[dict[str, object]] = []
    candidate_manifest: list[dict[str, object]] = []
    prompts: list[str] = []

    for index, (baseline, sample_name) in enumerate(
        zip(baseline_rows, sample_names, strict=True), start=1
    ):
        prompt = baseline["prompt"]
        prompts.append(prompt)
        sample_dir = output_dir / "audio" / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_uri = f"{destination}/audio/{sample_name}"

        baseline_wav_uri = f"{sample_uri}/baseline.wav"
        baseline_csv_uri = f"{sample_uri}/baseline.csv"
        r2_io.upload(baseline["wav_r2_uri"], baseline_wav_uri)
        r2_io.upload(baseline["csv_r2_uri"], baseline_csv_uri)

        candidate_wav = sample_dir / "candidate.wav"
        candidate_csv = sample_dir / "candidate.csv"
        candidate_wav_uri = f"{sample_uri}/candidate.wav"
        candidate_csv_uri = f"{sample_uri}/candidate.csv"
        if not candidate_wav.is_file() or not candidate_csv.is_file():
            candidate_wav.unlink(missing_ok=True)
            candidate_csv.unlink(missing_ok=True)
            if args.candidate_source == CANDIDATE_SOURCE_CLAP:
                _render_candidate(
                    prompt,
                    "--checkpoint",
                    args.candidate_checkpoint,
                    "--output",
                    str(candidate_wav),
                    "--upload-uri",
                    candidate_wav_uri,
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                )
            else:
                latent = np.load(sample_dir / "candidate-conditioning.npy", allow_pickle=False)
                render_meanaudio_candidate(
                    prompt,
                    latent,
                    checkpoint=args.candidate_checkpoint,
                    output=candidate_wav,
                    wav_r2_uri=candidate_wav_uri,
                    device=args.device,
                    seed=args.seed,
                )
                _publish(candidate_wav, candidate_wav_uri)
                _publish(candidate_csv, candidate_csv_uri)
        else:
            _publish(candidate_wav, candidate_wav_uri)
            _publish(candidate_csv, candidate_csv_uri)

        candidate = _read_csv(candidate_csv)[0]
        paired_rows.append(build_paired_row(index, baseline, candidate))
        baseline_manifest.append(
            {
                "index": index,
                **baseline,
                "wav_r2_uri": baseline_wav_uri,
                "csv_r2_uri": baseline_csv_uri,
            }
        )
        candidate_manifest.append({"index": index, **candidate})
        click.echo(f"[{index:03d}/{len(baseline_rows):03d}] {prompt}")

    prompts_path = output_dir / "prompts.txt"
    prompts_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    baseline_manifest_path = output_dir / "baseline_manifest.csv"
    candidate_manifest_path = output_dir / "candidate_manifest.csv"
    paired_path = output_dir / "paired_results.csv"
    aggregate_path = output_dir / "aggregate_comparison.csv"
    provenance_path = output_dir / "comparison_manifest.csv"
    manifest_fields = ("index", *baseline_rows[0].keys())
    _write_csv(baseline_manifest_path, manifest_fields, baseline_manifest)
    _write_csv(candidate_manifest_path, manifest_fields, candidate_manifest)
    _write_csv(paired_path, PAIR_FIELDS, paired_rows)
    write_aggregate_comparison(aggregate_path, paired_rows)
    _write_csv(
        provenance_path,
        ("key", "value"),
        [
            {"key": "created_at", "value": datetime.now(UTC).isoformat()},
            {"key": "baseline_suite", "value": baseline_suite},
            {"key": "baseline_checkpoint", "value": args.baseline_checkpoint},
            {"key": "destination", "value": destination},
            {"key": "prompt_count", "value": len(prompts)},
            *({"key": key, "value": value} for key, value in candidate_identity.items()),
        ],
    )

    for path, relative in (
        (prompts_path, "prompts.txt"),
        (identity_path, "candidate-identity.json"),
        (provenance_path, "comparison_manifest.csv"),
        (baseline_manifest_path, "manifests/baseline.csv"),
        (candidate_manifest_path, "manifests/candidate.csv"),
        (paired_path, "metrics/paired_results.csv"),
        (aggregate_path, "metrics/aggregate_comparison.csv"),
    ):
        _publish(path, f"{destination}/{relative}")
    baseline_summary = summarize_cosine_distances(
        [float(row["baseline_cosine_distance"]) for row in paired_rows]
    )
    candidate_summary = summarize_cosine_distances(
        [float(row["candidate_cosine_distance"]) for row in paired_rows]
    )
    for arm, summary in (("baseline", baseline_summary), ("candidate", candidate_summary)):
        path = output_dir / f"{arm}_aggregate_stats.csv"
        write_summary_csv(path, summary)
        _publish(path, f"{destination}/metrics/{arm}_aggregate_stats.csv")
    click.echo(destination)


if __name__ == "__main__":
    main()
