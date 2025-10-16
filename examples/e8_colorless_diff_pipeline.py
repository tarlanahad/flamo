
"""Pipeline for two-stage FDN optimization.

This script first trains the colorless FDN configuration to learn the input, output,
and feedback mixing gains. The learned parameters are then frozen and reused to
warm-start a DiffFDN configuration whose attenuation filter (parameterized in
reverberation times) is optimized against a target room impulse response.

Example
-------
To run both stages against ``rirs/s3_r4_o.wav`` while seeding the attenuation
with a measured RT60 profile spanning octave bands from 31.25 Hz to 16 kHz::

    python examples/e8_colorless_diff_pipeline.py rirs/s3_r4_o.wav \
        --device cpu \
        --band_start_hz 31.25 --band_end_hz 16000 --band_octave_interval 1 \
        --initial_rt 2.405 2.596 2.775 2.524 2.376 2.364 2.12 1.782 1.215 0.673

By default the script clones DecayFitNet, estimates octave-band RT60 and initial
level ("L") figures from the reference RIR, and uses the RT profile to seed the
DiffFDN attenuation. Pass ``--skip_decayfitnet`` to disable that behaviour or
``--initial_rt`` to override the automatically recovered RT60 values.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import auraloss
import numpy as np
import scipy.io
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from flamo.optimize.dataset import Dataset, DatasetColorless, load_dataset
from flamo.optimize.loss import mse_loss, sparsity_loss
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.utils import save_audio
from flamo.functional import find_onset, signal_gallery
from flamo.auxiliary.reverb import parallelFDNAccurateGEQ


class MultiResoSTFT(nn.Module):
    """Wrapper around auraloss multi-resolution STFT loss."""

    def __init__(self) -> None:
        super().__init__()
        self.loss = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, rir_a: torch.Tensor, rir_b: torch.Tensor) -> torch.Tensor:
        return self.loss(rir_a.permute(0, 2, 1), rir_b.permute(0, 2, 1))


# ---------------------------------------------------------------------------
# DecayFitNet integration helpers
# ---------------------------------------------------------------------------

DECAYFITNET_REPO_URL = "https://github.com/georg-goetz/DecayFitNet"


def ensure_decayfitnet_repo(repo_path: Path, branch: Optional[str] = None) -> Path:
    """Clone the DecayFitNet repository if it is not already present."""

    repo_path = repo_path.expanduser()
    if repo_path.exists():
        return repo_path

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    clone_cmd = ["git", "clone", "--depth", "1"]
    if branch:
        clone_cmd.extend(["--branch", branch])
    clone_cmd.extend([DECAYFITNET_REPO_URL, str(repo_path)])
    print(f"Cloning DecayFitNet into {repo_path} ...")
    subprocess.run(clone_cmd, check=True)
    return repo_path


def perform_decayfitnet_analysis(
    repo_dir: Path,
    rir: np.ndarray,
    samplerate: int,
    band_frequencies: List[float],
    n_slopes: int,
    device: str,
) -> Dict[str, np.ndarray]:
    """Estimate RT60 and initial level parameters using DecayFitNet."""

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from DecayFitNet.python.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
    from DecayFitNet.python.toolbox.utils import calc_mse
    from DecayFitNet.python.toolbox.core import (
        PreprocessRIR,
        decay_model,
        discard_last_n_percent,
    )
    from filters.utils import octave_filtering, db2mag, rt2slope

    rir_preprocessing = PreprocessRIR(
        sample_rate=samplerate, filter_frequencies=band_frequencies
    )

    true_edc, _ = rir_preprocessing.schroeder(rir, analyse_full_rir=True)
    time_axis = torch.linspace(0, true_edc.shape[2] - 1, true_edc.shape[2]) / samplerate

    true_edc = true_edc.permute(1, 0, 2)

    decayfitnet = DecayFitNetToolbox(
        n_slopes=n_slopes,
        sample_rate=samplerate,
        filter_frequencies=band_frequencies,
    )
    estimated_parameters, norm_vals = decayfitnet.estimate_parameters(
        rir, analyse_full_rir=True
    )

    def _to_numpy_2d(name: str, value) -> np.ndarray:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            return array.reshape(1, 1)
        if array.ndim == 1:
            return array.reshape(-1, 1)
        if array.ndim == 2:
            return array
        raise ValueError(
            f"Expected DecayFitNet '{name}' output to be at most 2-D, got shape {array.shape}."
        )

    def _flatten_if_singleton(array: np.ndarray) -> np.ndarray:
        if array.ndim == 2 and array.shape[1] == 1:
            return array[:, 0]
        return array

    norm_vals = _to_numpy_2d("norm", norm_vals)
    t60 = _to_numpy_2d("rt60", estimated_parameters[0])
    amplitude = _to_numpy_2d("amplitude", estimated_parameters[1])
    noise_floor = _to_numpy_2d("noise_floor", estimated_parameters[2])

    if norm_vals.shape != amplitude.shape:
        if norm_vals.T.shape == amplitude.shape:
            norm_vals = norm_vals.T
        else:
            raise ValueError(
                "DecayFitNet returned normalisation factors with shape "
                f"{norm_vals.shape} that are incompatible with amplitude shape {amplitude.shape}."
            )

    fitted_edc = decay_model(
        torch.from_numpy(t60).to(device),
        torch.from_numpy(amplitude).to(device),
        torch.from_numpy(noise_floor).to(device),
        time_axis=time_axis.to(device),
        compensate_uli=True,
        backend="torch",
        device=device,
    )

    true_edc = discard_last_n_percent(true_edc, 5)
    fitted_edc = discard_last_n_percent(fitted_edc, 5)
    mse_per_band = calc_mse(true_edc, fitted_edc)

    amplitude = amplitude * norm_vals
    noise_floor = noise_floor * norm_vals

    impulse = np.zeros((samplerate, 1))
    impulse[0] = 1
    rir_bands = np.asarray(octave_filtering(impulse, samplerate, band_frequencies))
    if rir_bands.ndim == 3 and rir_bands.shape[0] == 1:
        rir_bands = np.squeeze(rir_bands, axis=0)
    if rir_bands.ndim == 2 and rir_bands.shape[0] == len(band_frequencies):
        energy_axis = 1
    else:
        energy_axis = 0
    band_energy = np.sum(rir_bands**2, axis=energy_axis)

    t60 = _flatten_if_singleton(t60)
    amplitude = _flatten_if_singleton(amplitude)
    noise_floor = _flatten_if_singleton(noise_floor)
    norm_vals = _flatten_if_singleton(norm_vals)

    gain_per_sample = db2mag(rt2slope(t60, samplerate))
    decay_energy = 1 / (1 - gain_per_sample**2)
    level = np.sqrt(amplitude / band_energy / decay_energy * len(rir))

    return {
        "rt60": t60,
        "amplitude": amplitude,
        "noise_floor": noise_floor,
        "norm": norm_vals,
        "initial_level": level,
        "mse": _flatten_if_singleton(np.asarray(mse_per_band, dtype=np.float64)),
    }

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def build_colorless_shell(
    nfft: int,
    samplerate: int,
    delay_lengths: torch.Tensor,
    alias_decay_db: float,
    device: torch.device,
) -> system.Shell:
    """Instantiate the colorless FDN shell used for the warm-start stage."""

    delay_lengths = delay_lengths.to(device=device)
    max_delay = int(delay_lengths.max().item())

    n_delays = delay_lengths.numel()

    input_gain = dsp.Gain(
        size=(n_delays, 1),
        nfft=nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    output_gain = dsp.Gain(
        size=(1, n_delays),
        nfft=nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=device,
    )

    delays = dsp.parallelDelay(
        size=(n_delays,),
        max_len=max_delay,
        nfft=nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    delays.assign_value(delays.sample2s(delay_lengths))

    feedback_matrix = dsp.Matrix(
        size=(n_delays, n_delays),
        nfft=nfft,
        matrix_type="orthogonal",
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=device,
    )

    feedback_loop = system.Recursion(fF=delays, fB=feedback_matrix)
    fdn = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )

    input_layer = dsp.FFT(nfft)
    output_layer = dsp.Transform(transform=lambda x: torch.abs(x))
    return system.Shell(core=fdn, input_layer=input_layer, output_layer=output_layer)


def build_diff_shell(
    nfft: int,
    samplerate: int,
    delay_lengths: torch.Tensor,
    alias_decay_db: float,
    octave_interval: int,
    start_freq: float,
    end_freq: float,
    device: torch.device,
) -> system.Shell:
    """Instantiate the DiffFDN shell that includes the RT-controlled attenuation."""

    delay_lengths = delay_lengths.to(device=device)
    max_delay = int(delay_lengths.max().item())

    n_delays = delay_lengths.numel()

    input_gain = dsp.Gain(
        size=(n_delays, 1),
        nfft=nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    output_gain = dsp.Gain(
        size=(1, n_delays),
        nfft=nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=device,
    )

    delays = dsp.parallelDelay(
        size=(n_delays,),
        max_len=max_delay,
        nfft=nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    delays.assign_value(delays.sample2s(delay_lengths))

    mixing_matrix = dsp.Matrix(
        size=(n_delays, n_delays),
        nfft=nfft,
        matrix_type="orthogonal",
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    attenuation = parallelFDNAccurateGEQ(
        octave_interval=octave_interval,
        nfft=nfft,
        fs=samplerate,
        delays=delay_lengths,
        alias_decay_db=alias_decay_db,
        start_freq=start_freq,
        end_freq=end_freq,
        device=device,
    )

    feedback = system.Series(
        OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
    )

    feedback_loop = system.Recursion(fF=delays, fB=feedback)
    fdn = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )

    input_layer = dsp.FFT(nfft)
    output_layer = dsp.iFFTAntiAlias(
        nfft=nfft, alias_decay_db=alias_decay_db, device=device
    )
    return system.Shell(core=fdn, input_layer=input_layer, output_layer=output_layer)


def export_fdn_params(shell: system.Shell, filepath: str) -> Dict[str, np.ndarray]:
    """Dump key FDN tensors to a MAT file for later reuse."""

    core = shell.get_core()
    params: Dict[str, np.ndarray] = {}

    feedback_module = core.feedback_loop.feedback
    if hasattr(feedback_module, "mixing_matrix"):
        feedback_tensor = feedback_module.mixing_matrix.param
    else:
        feedback_tensor = feedback_module.param

    params["A"] = feedback_tensor.detach().cpu().numpy()
    params["B"] = core.input_gain.param.detach().cpu().numpy()
    params["C"] = core.output_gain.param.detach().cpu().numpy()
    feedforward = core.feedback_loop.feedforward
    params["m"] = (
        feedforward.s2sample(feedforward.map(feedforward.param))
        .detach()
        .cpu()
        .numpy()
    )

    scipy.io.savemat(filepath, params)
    return params


def write_band_summary(attenuation: dsp.Filter, directory: str) -> None:
    """Save and print the attenuation band's centre frequencies."""

    bands = attenuation.center_freq.detach().cpu().numpy()
    summary_path = os.path.join(directory, "attenuation_bands_hz.txt")
    np.savetxt(summary_path, bands, fmt="%.2f")
    print("Attenuation bands (Hz):", ", ".join(f"{freq:.2f}" for freq in bands))
    print(f"Saved band summary to {summary_path}")


def freeze_module(module: nn.Module) -> None:
    """Disable gradient tracking for every parameter inside the module."""

    module.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    """Enable gradient tracking for every parameter inside the module."""

    module.requires_grad_(True)


# ---------------------------------------------------------------------------
# Stage 1 - colorless optimization
# ---------------------------------------------------------------------------

def run_colorless_stage(args, delay_lengths: torch.Tensor, alias_decay_db: float):
    stage_dir = os.path.join(args.output_dir, "stage1_colorless")
    os.makedirs(stage_dir, exist_ok=True)

    model = build_colorless_shell(
        nfft=args.nfft,
        samplerate=args.samplerate,
        delay_lengths=delay_lengths,
        alias_decay_db=alias_decay_db,
        device=args.device,
    )

    with torch.no_grad():
        ir_init = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(stage_dir, "ir_init.wav"),
            ir_init / torch.max(torch.abs(ir_init)),
            fs=args.samplerate,
        )

    dataset = DatasetColorless(
        input_shape=(1, args.nfft // 2 + 1, 1),
        target_shape=(1, args.nfft // 2 + 1, 1),
        expand=args.colorless_dataset_size,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(
        dataset, batch_size=args.batch_size
    )

    trainer = Trainer(
        model,
        max_epochs=args.colorless_epochs,
        lr=args.colorless_lr,
        train_dir=stage_dir,
        device=args.device,
    )
    trainer.register_criterion(mse_loss(nfft=args.nfft, device=args.device), 1.0)
    trainer.register_criterion(
        sparsity_loss(), args.colorless_sparsity_weight, requires_model=True
    )

    trainer.train(train_loader, valid_loader)

    with torch.no_grad():
        ir_optim = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(stage_dir, "ir_optim.wav"),
            ir_optim / torch.max(torch.abs(ir_optim)),
            fs=args.samplerate,
        )

    params = export_fdn_params(
        model, os.path.join(stage_dir, "colorless_fdn_params.mat")
    )
    return params, stage_dir


# ---------------------------------------------------------------------------
# Stage 2 - DiffFDN attenuation fitting
# ---------------------------------------------------------------------------

def prepare_target_rir(args) -> torch.Tensor:
    """Load, normalise, and window the target RIR to match the FFT size."""

    waveform, _ = sf.read(args.target_rir)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    target = torch.tensor(waveform, dtype=torch.float32)
    peak = torch.max(torch.abs(target))
    if peak > 0:
        target = target / peak
    onset = find_onset(target)
    trimmed = target[onset: onset + args.nfft]
    if trimmed.numel() < args.nfft:
        trimmed = F.pad(trimmed, (0, args.nfft - trimmed.numel()))
    return trimmed.view(1, -1, 1)


def run_diff_stage(
    args,
    delay_lengths: torch.Tensor,
    alias_decay_db: float,
    colorless_params: Dict[str, np.ndarray],
):
    stage_dir = os.path.join(args.output_dir, "stage2_diff")
    os.makedirs(stage_dir, exist_ok=True)

    model = build_diff_shell(
        nfft=args.nfft,
        samplerate=args.samplerate,
        delay_lengths=delay_lengths,
        alias_decay_db=alias_decay_db,
        octave_interval=args.band_octave_interval,
        start_freq=args.band_start_hz,
        end_freq=args.band_end_hz,
        device=args.device,
    )

    core = model.get_core()

    # ------------------------------------------------------------------
    # Step 2.1 - load and freeze colorless parameters
    # ------------------------------------------------------------------
    core.input_gain.assign_value(
        torch.from_numpy(colorless_params["B"]).to(core.input_gain.param)
    )
    core.output_gain.assign_value(
        torch.from_numpy(colorless_params["C"]).to(core.output_gain.param)
    )

    mixing_matrix = core.feedback_loop.feedback.mixing_matrix
    mixing_matrix.assign_value(
        torch.from_numpy(colorless_params["A"]).to(mixing_matrix.param)
    )

    freeze_module(core.input_gain)
    freeze_module(core.output_gain)
    freeze_module(mixing_matrix)

    # ------------------------------------------------------------------
    # Step 2.2 - define/inspect attenuation bands and optionally seed their RT profile
    # ------------------------------------------------------------------
    attenuation = core.feedback_loop.feedback.attenuation
    band_frequencies = (
        attenuation.center_freq.detach().cpu().numpy().astype(np.float64)
    )

    decayfitnet_results: Optional[Dict[str, np.ndarray]] = None
    if not args.skip_decayfitnet:
        repo_dir = ensure_decayfitnet_repo(
            Path(args.decayfitnet_path), args.decayfitnet_branch
        )
        waveform, waveform_sr = sf.read(args.target_rir)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if waveform_sr != args.samplerate:
            raise ValueError(
                "DecayFitNet analysis expects the RIR sample rate to match --samplerate"
            )
        waveform = waveform.astype(np.float64)
        decayfitnet_results = perform_decayfitnet_analysis(
            repo_dir=repo_dir,
            rir=waveform,
            samplerate=waveform_sr,
            band_frequencies=band_frequencies.tolist(),
            n_slopes=args.decayfitnet_n_slopes,
            device=args.decayfitnet_device,
        )

        np.savetxt(
            os.path.join(stage_dir, "decayfitnet_rt60_seconds.txt"),
            decayfitnet_results["rt60"],
            fmt="%.6f",
        )
        np.savetxt(
            os.path.join(stage_dir, "decayfitnet_initial_level.txt"),
            decayfitnet_results["initial_level"],
            fmt="%.6f",
        )
        np.savez(
            os.path.join(stage_dir, "decayfitnet_full_analysis.npz"),
            **decayfitnet_results,
            band_frequencies=band_frequencies,
        )

        print(
            "DecayFitNet RT60 (s):",
            ", ".join(f"{val:.3f}" for val in decayfitnet_results["rt60"]),
        )
        print(
            "DecayFitNet initial levels:",
            ", ".join(f"{val:.3f}" for val in decayfitnet_results["initial_level"]),
        )

    write_band_summary(attenuation, stage_dir)
    param_count = attenuation.param.numel()
    band_count = attenuation.center_freq.numel()
    print(
        "Attenuation parameter count (include low/high shelves):",
        param_count,
    )

    rt_seed_values: Optional[np.ndarray] = None
    if args.initial_rt:
        rt_seed_values = np.asarray(args.initial_rt, dtype=np.float64)
    elif decayfitnet_results is not None:
        print("Seeding attenuation with DecayFitNet RT60 profile.")
        rt_seed_values = decayfitnet_results["rt60"]

    if rt_seed_values is not None:
        init_rt = torch.tensor(rt_seed_values, dtype=attenuation.param.dtype)
        if init_rt.numel() == band_count:
            print(
                "RT profile matches centre-band count; padding low/high shelves",
                "with edge values.",
            )
            padded_rt = torch.empty(param_count, dtype=init_rt.dtype)
            padded_rt[0] = init_rt[0]
            padded_rt[1:-1] = init_rt
            padded_rt[-1] = init_rt[-1]
            init_rt = padded_rt
        elif init_rt.numel() == band_count + 1:
            if args.band_start_hz < float(attenuation.center_freq[0]):
                print(
                    "Interpreting RT profile as including the low-shelf band;",
                    "duplicating the final centre-band value for the high shelf.",
                )
                padded_rt = torch.empty(param_count, dtype=init_rt.dtype)
                padded_rt[0] = init_rt[0]
                padded_rt[1:-1] = init_rt[1:]
                padded_rt[-1] = init_rt[-1]
                init_rt = padded_rt
            else:
                print(
                    "Interpreting RT profile as including the high-shelf band;",
                    "duplicating the first centre-band value for the low shelf.",
                )
                padded_rt = torch.empty(param_count, dtype=init_rt.dtype)
                padded_rt[0] = init_rt[0]
                padded_rt[1:-1] = init_rt[:-1]
                padded_rt[-1] = init_rt[-1]
                init_rt = padded_rt
        elif init_rt.numel() != param_count:
            raise ValueError(
                "RT profile length must match either the centre-band count, "
                "that count plus one (one shelf provided), or the attenuation parameter "
                f"count ({param_count})."
            )
        attenuation.assign_value(init_rt.to(attenuation.param))

    with torch.no_grad():
        ir_init = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(stage_dir, "ir_init.wav"),
            ir_init / torch.max(torch.abs(ir_init)),
            fs=args.samplerate,
        )

    # ------------------------------------------------------------------
    # Step 2.3 - dataset and trainer setup
    # ------------------------------------------------------------------
    excitation = signal_gallery(
        1,
        n_samples=args.nfft,
        n=1,
        signal_type="impulse",
        fs=args.samplerate,
        device=args.device,
    )
    target_rir = prepare_target_rir(args).to(args.device)

    dataset = Dataset(
        input=excitation,
        target=target_rir,
        expand=args.diff_dataset_size,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(
        dataset, batch_size=args.batch_size
    )

    trainer = Trainer(
        model,
        max_epochs=args.diff_epochs,
        lr=args.diff_lr,
        train_dir=stage_dir,
        device=args.device,
    )
    trainer.register_criterion(MultiResoSTFT(), 1.0)
    trainer.register_criterion(
        sparsity_loss(), args.diff_sparsity_weight, requires_model=True
    )
    trainer.train(train_loader, valid_loader)

    # ------------------------------------------------------------------
    # Step 2.4 - optional fine-tuning of frozen modules
    # ------------------------------------------------------------------
    if args.finetune_epochs > 0:
        unfreeze_module(core.input_gain)
        unfreeze_module(core.output_gain)
        unfreeze_module(mixing_matrix)

        finetune_dir = os.path.join(stage_dir, "finetune")
        os.makedirs(finetune_dir, exist_ok=True)

        finetune_trainer = Trainer(
            model,
            max_epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            train_dir=finetune_dir,
            device=args.device,
        )
        finetune_trainer.register_criterion(MultiResoSTFT(), 1.0)
        finetune_trainer.register_criterion(
            sparsity_loss(), args.diff_sparsity_weight, requires_model=True
        )
        finetune_trainer.train(train_loader, valid_loader)

    with torch.no_grad():
        ir_final = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(stage_dir, "ir_optim.wav"),
            ir_final / torch.max(torch.abs(ir_final)),
            fs=args.samplerate,
        )

    export_fdn_params(model, os.path.join(stage_dir, "diff_fdn_params.mat"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage FDN optimisation pipeline")

    parser.add_argument("target_rir", type=str, help="Path to the reference RIR wav file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("output", time.strftime("%Y%m%d-%H%M%S")),
        help="Directory to store training artefacts",
    )
    parser.add_argument("--nfft", type=int, default=96000, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="Sampling rate")
    parser.add_argument(
        "--delay_lengths",
        type=int,
        nargs="+",
        default=[887, 911, 941, 1699, 1951, 2053],
        help="Integer delay lengths for the FDN",
    )
    parser.add_argument(
        "--alias_decay_db", type=float, default=30.0, help="Alias decay in dB"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Computation device")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for both stages")
    parser.add_argument("--seed", type=int, default=130709, help="Random seed")

    # Stage 1
    parser.add_argument(
        "--colorless_epochs", type=int, default=10, help="Training epochs for stage 1"
    )
    parser.add_argument(
        "--colorless_lr", type=float, default=1e-3, help="Learning rate for stage 1"
    )
    parser.add_argument(
        "--colorless_dataset_size",
        type=int,
        default=2**8,
        help="Synthetic dataset expansion factor for stage 1",
    )
    parser.add_argument(
        "--colorless_sparsity_weight",
        type=float,
        default=0.2,
        help="Weight of the sparsity loss during stage 1",
    )

    # Stage 2
    parser.add_argument(
        "--diff_epochs", type=int, default=50, help="Training epochs for stage 2"
    )
    parser.add_argument(
        "--diff_lr", type=float, default=1e-4, help="Learning rate for stage 2"
    )
    parser.add_argument(
        "--diff_dataset_size",
        type=int,
        default=2**4,
        help="Dataset expansion factor for stage 2",
    )
    parser.add_argument(
        "--diff_sparsity_weight",
        type=float,
        default=1.0,
        help="Weight of the sparsity loss during stage 2",
    )
    parser.add_argument(
        "--band_octave_interval",
        type=int,
        default=1,
        help="Octave spacing for the attenuation bands (1=octave, 3=third-octave, ...)",
    )
    parser.add_argument(
        "--band_start_hz",
        type=float,
        default=31.25,
        help="Lowest centre frequency for the attenuation filterbank",
    )
    parser.add_argument(
        "--band_end_hz",
        type=float,
        default=16000.0,
        help="Highest centre frequency for the attenuation filterbank",
    )
    parser.add_argument(
        "--initial_rt",
        type=float,
        nargs="*",
        default=None,
        help="Optional initial RT profile (seconds) for the attenuation bands (match the band count reported in attenuation_bands_hz.txt)",
    )
    parser.add_argument(
        "--skip_decayfitnet",
        action="store_true",
        help="Disable automatic RT/L estimation with DecayFitNet",
    )
    parser.add_argument(
        "--decayfitnet_path",
        type=str,
        default=os.path.join( "DecayFitNet"),
        help="Location of the DecayFitNet repository (cloned automatically if missing)",
    )
    parser.add_argument(
        "--decayfitnet_branch",
        type=str,
        default="main",
        help="Git branch to checkout when cloning DecayFitNet",
    )
    parser.add_argument(
        "--decayfitnet_n_slopes",
        type=int,
        default=1,
        help="Number of decay slopes used by DecayFitNet",
    )
    parser.add_argument(
        "--decayfitnet_device",
        type=str,
        default="cpu",
        help="Device string passed to DecayFitNet when synthesising EDC fits",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=0,
        help="Optional fine-tune epochs with unfrozen gains",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=5e-5,
        help="Learning rate used during the optional fine-tuning stage",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    args.device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.txt"), "w", encoding="utf-8") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key},{value}\n")

    delay_lengths = torch.tensor(args.delay_lengths)

    colorless_params, _ = run_colorless_stage(
        args, delay_lengths=delay_lengths, alias_decay_db=args.alias_decay_db
    )
    run_diff_stage(
        args,
        delay_lengths=delay_lengths,
        alias_decay_db=args.alias_decay_db,
        colorless_params=colorless_params,
    )


if __name__ == "__main__":
    main()
