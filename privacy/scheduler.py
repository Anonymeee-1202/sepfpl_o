# -*- coding: utf-8 -*-
"""
Simplified RDP budget scheduler for SepFPL (independent module).
- Tracks per-client remaining RDP budget ε_{i,α}^{rem}
- Chooses η_i^t by distributing remaining budget uniformly over remaining rounds
- Applies simple subsampled Gaussian upper bound: ε_α ≈ (ρ^2 * α) / (2 * η^2)
- Computes harmonic mean η^t and per-client clipping C_i^t = C_avg * η^t / η_i^t
- Returns Poisson-sampled client set according to ρ_i^t

Notes:
- This is a pragmatic simplified bound to keep implementation dependency-free.
- Only used when factorization == 'sepfpl'. Other algorithms remain unaffected.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class ClientRDPState:
    # Remaining RDP budget at order alpha
    eps_rem_alpha: float


def per_round_rdp_gaussian(alpha: float, rho: float, eta: float) -> float:
    """Simplified RDP per-round cost for subsampled Gaussian.
    ε_α ≈ (ρ^2 * α) / (2 * η^2)
    """
    if eta <= 0:
        return float('inf')
    return (rho * rho) * (alpha) / (2.0 * eta * eta)


def compute_eta_from_target(alpha: float, rho: float, eps_target_alpha: float, min_eta: float = 1e-3) -> float:
    """Invert simplified bound to get minimal eta to meet per-round target RDP cost."""
    if eps_target_alpha <= 0:
        return 1.0
    eta_sq = (rho * rho) * alpha / (2.0 * eps_target_alpha)
    eta = float(np.sqrt(max(eta_sq, min_eta * min_eta)))
    return eta


def compute_round_privacy_params(
    num_users: int,
    round_idx: int,
    total_rounds: int,
    rho_base: float,
    tau: int,
    rho_conserve: float,
    C_avg: float,
    alpha: float,
    client_states: List[ClientRDPState],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[int]]:
    """
    Returns (rho_i, C_i, eta_i, eta_hmean, sampled_users)
    - rho schedule: rho_conserve before tau; rho_base after tau
    - target per-round budget: eps_rem / remaining_rounds
    - eta_i: minimal to satisfy per-round target using simplified bound
    - eta^t: harmonic mean; C_i = C_avg * eta^t / eta_i
    - sampled_users: Poisson sampling with rho_i
    """
    if tau is None or tau < 0:
        tau = max(1, total_rounds // 2)
    rho_i = np.full(num_users, rho_base, dtype=np.float32)
    if round_idx < tau:
        rho_i = np.full(num_users, min(rho_base, rho_conserve), dtype=np.float32)

    # remaining rounds including current
    remaining = max(1, total_rounds - round_idx)

    eta_i = np.zeros(num_users, dtype=np.float32)
    for i in range(num_users):
        eps_rem = max(0.0, float(client_states[i].eps_rem_alpha))
        eps_target = eps_rem / float(remaining)
        eta_i[i] = compute_eta_from_target(alpha=alpha, rho=float(rho_i[i]), eps_target_alpha=eps_target)

    # harmonic mean eta^t
    inv = np.where(eta_i > 0, 1.0 / eta_i, 0.0)
    eta_hmean = 1.0 / float(np.mean(inv[inv > 0])) if np.any(inv > 0) else float(np.mean(eta_i) if np.mean(eta_i) > 0 else 1.0)

    # per-client clipping
    C_i = (C_avg * (eta_hmean / np.maximum(eta_i, 1e-6))).astype(np.float32)

    # Poisson subsampling
    sampled = [i for i in range(num_users) if np.random.rand() < rho_i[i]]
    if len(sampled) == 0:
        sampled = [int(np.random.randint(0, num_users))]

    return rho_i, C_i, eta_i, float(eta_hmean), sampled


def update_rdp_states_after_round(
    sampled_users: List[int],
    alpha: float,
    rho_i: np.ndarray,
    eta_i: np.ndarray,
    client_states: List[ClientRDPState],
) -> None:
    """Subtract the consumed per-round RDP cost from remaining budgets for sampled clients."""
    for i in sampled_users:
        cost = per_round_rdp_gaussian(alpha=alpha, rho=float(rho_i[i]), eta=float(eta_i[i]))
        client_states[i].eps_rem_alpha = max(0.0, float(client_states[i].eps_rem_alpha) - cost)
