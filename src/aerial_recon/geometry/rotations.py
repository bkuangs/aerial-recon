"""M1 — Rotations and the Lie group SO(3).

Bundle adjustment optimizes rotations, but a 3x3 matrix has 9 numbers and only 3 degrees
of freedom. The standard fix is to optimize a 3-vector w (axis * angle) and map it to a
rotation with the exponential map. Every later milestone depends on these being right.

Reading: Szeliski 2e §2.1.3–2.1.4; Barfoot "State Estimation for Robotics" ch. 7;
Solà et al. "A micro Lie theory for state estimation in robotics" (2018), §I–III.
"""

from __future__ import annotations

import numpy as np


def hat(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [w]x such that hat(w) @ v == np.cross(w, v)."""
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0],
    ])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: axis-angle vector w (3,) -> rotation matrix (3, 3).

    R = I + sin(θ)/θ [w]x + (1 - cos θ)/θ² [w]x², θ = |w|.
    Must be numerically stable as θ -> 0 (use a Taylor expansion below ~1e-8).
    """
    theta = np.linalg.norm(w)
    K = hat(w)

    if theta < 1e-8:
        A = 1 - (theta ** 2) / 6
        B = 1/2 - (theta ** 2) / 24
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta ** 2)

    return np.eye(3) + A * K + B * (K @ K)  # np.eye() -> identity matrix


def so3_log(R: np.ndarray) -> np.ndarray:  # noqa: N803
    """Inverse of so3_exp: rotation (3, 3) -> axis-angle vector (3,) with |w| in [0, π].

    Careful near θ = 0 (you divide by sin θ) and near θ = π, where the antisymmetric part
    R - Rᵀ vanishes: recover the axis from the symmetric part (R + Rᵀ)/2 = cosθ I +
    (1 - cosθ) a aᵀ and take its sign from the (tiny) antisymmetric part. Tests go to
    within 1e-7 rad of π.
    """
    cos_t = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)   # tr(R) = 1 + 2cos(θ)
    theta = np.arccos(cos_t)
    # vee(R - Rᵀ) = 2 sin(θ) a
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if theta < 1e-8:
        return (0.5 + theta**2 / 12) * v

    if np.pi - theta < 1e-3:
        S = (R + R.T) / 2
        aaT = (S - cos_t * np.eye(3)) / (1 - cos_t)
        i = np.argmax(np.diag(aaT))
        a = aaT[:, i] / np.linalg.norm(aaT[:, i])
        if a @ v < 0:
            a = -a
        return theta * a

    return theta / (2 * np.sin(theta)) * v


def project_to_so3(M: np.ndarray) -> np.ndarray:  # noqa: N803
    """Closest rotation to an arbitrary 3x3 matrix in Frobenius norm (via SVD, det = +1).

    You will need this after every linear solver (DLT, 8-point, PnP) returns an
    "almost rotation".
    """
    U, _, Vt = np.linalg.svd(M)

    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1

    return U @ Vt


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:  # noqa: N803
    """Geodesic distance between two rotations in degrees: angle of R_aᵀ R_b.

    Clip the arccos argument to [-1, 1]; floating point will betray you otherwise.
    """
    R = R_a.T @ R_b     # relative rotation
    cos_t = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)

    theta = np.degrees(np.arccos(cos_t))

    return float(theta)
