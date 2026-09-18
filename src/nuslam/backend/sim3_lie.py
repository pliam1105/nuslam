"""Closed-form Sim(3) Lie-group helpers (§14 factor Jacobians).

GTSAM's C++ computes the adjoint and the exp/log Jacobians in closed form (``Similarity3::AdjointMap`` /
``ExpmapDerivative`` / ``LogmapDerivative``), but this build's Python wrapper exposes NONE of them for
``Similarity3`` (nor ``numericalDerivative``). These are the same formulas, re-implemented, so the custom
factors can supply analytic Jacobians without finite differences at runtime.

These are GENERIC Lie-group math -- not factor design. They are the building blocks; compose them into the
actual factor Jacobians in :mod:`.factors`. For the submap-alignment factor
``r = Log(meas^-1 (Hm^-1 Hn))``, ``meas = (I, 0, s_hat)``, ``s_hat = exp((rho_m - rho_n) - log_s)``:

  * the two H columns are a plain BetweenFactor (RIGHT perturbations -> right Jacobian inverse):
    ``H[Hm] = right_jacobian_inv(r) @ (-adjoint(rel.inverse()))``, ``H[Hn] = right_jacobian_inv(r)``;
  * the two rho columns are NOT the same -- rho enters through ``meas`` on the LEFT, as a pure-scale
    increment, so they use the LEFT Jacobian inverse: with ``e_lam = [0,0,0,0,0,0,1]``,
    ``H[rho_m] = -left_jacobian_inv(r) @ e_lam``, ``H[rho_n] = +left_jacobian_inv(r) @ e_lam``.

The anchor Jacobian is the ``z``-row of :func:`point_transform_jacobian` plus the rho column.

Convention (GTSAM ``Similarity3``): the element stores ``H = [[R, t],[0, 1/s]]`` (``H.matrix()``), acts as
``transformFrom(p) = s(Rp + t)``, and the tangent is ``zeta = [omega(3), rho(3), lambda(1)]`` with
``lambda = log s`` (``Similarity3::Logmap``). Derivations + verification: GTSAM handbook Sec 2.3, 6.3.

Every function is checked against GTSAM's own ``Expmap``/``Logmap`` to ~1e-10 in ``tests/test_sim3_lie.py``.
"""
from __future__ import annotations

import numpy as np

_NTERMS = 30   # matrix-series terms; (e^ad - I)/ad is entire, so this is machine-exact for |zeta| < ~pi


def skew(v) -> np.ndarray:
    """3-vector -> 3x3 skew-symmetric ``[v]_x`` (so ``[v]_x w = v x w``)."""
    v = np.asarray(v, float)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def hat(zeta) -> np.ndarray:
    """7-vector ``[omega, rho, lambda]`` -> 4x4 algebra element ``[[ [omega]_x, rho ],[ 0, -lambda ]]``."""
    zeta = np.asarray(zeta, float)
    M = np.zeros((4, 4))
    M[:3, :3] = skew(zeta[:3]); M[:3, 3] = zeta[3:6]; M[3, 3] = -zeta[6]
    return M


def vee(M) -> np.ndarray:
    """Inverse of :func:`hat`: 4x4 algebra element -> 7-vector ``[omega, rho, lambda]``."""
    M = np.asarray(M, float)
    return np.concatenate([[M[2, 1], M[0, 2], M[1, 0]], M[:3, 3], [-M[3, 3]]])


def _srt(H):
    """(s, R, t) from a gtsam.Similarity3 (or pass the tuple through)."""
    if isinstance(H, tuple):
        s, R, t = H
        return float(s), np.asarray(R, float), np.asarray(t, float)
    return float(H.scale()), np.asarray(H.rotation().matrix(), float), np.asarray(H.translation(), float)


def adjoint(H) -> np.ndarray:
    """Ad_H (7x7): ``Ad_H @ zeta = vee(H zeta^ H^-1)`` (and ``H Exp(zeta) H^-1 = Exp(Ad_H zeta)``).

    Closed form in the order ``[omega, rho, lambda]``:
        [[ R,          0,    0   ],
         [ s[t]_x R,   sR,  -s t ],
         [ 0,          0,    1   ]]
    ``H`` is a gtsam.Similarity3 or an ``(s, R, t)`` tuple.
    """
    s, R, t = _srt(H)
    A = np.zeros((7, 7))
    A[:3, :3] = R
    A[3:6, :3] = s * skew(t) @ R; A[3:6, 3:6] = s * R; A[3:6, 6] = -s * t
    A[6, 6] = 1.0
    return A


def little_adjoint(zeta) -> np.ndarray:
    """ad_zeta (7x7): ``ad_zeta @ eta = vee([zeta^, eta^])`` (the Lie bracket; d/dt of Ad at identity).

    Closed form in the order ``[omega, rho, lambda]``:
        [[ [omega]_x,   0,               0     ],
         [ [rho]_x,     [omega]_x + lam I, -rho ],
         [ 0,           0,               0     ]]
    """
    zeta = np.asarray(zeta, float); w, rho, lam = zeta[:3], zeta[3:6], zeta[6]
    A = np.zeros((7, 7))
    A[:3, :3] = skew(w)
    A[3:6, :3] = skew(rho); A[3:6, 3:6] = skew(w) + lam * np.eye(3); A[3:6, 6] = -rho
    return A


def _phi(ad: np.ndarray, nterms: int = _NTERMS) -> np.ndarray:
    """(e^ad - I) ad^-1 = sum_{n>=0} ad^n / (n+1)!  (entire matrix function; the left Jacobian of ad)."""
    J = np.eye(7); term = np.eye(7)
    for n in range(1, nterms):
        term = term @ ad / (n + 1)
        J = J + term
    return J


def left_jacobian(zeta, nterms: int = _NTERMS) -> np.ndarray:
    """J_l(zeta): ``Exp(zeta + d) ~ Exp(J_l d) Exp(zeta)``. = ``sum ad_zeta^n / (n+1)!``."""
    return _phi(little_adjoint(zeta), nterms)


def right_jacobian(zeta, nterms: int = _NTERMS) -> np.ndarray:
    """J_r(zeta): ``Exp(zeta + d) ~ Exp(zeta) Exp(J_r d)``. = ``J_l(-zeta) = sum (-ad_zeta)^n / (n+1)!``."""
    return _phi(-little_adjoint(zeta), nterms)


def left_jacobian_inv(zeta, nterms: int = _NTERMS) -> np.ndarray:
    """J_l(zeta)^-1."""
    return np.linalg.inv(left_jacobian(zeta, nterms))


def right_jacobian_inv(zeta, nterms: int = _NTERMS) -> np.ndarray:
    """J_r(zeta)^-1 -- the Logmap ("Local") derivative: ``d/d(xi) Log(Exp(zeta) Exp(xi))|_0``.

    This is the left factor of GTSAM's BetweenFactor Jacobian: for ``r = Log(meas^-1 (Hm^-1 Hn))``,
    ``H[Hm] = J_r_inv(r) @ (-adjoint(rel.inverse()))`` and ``H[Hn] = J_r_inv(r)``. Equals I at ``r = 0``.
    """
    return np.linalg.inv(right_jacobian(zeta, nterms))


def point_transform_jacobian(H, p) -> np.ndarray:
    """d(H . p)/d(zeta_H) (3x7) in GTSAM's chart -- what ``transformFrom``'s OptionalJacobian<3,7> returns.

    Closed form ``[ -sR[p]_x | sR | +sRp ]`` (note the +sRp: +lambda grows the scale). For an anchor
    residual ``e_z^T (H . p)`` the Jacobian on H is row 2 of this; the point Jacobian ``d(H.p)/dp = sR``.
    """
    s, R, t = _srt(H)
    p = np.asarray(p, float)
    return np.hstack([-s * R @ skew(p), s * R, (s * R @ p).reshape(3, 1)])
