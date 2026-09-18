"""Verify the closed-form Sim(3) Lie helpers against GTSAM's own Expmap/Logmap (ground truth)."""
import numpy as np
import pytest

gtsam = pytest.importorskip("gtsam")
from nuslam.backend import sim3_lie as L  # noqa: E402

Exp, Log = gtsam.Similarity3.Expmap, gtsam.Similarity3.Logmap


def _num(f, dout=7, eps=1e-6):
    """Central-difference a R^7 -> R^dout map at 0."""
    J = np.zeros((dout, 7))
    for k in range(7):
        d = np.zeros(7); d[k] = eps
        J[:, k] = (np.asarray(f(d)) - np.asarray(f(-d))) / (2 * eps)
    return J


def _rand_sim3(rng):
    return gtsam.Similarity3(gtsam.Rot3.Expmap(rng.uniform(-1, 1, 3)),
                             rng.uniform(-4, 4, 3), float(np.exp(rng.uniform(-0.8, 0.8))))


def _rand_zeta(rng):
    return np.concatenate([rng.uniform(-0.8, 0.8, 3), rng.uniform(-3, 3, 3), rng.uniform(-0.6, 0.6, 1)])


@pytest.mark.parametrize("seed", range(5))
def test_hat_vee_roundtrip(seed):
    z = _rand_zeta(np.random.default_rng(seed))
    assert np.allclose(L.vee(L.hat(z)), z, atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_adjoint(seed):
    rng = np.random.default_rng(seed); g = _rand_sim3(rng)
    T, Tinv = np.asarray(g.matrix()), np.asarray(g.inverse().matrix())
    Ad = L.adjoint(g)
    for _ in range(3):
        z = _rand_zeta(rng)
        assert np.allclose(Ad @ z, L.vee(T @ L.hat(z) @ Tinv), atol=1e-9)
    # group identity: g Exp(z) g^-1 == Exp(Ad z)
    z = _rand_zeta(rng)
    lhs = g.compose(Exp(z)).compose(g.inverse())
    assert np.allclose(Log(lhs), Ad @ z, atol=1e-7)


@pytest.mark.parametrize("seed", range(5))
def test_little_adjoint_is_bracket(seed):
    rng = np.random.default_rng(seed); x, e = _rand_zeta(rng), _rand_zeta(rng)
    bracket = L.vee(L.hat(x) @ L.hat(e) - L.hat(e) @ L.hat(x))
    assert np.allclose(L.little_adjoint(x) @ e, bracket, atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_right_jacobian_inv_is_logmap_derivative(seed):
    z = _rand_zeta(np.random.default_rng(seed))
    num = _num(lambda d: Log(Exp(z).compose(Exp(d))))          # d/dxi Log(Exp(z) Exp(xi))
    assert np.allclose(L.right_jacobian_inv(z), num, atol=1e-7)


@pytest.mark.parametrize("seed", range(5))
def test_right_and_left_jacobian(seed):
    z = _rand_zeta(np.random.default_rng(seed))
    Jr_num = _num(lambda d: Log(Exp(z).inverse().compose(Exp(z + d))))   # Exp(z+d)=Exp(z)Exp(Jr d)
    Jl_num = _num(lambda d: Log(Exp(z + d).compose(Exp(z).inverse())))   # Exp(z+d)=Exp(Jl d)Exp(z)
    assert np.allclose(L.right_jacobian(z), Jr_num, atol=1e-7)
    assert np.allclose(L.left_jacobian(z), Jl_num, atol=1e-7)
    assert np.allclose(L.left_jacobian(z) @ L.left_jacobian_inv(z), np.eye(7), atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_point_transform_jacobian(seed):
    rng = np.random.default_rng(seed); g = _rand_sim3(rng); p = rng.uniform(-3, 3, 3)
    num = _num(lambda d: np.asarray(g.retract(d).transformFrom(p)), dout=3)   # d/dzeta (H.p) through retract
    assert np.allclose(L.point_transform_jacobian(g, p), num, atol=1e-6)


def test_between_factor_jacobian_matches_gtsam():
    """The BetweenFactor formula H[Hm] = Jr_inv(r) @ (-Ad(rel^-1)), H[Hn] = Jr_inv(r), far from soln."""
    rng = np.random.default_rng(0); Hm, Hn, meas = _rand_sim3(rng), _rand_sim3(rng), _rand_sim3(rng)
    rel = Hm.between(Hn); r = Log(meas.inverse().compose(rel)); Jri = L.right_jacobian_inv(r)

    def resid(a, b): return Log(meas.inverse().compose(a.between(b)))
    def numjac(fn, X):
        J = np.zeros((7, 7))
        for k in range(7):
            d = np.zeros(7); d[k] = 1e-6
            J[:, k] = (fn(X.retract(d)) - fn(X.retract(-d))) / 2e-6
        return J
    H_Hm = Jri @ (-L.adjoint(rel.inverse())); H_Hn = Jri
    assert np.allclose(H_Hm, numjac(lambda X: resid(X, Hn), Hm), atol=1e-6)
    assert np.allclose(H_Hn, numjac(lambda X: resid(Hm, X), Hn), atol=1e-6)


def test_submap_alignment_rho_jacobians():
    """rho columns use the LEFT Jacobian inverse (meas is on the left), NOT J_r^-1, far from soln."""
    rng = np.random.default_rng(2); Hm, Hn = _rand_sim3(rng), _rand_sim3(rng); log_s = 0.4
    def meas(rm, rn): return gtsam.Similarity3(gtsam.Rot3(), [0., 0, 0], float(np.exp((rm - rn) - log_s)))
    def resid(rm, rn): return Log(meas(rm, rn).inverse().compose(Hm.between(Hn)))
    rm, rn, eps = 0.7, -0.3, 1e-6
    r = resid(rm, rn); e = np.zeros(7); e[6] = 1.0
    Jli = L.left_jacobian_inv(r)
    assert np.abs(r).max() > 0.5   # genuinely far from the solution
    drm = (resid(rm + eps, rn) - resid(rm - eps, rn)) / (2 * eps)
    drn = (resid(rm, rn + eps) - resid(rm, rn - eps)) / (2 * eps)
    assert np.allclose(-Jli @ e, drm, atol=1e-6)
    assert np.allclose(Jli @ e, drn, atol=1e-6)
