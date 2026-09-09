"""Regression tests for Route-B scale resolution (lever-arm Umeyama).

The synthetic scene below has a KNOWN scale, so the resolver must recover it: a
reconstruction is built in "recon units", a lever arm and a per-frame orientation are
applied, the GPS ground track is generated at the true metric scale, and
``resolve_scale_gps`` must return that scale. See ``src/nuslam/recon/scale.py`` and the
Umeyama handout.
"""
import numpy as np
import pytest

from nuslam.recon import ScaleResult, resolve_scale_gps


def _synthetic_scene(n=40, true_scale=17.0, seed=0):
    """Build (world_from_cam, gps_xy, sensor2ego, valid) with a known ``true_scale``.

    The camera drives a gentle turning arc. Ground truth is *not* used by the resolver;
    it is only how this fixture manufactures a consistent GPS track at metric scale.
    """
    rng = np.random.default_rng(seed)
    # A turning heading so per-frame orientation varies (else the lever arm is constant
    # and the problem degenerates to plain Umeyama -- see handout section 7).
    th = np.linspace(0, 1.1, n)
    # Ego ground track in the map frame (metres), on the z=0 plane.
    ego_xy = np.stack([30 * np.sin(th), 30 * (1 - np.cos(th))], axis=1)
    gps_xy = ego_xy + rng.normal(scale=0.05, size=ego_xy.shape)  # small GPS noise

    # Camera->ego extrinsic: 1.5 m up, 1.7 m forward, camera looking along +x (RDF-ish).
    sensor2ego = np.eye(4)
    sensor2ego[:3, 3] = [1.7, 0.0, 1.5]

    # Build est camera->world in RECON units (metric / true_scale) so the resolver must
    # recover ``true_scale``. Camera centre = ego + R_ego * lever, in metres, /scale.
    world_from_cam = np.zeros((n, 4, 4))
    for i in range(n):
        c, s = np.cos(th[i]), np.sin(th[i])
        R_ego = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        cam_center_m = np.array([ego_xy[i, 0], ego_xy[i, 1], 0.0]) + R_ego @ sensor2ego[:3, 3]
        T = np.eye(4)
        T[:3, :3] = R_ego            # (orientation convention is up to the resolver)
        T[:3, 3] = cam_center_m / true_scale
        world_from_cam[i] = T
    valid = np.ones(n, dtype=bool)
    return world_from_cam, gps_xy, sensor2ego, valid, true_scale


def test_resolve_scale_gps_recovers_known_scale():
    wfc, gps_xy, s2e, valid, true_scale = _synthetic_scene()
    res = resolve_scale_gps(wfc, gps_xy, s2e, valid)
    assert isinstance(res, ScaleResult)
    assert res.scale == pytest.approx(true_scale, rel=0.05)
    assert res.residual < 1.0  # metres, dominated by the injected GPS noise


def test_resolve_scale_gps_honours_valid_mask():
    wfc, gps_xy, s2e, valid, true_scale = _synthetic_scene()
    gps_xy = gps_xy.copy()
    valid[:3] = False
    gps_xy[:3] = np.nan  # uncovered frames must be ignored, not poison the fit
    res = resolve_scale_gps(wfc, gps_xy, s2e, valid)
    assert res.scale == pytest.approx(true_scale, rel=0.05)


def test_scale_result_is_a_clean_contract():
    # The dataclass the harness reads exists and carries the fields it prints,
    # independent of whether the resolver body is written yet.
    r = ScaleResult(scale=1.0, T=np.eye(4), aligned_xy=np.zeros((2, 2)),
                    residual=0.0, num_iters=1)
    assert r.scale == 1.0 and r.T.shape == (4, 4) and r.aligned_xy.shape == (2, 2)
