"""Tests for the rendered-view scorer (PSNR/SSIM/L1) and the held-out split."""
import numpy as np
import pytest

from nuslam.eval import RenderErrors, evaluate_render, holdout_indices


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(32, 48, 3), dtype=np.uint8)


def test_identical_images_are_perfect():
    im = _img()
    e = evaluate_render(im, im)
    assert e.psnr == float("inf")
    assert e.ssim == pytest.approx(1.0, abs=1e-6)
    assert e.l1 == 0.0


def test_uint8_and_float01_agree():
    im = _img(1)
    noisy = np.clip(im.astype(np.float32) + 20, 0, 255).astype(np.uint8)
    e_u8 = evaluate_render(noisy, im)
    e_f = evaluate_render(noisy.astype(np.float32) / 255.0, im.astype(np.float32) / 255.0)
    assert e_u8.psnr == pytest.approx(e_f.psnr, rel=1e-4)
    assert e_u8.ssim == pytest.approx(e_f.ssim, rel=1e-4)


def test_more_noise_lowers_psnr():
    im = _img(2).astype(np.float32) / 255.0
    rng = np.random.default_rng(3)
    a = np.clip(im + rng.normal(0, 0.02, im.shape), 0, 1)
    b = np.clip(im + rng.normal(0, 0.10, im.shape), 0, 1)
    assert evaluate_render(a, im).psnr > evaluate_render(b, im).psnr


def test_mask_restricts_psnr_and_count():
    im = _img(4).astype(np.float32) / 255.0
    bad = im.copy()
    bad[:16] = np.clip(bad[:16] + 0.3, 0, 1)   # corrupt only the top half
    mask = np.zeros(im.shape[:2], bool)
    mask[16:] = True                            # score only the clean bottom half
    e_all = evaluate_render(bad, im)
    e_clean = evaluate_render(bad, im, mask=mask)
    assert e_clean.psnr > e_all.psnr
    assert e_clean.num_pixels == int(mask.sum()) * 3


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_render(_img(5), _img(6)[:, :40])


def test_holdout_split_is_a_partition():
    train, test = holdout_indices(40, every=8)
    assert set(train.tolist()) | set(test.tolist()) == set(range(40))
    assert not (set(train.tolist()) & set(test.tolist()))
    assert list(test) == [0, 8, 16, 24, 32]


def test_holdout_offset_keeps_frame0_in_train():
    train, test = holdout_indices(40, every=8, offset=4)
    assert 0 in train and 4 in test
