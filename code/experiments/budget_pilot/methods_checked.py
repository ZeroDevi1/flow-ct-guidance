"""Isolated pilot methods. No changes to the historical sampler.

WLS uses observed log-counts, not ground-truth variance. Delta-method
variance is unreliable near zero counts; this pilot is not a Poisson likelihood.
"""
import math
import torch


def measured_wls_weights(y, i0, mu_scale=0.06, floor=1.0):
    if not all(math.isfinite(v) and v > 0 for v in (i0, mu_scale, floor)):
        raise ValueError('Positive noise parameters required')
    if not torch.isfinite(y).all():
        raise ValueError('Nonfinite sinogram')
    counts = (i0 * torch.exp(-mu_scale * y)).clamp(min=floor)
    precision = counts * mu_scale ** 2
    if not torch.isfinite(precision).all():
        raise ValueError('Nonfinite precision')
    # Remove only overall scale, retain heteroscedasticity. Lambda still needs validation.
    return precision / precision.flatten(1).mean(1).view(-1, 1, 1, 1)


def pixel_weighted_dc(op, x0, y, weights, lam_abs, n_iter=5):
    if not math.isfinite(lam_abs) or lam_abs <= 0 or n_iter <= 0:
        raise ValueError('Positive lambda/iterations required')
    if weights.shape != y.shape or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Finite nonnegative pixel weights must match sinogram')
    with torch.no_grad():
        rhs = op.A_lin_T(weights * (y - op.offset)) + lam_abs * x0
        def matvec(v):
            return op.A_lin_T(weights * op.A_lin(v)) + lam_abs * v
        x = x0.clone()
        r = rhs - matvec(x)
        p = r.clone()
        rs = (r*r).flatten(1).sum(1)
        for _ in range(n_iter):
            ap = matvec(p)
            alpha = rs / ((p*ap).flatten(1).sum(1) + 1e-12)
            x += alpha[:, None, None, None] * p
            r -= alpha[:, None, None, None] * ap
            new = (r*r).flatten(1).sum(1)
            if new.max() < 1e-10:
                break
            p = r + (new / (rs+1e-12))[:, None, None, None] * p
            rs = new
    return x


@torch.no_grad()
def sample_dc(model, op, y, shape, *, nfe, lam_relative, n_iter, seed, weights=None):
    if nfe <= 0 or n_iter <= 0 or not math.isfinite(lam_relative) or lam_relative <= 0:
        raise ValueError('Positive finite sampler settings required')
    g = torch.Generator(device=y.device).manual_seed(seed)
    x = torch.randn(shape, generator=g, device=y.device)
    ts = torch.linspace(0, 1, nfe+1, device=y.device)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        v = model(x, t0.expand(shape[0]))
        clean = x + (1-t0)*v
        eps = x - t0*v
        if weights is None:
            corrected = op.cg_data_consistency(clean, y, lam=lam_relative, n_iter=n_iter, relative=True)
        else:
            corrected = pixel_weighted_dc(op, clean, y, weights, lam_relative*op.ata_scale, n_iter)
        x = (1-t1)*eps + t1*corrected
    if not torch.isfinite(x).all():
        raise FloatingPointError('Non-finite pre-clamp sampler output')
    return x.clamp(-1, 1)
