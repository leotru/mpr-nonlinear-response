#!/usr/bin/env python3
"""Generate all manuscript figures from the data in data/.

Usage: python scripts/make_figures.py [--data data] [--out figures] [--config configs/paper.toml]

Output file -> figure in the manuscript:
    fig1_linear.pdf          Fig. 1  linear susceptibility
    fig_bispectrum.pdf       Fig. 2  map of the second-order susceptibility
    fig2_harmonic_IM.pdf     Fig. 3  second harmonic and intermodulation
    fig3_rectification.pdf   Fig. 4  mean-rate shift (rectification)
    fig4_KK_sumrules.pdf     Fig. 5  Kramers-Kronig relations and sum rules
    fig5_pulling.pdf         Fig. 6  neuron-resolved mean-rate shift
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, SymLogNorm  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy.integrate import cumulative_trapezoid  # noqa: E402

from mprresponse.config import load_config, theory_from_config  # noqa: E402
from mprresponse.kk import kk_real_from_imag  # noqa: E402

C_RE, C_IM, C_GREEN = "#1f5fa8", "#c2452d", "#2a8a57"
COL, DCOL = 3.4, 7.0                       # APS column and page widths (inches)
XLAB = r"$\omega$ (units of $1/\tau$)"
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def _arr(rows, key):
    return np.array([d[key] for d in rows])


def fig_linear(m, S, out):
    W = np.linspace(0.02, 13, 2000)
    ws = _arr(S, "w")
    fig, a = plt.subplots(figsize=(COL, 2.5))
    c1 = m.chi1_r(W)
    a.plot(W, c1.real, c=C_RE, lw=1.3, label=r"Re $\chi^{(1)}_r$")
    a.plot(W, c1.imag, c=C_IM, lw=1.3, label=r"Im $\chi^{(1)}_r$")
    x, e = _arr(S, "chi1"), _arr(S, "chi1_err")
    a.errorbar(ws, x[:, 0], e, fmt="o", c=C_RE, ms=3.5, mfc="w", mew=0.9)
    a.errorbar(ws, x[:, 1], e, fmt="s", c=C_IM, ms=3.5, mfc="w", mew=0.9)
    a.axvline(m.w0, c="0.6", ls=":", lw=0.8)
    a.text(m.w0 + 0.2, 0.9, r"$\omega_0$", c="0.4")
    a.set(xlabel=XLAB, ylabel=r"$\chi^{(1)}_r(\omega)$")
    a.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(out / "fig1_linear.pdf")


def fig_bispectrum(m, R, out):
    w1 = np.linspace(0.02, 12, 700)
    w2 = np.linspace(-12, 12, 1400)
    W1, W2 = np.meshgrid(w1, w2)
    fig, a = plt.subplots(figsize=(COL, 3.3))
    pc = a.pcolormesh(W1, W2, np.abs(m.chi2_r(W1, W2)), norm=LogNorm(1e-4, 3), cmap="magma",
                      shading="auto", rasterized=True)
    fig.colorbar(pc, ax=a, pad=0.02).set_label(r"$|\chi^{(2)}_r(\omega_1,\omega_2)|$")
    ls = dict(c="w", lw=0.6, ls=":", alpha=0.7)
    a.axvline(m.w0, **ls)
    for y in (m.w0, -m.w0):
        a.axhline(y, **ls)
        a.plot(w1, y - w1, **ls)
    a.plot(w1, -w1, c="#7fd1a8", lw=0.9, ls="--")
    a.plot(w1, w1, c="#9cc3ff", lw=0.9, ls="--")
    ws = _arr(R["single"], "w")
    a.plot(ws, ws, "o", ms=2.6, mfc="none", mec="#9cc3ff", mew=0.8)
    a.plot(ws, -ws, "o", ms=2.6, mfc="none", mec="#7fd1a8", mew=0.8)
    for d in R["two"]:
        p, q = d["w12"]
        a.plot([q, q], [p, -p], "s", ms=3, mfc="none", mec="w", mew=0.8)
    wc = m.omega_c
    a.plot([wc], [-wc], "*", ms=8, mfc="#7fd1a8", mec="k", mew=0.5)
    a.annotate(r"$\omega_c$", (wc, -wc), xytext=(wc - 2.4, -wc - 2.4), color="w", fontsize=8,
               arrowprops=dict(arrowstyle="-", color="w", lw=0.6))
    a.text(7.2, 10.6, "harmonic", c="#9cc3ff", fontsize=7, ha="center")
    a.text(2.4, -7.6, "mean-rate shift", c="#7fd1a8", fontsize=7, ha="center")
    a.text(10.9, -9.3, r"$\chi^{(2)}_r=0$", color="w", fontsize=7.5, ha="center")
    a.set(xlabel=r"$\omega_1$ (units of $1/\tau$)", ylabel=r"$\omega_2$ (units of $1/\tau$)",
          xlim=(0, 12), ylim=(-12, 12))
    for s in ("top", "right"):
        a.spines[s].set_visible(True)
    fig.tight_layout()
    fig.savefig(out / "fig_bispectrum.pdf", dpi=300)


def fig_harmonic(m, R, out):
    S, W = R["single"], np.linspace(0.02, 13, 2000)
    ws = _arr(S, "w")
    fig, ax = plt.subplots(1, 2, figsize=(DCOL, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    a, h = ax[0], m.chi2_r(W, W)
    a.plot(W, h.real, c=C_RE, lw=1.3, label=r"Re $\chi^{(2)}_r(\omega,\omega)$")
    a.plot(W, h.imag, c=C_IM, lw=1.3, label=r"Im $\chi^{(2)}_r(\omega,\omega)$")
    x, e, o = _arr(S, "chi2h"), _arr(S, "chi2h_err"), _arr(S, "ode_chi2h")
    a.errorbar(ws, x[:, 0], e, fmt="o", c=C_RE, ms=3.5, mfc="w", mew=0.9, label="QIF network")
    a.errorbar(ws, x[:, 1], e, fmt="s", c=C_IM, ms=3.5, mfc="w", mew=0.9)
    a.plot(ws, o[:, 0], "+", c="k", ms=5, label=r"MPR, same $\varepsilon$")
    a.plot(ws, o[:, 1], "+", c="k", ms=5)
    for v, lab in ((m.w0 / 2, r"$\omega_0/2$"), (m.w0, r"$\omega_0$")):
        a.axvline(v, c="0.6", ls=":", lw=0.8)
        a.text(v + 0.15, 0.6, lab, c="0.4")
    a.set(xlabel=XLAB, ylabel=r"$\chi^{(2)}_r(\omega,\omega)$", ylim=(-0.75, 0.75))
    a.legend(loc="lower right", fontsize=7.5)
    a.text(-0.16, 1.02, "(a)", transform=a.transAxes, weight="bold")
    a, pts = ax[1], []
    for d in R["two"]:
        for k in ("sum", "dif"):
            th, ne, er = d["th_" + k], d[k], d[k + "_err"]
            a.errorbar(th[0], ne[0], er, fmt="o", c=C_RE, ms=4, mfc="w", mew=0.9)
            a.errorbar(th[1], ne[1], er, fmt="s", c=C_IM, ms=4, mfc="w", mew=0.9)
            pts += th
    lo, hi = min(pts) - 0.04, max(pts) + 0.04
    a.plot([lo, hi], [lo, hi], c="0.6", lw=0.8)
    a.plot([], [], "o", c=C_RE, mfc="w", label="Re")
    a.plot([], [], "s", c=C_IM, mfc="w", label="Im")
    a.set(xlabel=r"$\chi^{(2)}_r(\omega_1,\pm\omega_2)$ theory", ylabel="QIF network")
    a.legend(loc="upper left")
    a.text(-0.25, 1.02, "(b)", transform=a.transAxes, weight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig2_harmonic_IM.pdf")


def fig_rectification(m, S, out):
    W = np.linspace(0.02, 13, 2000)
    ws = _arr(S, "w")
    fig, a = plt.subplots(figsize=(COL, 2.7))
    a.plot(W, m.rectification(W), c="k", lw=1.3, label="theory")
    a.errorbar(ws, _arr(S, "chi2d"), _arr(S, "chi2d_err"), fmt="o", c=C_GREEN, ms=3.8,
               mfc="w", mew=0.9, label="QIF network")
    a.plot(ws, _arr(S, "ode_chi2d"), "+", c="k", ms=5, label=r"MPR, same $\varepsilon$")
    a.axhline(0, c="0.6", lw=0.6)
    a.axvline(m.omega_c, c=C_GREEN, ls="--", lw=0.9)
    a.text(m.omega_c + 0.25, -0.25, r"$\omega_c$", c=C_GREEN)
    a.set_yscale("symlog", linthresh=2e-4, linscale=1.0)
    a.set(xlabel=XLAB, ylabel=r"$\chi^{(2)}_r(\omega,-\omega)$")
    a.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig3_rectification.pdf")


def fig_kk(m, out):
    W = np.linspace(0.02, 13, 2000)
    fig, ax = plt.subplots(1, 2, figsize=(DCOL, 2.5))
    a = ax[0]
    a.plot(W, m.chi2_r(W, W).real, c=C_RE, lw=1.3, label=r"Re $\chi^{(2)}_r(\omega,\omega)$")
    Wk = np.linspace(0.2, 13, 60)
    a.plot(Wk, [kk_real_from_imag(lambda z: m.chi2_r(z, z), w) for w in Wk], ".", c="k", ms=3,
           label="KK from Im")
    a.plot(W, m.rectification(W), c=C_GREEN, lw=1.1, ls="--",
           label=r"$\chi^{(2)}_r(\omega,-\omega)$ (Im $\equiv0$)")
    a.set(xlabel=XLAB, ylim=(-0.75, 0.75))
    a.legend(loc="lower right", fontsize=7.5)
    a.text(-0.16, 1.02, "(a)", transform=a.transAxes, weight="bold")
    a = ax[1]
    L = np.linspace(1e-4, 200, 400001)
    h2 = m.chi2_r(L, L).imag
    a.plot(L, cumulative_trapezoid(L * m.chi1_r(L).imag, L, initial=0) / (np.pi * m.r), c=C_RE,
           lw=1.2, label=r"$\int_0^\Lambda\omega\,{\rm Im}\chi^{(1)}_r\,d\omega\,/\,\pi r^*$")
    a.plot(L, cumulative_trapezoid(L**3 * h2, L, initial=0) * (2 / np.pi) / (-2.5 * m.r),
           c=C_IM, lw=1.2,
           label=r"$\frac{2}{\pi}\int_0^\Lambda\omega^3{\rm Im}\chi^{(2)}_r\,d\omega\,/\,"
                 r"(-\frac{5}{2}r^*)$")
    a.plot(L, cumulative_trapezoid(L * h2, L, initial=0) / (np.pi * m.r), c=C_GREEN, lw=1.2,
           label=r"$\int_0^\Lambda\omega\,{\rm Im}\chi^{(2)}_r\,d\omega\,/\,\pi r^*$")
    for y in (0, 1):
        a.axhline(y, c="0.6", ls=":", lw=0.8)
    a.set_xscale("log")
    a.set(xlabel=r"cutoff $\Lambda$ (units of $1/\tau$)", xlim=(0.3, 200), ylim=(-2.2, 1.6))
    a.legend(loc="center left", fontsize=7)
    a.text(-0.14, 1.02, "(b)", transform=a.transAxes, weight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig4_KK_sumrules.pdf")


def fig_pulling(m, M, out, min_count=15):
    base, stim, eps = M["base_rate"], M["stim_rate"], M["eps"]
    ws = M["frequencies"]
    Om = 2 * np.pi * base                       # intrinsic angular firing frequency
    silent = base < 1e-3
    ye = np.arange(0, 14.01, 0.25)
    yc = 0.5 * (ye[1:] + ye[:-1])
    idx = np.digitize(Om, ye) - 1
    idx[silent] = 0
    shift = (stim - base) / (eps[:, None] ** 2 / 2)
    grid = np.full((ye.size - 1, ws.size), np.nan)
    for b in range(ye.size - 1):
        sel = idx == b
        if sel.sum() >= min_count:
            grid[b] = shift[:, sel].mean(axis=1)
    fig = plt.figure(figsize=(7.0, 3.9))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[4.2, 0.9, 0.15], height_ratios=[1, 3.2],
                  hspace=0.06, wspace=0.06)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    cax = fig.add_subplot(gs[1, 2])
    dw = ws[1] - ws[0]
    xe = np.concatenate([ws - dw / 2, [ws[-1] + dw / 2]])
    pc = ax.pcolormesh(xe, ye, grid, cmap="RdBu_r", shading="flat", rasterized=True,
                       norm=SymLogNorm(linthresh=1e-3, linscale=0.6, vmin=-0.3, vmax=0.3))
    ax.set_facecolor("0.85")
    ax.plot([0, 14.5], [0, 14.5], c="k", lw=0.8, ls="--")
    ax.text(11.6, 12.4, r"$\Omega_j=\omega$", rotation=28, fontsize=8)
    ax.axvline(m.w0, c="0.3", lw=0.7, ls=":")
    ax.text(m.w0 + 0.1, 13.3, r"$\omega_0$", fontsize=8, c="0.3")
    ax.axvline(m.omega_c, c=C_GREEN, lw=0.9, ls="--")
    ax.text(m.omega_c + 0.1, 13.3, r"$\omega_c$", fontsize=8, c=C_GREEN)
    ax.text(0.4, 0.45, "silent neurons", fontsize=7, c="0.15")
    ax.set(xlim=(xe[0], xe[-1]), ylim=(0, 14), xlabel=r"stimulus frequency " + XLAB,
           ylabel=r"intrinsic firing frequency $\Omega_j$ (units of $1/\tau$)")
    fig.colorbar(pc, cax=cax).set_label(r"$\Delta\nu_j/(\varepsilon^2/2)$")
    W = np.linspace(xe[0], xe[-1], 600)
    axt.plot(W, m.rectification(W), c="k", lw=1.1, label="theory (population)")
    axt.plot(ws, shift.mean(axis=1), "o", ms=3, mfc="w", mec=C_GREEN, mew=0.9,
             label="sum over neurons")
    axt.axhline(0, c="0.6", lw=0.6)
    axt.axvline(m.omega_c, c=C_GREEN, lw=0.9, ls="--")
    axt.set_yscale("symlog", linthresh=1e-3, linscale=0.6)
    axt.set_yticks([-1e-1, -1e-2, 0, 1e-3])
    axt.tick_params(labelbottom=False)
    axt.legend(fontsize=6.5, loc="lower right")
    axt.set_ylabel(r"$\chi^{(2)}_r(\omega,-\omega)$", fontsize=7.5)
    hist, _ = np.histogram(Om[~silent], bins=ye)
    axr.barh(yc, hist / Om.size, height=0.25, color="0.4")
    axr.barh([0.125], [silent.mean()], height=0.25, color="0.7")
    axr.set_xscale("log")
    axr.set_xlim(1e-4, 1)
    axr.tick_params(labelleft=False)
    axr.set_xlabel("fraction of\nneurons", fontsize=7.5)
    fig.savefig(out / "fig5_pulling.pdf", dpi=300, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--config", default="configs/paper.toml")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = theory_from_config(load_config(args.config))
    with open(data / "population_sweep.json") as f:
        R = json.load(f)
    M = dict(np.load(data / "neuron_map.npz"))
    fig_linear(m, R["single"], out)
    fig_bispectrum(m, R, out)
    fig_harmonic(m, R, out)
    fig_rectification(m, R["single"], out)
    fig_kk(m, out)
    fig_pulling(m, M, out)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
