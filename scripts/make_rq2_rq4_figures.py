"""Hình RQ2, RQ4 của bài 2 (PG-04, PE-04…PE-07; lịch 22/11 của KBTN v2.6), vẽ từ các bảng đã sinh bằng script đã commit.

Không tính lại thống kê: mọi điểm và CI đọc nguyên từ CSV của make_rq2_tables.py, make_rq4_tables.py, make_qd97_tables.py (gộp
chính = 3 cohort GPL570, pool_set = gpl570; CI 95 % hai phía của IVW và của trọng số bằng nhau). Trục n_T theo thứ tự 5, 10, 20, 40, pool.

    python scripts/make_rq2_rq4_figures.py --tables results/survey_v2

Hình (PDF vector, chữ ≥ 8 pt; thư mục <tables>/RQ2_RQ4_figures/ hoặc --out):
  fig_rq2_comparisons.pdf    Δ AUROC = MVP − đối chứng (target-only H2; select-on-source, pooled H2c; IHC-2, IHC-4 H2b; ptLasso,
                             Trans-GLM mô tả), có đường biên không thua kém của từng họ.
  fig_rq2_cohorts.pdf        Δ AUROC MVP − target-only theo từng cohort (Δ ± 1,96 SE của rq2_cells.csv).
  fig_rq2_decomposition.pdf  Phân rã QĐ97a: Δ_space, Δ_π, Δ_D3 (so với select-on-source và pooled), tổng.
  fig_rq4_h4a.pdf            H4a: Δ Φ Nogueira MVP − target-only và MVP − pooled, đường ngưỡng 0,1.
  fig_rq4_h4b.pdf            H4b: MVP với nhãn nguồn xáo − target-only, biên −0,01.
  fig_rq4_gamma.pdf          Phân bố γ được chọn (m5_pos, m5_cv) theo n_T, cộng qua cohort và k.
Kèm figures_manifest.csv (tệp, SHA-256, bảng nguồn).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2.data import sha256  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402

NT_ORDER = ["5", "10", "20", "40", "pool"]
POOL = "gpl570"
LABEL = {"target_only": "target-only (H2)", "select_on_source": "select-on-source (H2c)", "pooled": "pooled (H2c)",
         "ihc_2": "IHC-2, k = 2 (H2b)", "ihc_4": "IHC-4, k = 4 (H2b)", "ptlasso": "ptLasso (descriptive)",
         "transglm": "Trans-GLM (descriptive)"}
MARGIN = {"target_only": None, "select_on_source": -0.01, "pooled": -0.01, "ihc_2": -0.02, "ihc_4": -0.02, "ptlasso": None, "transglm": None}
DECOMP = {"decomp_space": r"$\Delta_{space}$", "decomp_pi": r"$\Delta_{\pi}$", "decomp_d3_select_on_source": r"$\Delta_{D3}$ vs select-on-source",
          "decomp_d3_pooled": r"$\Delta_{D3}$ vs pooled", "decomp_total": "total (MVP − target-only)"}
SOURCES = {"rq2_pooled": ("RQ2_tables", "rq2_pooled.csv"), "rq2_h2b": ("RQ2_tables", "rq2_h2b.csv"), "rq2_cells": ("RQ2_tables", "rq2_cells.csv"),
           "decomp": ("RQ2_qd97_tables", "qd97a_decomposition_pooled.csv"), "h4a": ("RQ4_tables", "rq4_h4a_pooled.csv"),
           "h4b": ("RQ4_tables", "rq4_h4b_pooled.csv"), "gamma": ("RQ4_tables", "rq2_gamma_distribution.csv")}
FONT = 8


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=None, help="thư mục chứa RQ2_tables/, RQ4_tables/, RQ2_qd97_tables/ "
                    "(mặc định $KLTN_DATA_ROOT/results/survey_v2)")
    ap.add_argument("--out", default="RQ2_RQ4_figures")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------- dữ liệu
def read_tables(root: Path) -> dict:
    out = {}
    for key, (folder, name) in SOURCES.items():
        p = Path(root) / folder / name
        if not p.exists():
            raise SystemExit(f"thiếu bảng {p}")
        out[key] = pd.read_csv(p, dtype={"n_t": str}, encoding="utf-8-sig")
    return out


def series(df: pd.DataFrame, comparison: str, pool: str = POOL) -> pd.DataFrame:
    """Các dòng gộp của một so sánh, theo thứ tự NT_ORDER (bỏ mức không có)."""
    d = df[(df.comparison == comparison) & (df.pool_set == pool)].copy()
    d["pos"] = d.n_t.map({n: i for i, n in enumerate(NT_ORDER)})
    if d.pos.isna().any():
        raise ValueError(f"{comparison}: n_t ngoài {NT_ORDER}: {sorted(d.n_t[d.pos.isna()])}")
    if d.n_t.duplicated().any():
        raise ValueError(f"{comparison}: trùng n_t trong pool_set {pool}")
    return d.sort_values("pos")


def gamma_shares(g: pd.DataFrame) -> pd.DataFrame:
    """Tỉ lệ lần chạy chọn từng γ theo (rule, n_t), cộng qua cohort và k."""
    t = g.groupby(["rule", "n_t", "gamma"], as_index=False).n_runs.sum()
    t["share"] = t.n_runs / t.groupby(["rule", "n_t"]).n_runs.transform("sum")
    return t


# ------------------------------------------------------------------------------------------------- vẽ
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT,
                         "ytick.labelsize": FONT, "legend.fontsize": FONT, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "figure.dpi": 100, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _ivw_eq(ax, d: pd.DataFrame, color="C0", label=True):
    x = d.pos.to_numpy(float)
    for off, col, mk, name in ((-0.08, "ivw", "o", "IVW"), (0.08, "eq", "s", "equal weights")):
        y = d[f"delta_{col}"].to_numpy(float)
        lo, hi = d[f"ci95_{col}_low"].to_numpy(float), d[f"ci95_{col}_high"].to_numpy(float)
        ax.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt=mk, ms=3.5, lw=0.9, capsize=2, color=color,
                    mfc=color if col == "ivw" else "white", label=name if label else None)


def _axis_nt(ax, present):
    ax.set_xticks(range(len(NT_ORDER)), [f"$n_T$={n}" if n != "pool" else "pool" for n in NT_ORDER])
    ax.set_xlim(-0.5, len(NT_ORDER) - 0.5)
    ax.axhline(0, color="0.5", lw=0.6)


def fig_comparisons(t: dict, path: Path):
    plt = _plt()
    both = pd.concat([t["rq2_pooled"], t["rq2_h2b"]], ignore_index=True).drop_duplicates(["comparison", "n_t", "pool_set"])
    comps = list(LABEL)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharey=False)
    for ax, comp in zip(axes.flat, comps):
        d = series(both, comp)
        _ivw_eq(ax, d, label=comp == comps[0])
        _axis_nt(ax, d.n_t)
        if MARGIN[comp] is not None:
            ax.axhline(MARGIN[comp], color="C3", lw=0.7, ls="--")
        ax.set_title(LABEL[comp])
        ax.tick_params(axis="x", rotation=45)
    axes.flat[-1].axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, [f"{x} (95 % CI)" for x in lab] + [], loc="center", frameon=False, title="Δ = MVP − comparator")
    axes.flat[-1].text(0.5, 0.12, "dashed: non-inferiority margin", ha="center", transform=axes.flat[-1].transAxes, color="C3")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ AUROC")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_cohorts(t: dict, path: Path):
    plt = _plt()
    c = t["rq2_cells"]
    c = c[c.comparison == "target_only"].copy()
    c["pos"] = c.n_t.map({n: i for i, n in enumerate(NT_ORDER)})
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    cohorts = sorted(c.cohort.unique())
    for i, (coh, d) in enumerate(c.groupby("cohort")):
        d = d.sort_values("pos")
        off = (i - (len(cohorts) - 1) / 2) * 0.12
        ax.errorbar(d.pos + off, d.delta, yerr=1.96 * d.se, fmt="o-", ms=3, lw=0.8, capsize=2, label=coh.upper(), color=f"C{i}")
    _axis_nt(ax, c.n_t)
    ax.set_ylabel("Δ AUROC (MVP − target-only)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_decomposition(t: dict, path: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    n = len(DECOMP)
    for i, (comp, name) in enumerate(DECOMP.items()):
        d = series(t["decomp"], comp)
        off = (i - (n - 1) / 2) * 0.13
        y = d.delta_ivw.to_numpy(float)
        ax.errorbar(d.pos + off, y, yerr=[y - d.ci95_ivw_low, d.ci95_ivw_high - y], fmt="o", ms=3, lw=0.8, capsize=2, label=name, color=f"C{i}")
    _axis_nt(ax, NT_ORDER)
    ax.set_ylabel("Δ AUROC (IVW, 95 % CI)")
    ax.legend(frameon=False, fontsize=FONT, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_h4a(t: dict, path: Path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6), sharey=True)
    for ax, comp, title in zip(axes, ("mvp_vs_target_only", "mvp_vs_pooled"), ("MVP − target-only", "MVP − pooled")):
        d = series(t["h4a"], comp)
        _ivw_eq(ax, d, label=ax is axes[0])
        _axis_nt(ax, d.n_t)
        ax.axhline(0.1, color="C3", lw=0.7, ls="--")
        ax.set_title(title)
    axes[0].set_ylabel("Δ Nogueira stability Φ")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_h4b(t: dict, path: Path):
    plt = _plt()
    h = t["h4b"]
    comps = list(dict.fromkeys(h.comparison))
    fig, axes = plt.subplots(1, len(comps), figsize=(2.9 * len(comps), 2.6), sharey=True, squeeze=False)
    for ax, comp in zip(axes[0], comps):
        d = series(h, comp)
        _ivw_eq(ax, d, label=ax is axes[0][0])
        _axis_nt(ax, d.n_t)
        ax.axhline(-0.01, color="C3", lw=0.7, ls="--")
        ax.set_title(comp.replace("h4b_", "").replace("_", " "))
    axes[0][0].set_ylabel("Δ AUROC (shuffled-source MVP − target-only)")
    axes[0][0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_gamma(t: dict, path: Path):
    plt = _plt()
    s = gamma_shares(t["gamma"])
    gammas = sorted(s.gamma.unique())
    rules = [r for r in ("m5_pos", "m5_cv") if r in set(s.rule)]
    fig, axes = plt.subplots(1, len(rules), figsize=(2.9 * len(rules), 2.6), sharey=True, squeeze=False)
    for ax, rule in zip(axes[0], rules):
        d = s[s.rule == rule]
        nts = [n for n in NT_ORDER if n in set(d.n_t)]
        bottom = np.zeros(len(nts))
        for j, g in enumerate(gammas):
            v = np.array([d[(d.n_t == n) & (d.gamma == g)].share.sum() for n in nts])
            ax.bar(range(len(nts)), v, bottom=bottom, color=f"C{j}", label=f"γ = {g:g}", width=0.7)
            bottom += v
        ax.set_xticks(range(len(nts)), [f"$n_T$={n}" if n != "pool" else "pool" for n in nts], rotation=45)
        ax.set_title(rule)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("share of runs")
    axes[0][-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


FIGURES = {"fig_rq2_comparisons.pdf": (fig_comparisons, ["rq2_pooled", "rq2_h2b"]), "fig_rq2_cohorts.pdf": (fig_cohorts, ["rq2_cells"]),
           "fig_rq2_decomposition.pdf": (fig_decomposition, ["decomp"]), "fig_rq4_h4a.pdf": (fig_h4a, ["h4a"]),
           "fig_rq4_h4b.pdf": (fig_h4b, ["h4b"]), "fig_rq4_gamma.pdf": (fig_gamma, ["gamma"])}


def build(tables: dict, out: Path) -> pd.DataFrame:
    rows = []
    for name, (fn, src) in FIGURES.items():
        p = Path(out) / name
        fn(tables, p)
        rows.append({"figure": name, "sha256": sha256(p), "sources": " ".join("/".join(SOURCES[s]) for s in src)})
    return pd.DataFrame(rows)


def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    root = args.tables or DATA_ROOT.joinpath("results", "survey_v2")
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t0 = time.time()
    tables = read_tables(root)
    out = results_dir(args.out)
    manifest = build(tables, out)
    manifest.to_csv(out / "figures_manifest.csv", index=False, encoding="utf-8-sig")
    cfg = {"experiment": "RQ2_RQ4_figures", "tables_root": str(root), "pool_set": POOL, "n_t_order": NT_ORDER,
           "source_sha256": {"/".join(v): sha256(Path(root, *v)) for v in SOURCES.values()},
           "note": "chỉ vẽ lại số của các bảng; không tính lại thống kê", "elapsed_s": time.time() - t0, "versions": all_versions()}
    write_config(out, prov, cfg)
    print(manifest.to_string(index=False))
    print("ghi", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
