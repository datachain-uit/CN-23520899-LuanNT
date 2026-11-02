"""Sinh mỗi thí nghiệm (E0–E7, sanity, ablation) một notebook Kaggle riêng trong `notebooks/kaggle/`.

Mọi notebook dùng chung các ô Gate 1–3, pytest, chạy, lưu output của `notebooks/kaggle_survey_v3_runner.ipynb` (một nguồn duy
nhất); chỉ khác phần giới thiệu, danh sách việc (JOB) được phép và thứ tự phiên. Notebook chỉ chọn việc và gọi
`scripts/kaggle_jobs.py` (M18) — lệnh, shard, đầu vào cần khôi phục và kiểm tra đầu ra nằm trong script đó.

    python scripts/make_kaggle_notebooks.py           # ghi lại notebooks/kaggle/*.ipynb
    python scripts/make_kaggle_notebooks.py --check   # mã 1 nếu notebook nào lệch với runner/danh mục (dùng trong test)

Sửa logic chung -> sửa runner rồi chạy lại script này; không sửa tay notebooks/kaggle/*.ipynb.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "notebooks" / "kaggle_survey_v3_runner.ipynb"
OUT_DIR = REPO / "notebooks" / "kaggle"


def _load_jobs():
    spec = importlib.util.spec_from_file_location("kaggle_jobs_for_notebooks", REPO / "scripts" / "kaggle_jobs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.JOBS


@dataclass
class Experiment:
    file: str
    title: str
    intro: str          # mục đích, đầu ra, nghiệm thu (markdown)
    jobs: list          # theo thứ tự chạy
    order: str          # hướng dẫn thứ tự phiên (markdown)


EXPERIMENTS = [
    Experiment(
        "kaggle_00_sanity.ipynb", "Sanity Y10 — TCGA normal vs tumor",
        "Kiểm tra đường ống đọc dữ liệu + phân loại trên bài toán dễ (normal vs tumor). Đầu ra `results/survey_v2/sanity/`.\n"
        "Nghiệm thu: AUROC ≥ 0,99. Không đạt → dừng, báo GVHD trước khi chạy E0.",
        ["sanity"], "Một phiên. Có thể chạy trên máy cá nhân thay vì Kaggle."),
    Experiment(
        "kaggle_E0_signal_check.ipynb", "E0 — kiểm tra tín hiệu (5 bộ)",
        "200 hoán vị nhãn + 100 random-k cho TCGA và 4 cohort GEO. Đầu ra `E0/<bộ>/`.\n"
        "Nghiệm thu: `signal_present = True` ở cả 5 bộ. Không đạt → báo GVHD, **không sửa ngưỡng**.",
        ["e0"], "Một phiên (~1 giờ), không cần khôi phục. Chạy song song được với các phiên E1 pha A."),
    Experiment(
        "kaggle_E1_feature_selection_classification.ipynb", "E1 — chọn gen (pha A) + phân loại (pha B), 5×5 CV",
        "Pha A: 11 phương pháp + IHC/panel trên bản chia 5×5 → `E1_feature_selection/<bộ>/selected_genes.csv`.\n"
        "Pha B: LR-L2 (+RF, SVM ở k ∈ {5, 25}), B0, random-k → `E1_classification/<bộ>/`.\n"
        "Wrapper GA + BPSO (3 cohort GPL570, k ∈ {5, 10, 50}) → `E1_wrapper_feature_selection/<cohort>/`, rồi nối vào pha B.\n"
        "Nghiệm thu mỗi phiên: notebook in `JOB_OK <job>` (config.json traceable, commit = HEAD).",
        ["e1_phaseA_geo", "e1_phaseB_geo", "e1_phaseA_tcga_r012", "e1_phaseA_tcga_r34", "e1_phaseB_tcga",
         "wrap_gse10245", "wrap_gse30219", "wrap_gse37745", "e1_phaseB_wrappers"],
        "1. **Đợt 1 (03/11), GEO:** `e1_phaseA_geo` → `e1_phaseB_geo` (khôi phục output `e1_phaseA_geo`).\n"
        "2. **Đợt 1T (04/11), TCGA:** `e1_phaseA_tcga_r012` và `e1_phaseA_tcga_r34` chạy **song song** (2 phiên) → "
        "`e1_phaseB_tcga` (khôi phục output cả hai).\n"
        "3. **Đợt 3 (08/11), wrapper:** `wrap_gse10245`, `wrap_gse30219`, `wrap_gse37745` (3 phiên song song) → "
        "`e1_phaseB_wrappers` (khôi phục `e1_phaseB_geo` + 3 phiên wrap).\n\n"
        "Mỗi lần đổi JOB: Save Version mới (hoặc nhân bản notebook để chạy song song)."),
    Experiment(
        "kaggle_E1b_fewshot.ipynb", "E1b — đường cong học few-shot (4 cohort GEO)",
        "5 split × 4 tập con lồng nhau cho mỗi n_T, trên 4 cohort GEO → `E1b/<cohort>/`.\n"
        "Cần `fewshot_splits.csv` bản 29/10 (có cột `subset`) — Gate 1 kiểm tra.",
        ["fewshot_e1b"], "Một phiên (3–5 giờ), không cần khôi phục. Đợt 2 (06/11)."),
    Experiment(
        "kaggle_E2_E3_timing.ipynb", "E2 + E3 — chi phí tính toán (đo thời gian)",
        "E2: phiên tham chiếu đo tại một điểm, ghi `E2_E3_ref/calibration.csv`. E3: lưới n × p chia 3 phần theo chi phí, mỗi phần tự "
        "hiệu chuẩn với phiên tham chiếu (QĐ18: lệch ≥ 10 % → phiên dừng, chạy lại). `e2e3_merge` gộp → `E2_E3/`.\n\n"
        "**Đo thời gian:** mỗi phiên chạy **một mình** (không chạy notebook khác cùng lúc trên cùng phiên), Accelerator None, "
        "cùng loại máy CPU 4 nhân với phiên `e2_ref`. Nghiệm thu: `calibration_ok` không `false`.",
        ["e2_ref", "e3_p1", "e3_p2", "e3_p3", "e2e3_merge"],
        "**v3.1.6 (QĐ27, 31/10):** các phiên `e2_ref`, `e3_p2`, `e3_p3` ngày 29–30/10 đo ở bản có tracemalloc bật và bộ nhớ tiến trình "
        "con kế thừa từ cha → KHÔNG dùng được (chuyển vào 04_Luu-vet); `merge-e2e3` từ chối phiên đo ở bản cũ. Chạy lại cả bốn "
        "phiên ở commit v3.1.6, cùng loại máy (Intel Xeon 2,20 GHz như phiên 30/10).\n\n"
        "1. `e2_ref` (đợt 2, 06/11; ≈ 3 giờ).\n"
        "2. `e3_p1`, `e3_p2`, `e3_p3` — mỗi phiên khôi phục output `e2_ref` MỚI; chạy lần lượt hoặc song song ở các "
        "phiên Kaggle riêng (mỗi phiên vẫn một mình trong máy của nó).\n"
        "3. `e2e3_merge` — khôi phục output `e2_ref` + 3 phiên E3 (vài phút; có thể chạy trên máy)."),
    Experiment(
        "kaggle_E4_E5_stability.ipynb", "E4 + E5 — ổn định chọn gen (TCGA, GSE37745)",
        "E4 (M = 30 tập con) trên TCGA và GSE37745; E5a dư thừa, E5b nhạy seed (rf_importance, stability_selection mọi k; GA, BPSO "
        "k = 10), E5c trùng lặp ở k = 10 — chỉ GSE37745 (E5c có cả TCGA). Đầu ra `E4_E5/<bộ>/`.",
        ["e4e5_gse37745", "e4_tcga"],
        "1. **Đợt 2 (06/11):** `e4e5_gse37745` (khôi phục output `e1_phaseA_geo` của notebook E1) và `e4_tcga` (khôi phục output "
        "`e1_phaseB_tcga`); hai phiên chạy song song được.\n"
        "2. `e4_tcga` có thể quá 12 giờ → phiên dừng mã 124, chạy lại cùng JOB với output vừa lưu.\n"
        "3. **Đợt 3 (08/11):** chạy lại `e4e5_gse37745`, khôi phục output lần 1 + `wrap_gse37745` (notebook E1) — chỉ bổ sung E5c "
        "có GA/BPSO (vài phút)."),
    Experiment(
        "kaggle_E6_tables.ipynb", "E6 + bảng, hình, provenance (make_survey_tables.py)",
        "Gộp mọi kết quả đã có → `tables/*.csv|*.tex`, `figures/*`, `provenance_manifest.json`; E6 ghép tại k = 10.\n"
        "Chỉ dùng output của các phiên có `JOB_OK` cùng một commit.",
        ["tables"],
        "Khôi phục đúng danh sách và **đúng thứ tự** cột *Khôi phục* (zip sau ghi đè zip trước; `redo_svm_rfe_B` rồi `panels` ở cuối). "
        "v3.1.7: chạy lại sau `panels` → thêm bảng ablation IHC (`e1_ablation_noihc*`), hàng panel công bố, CI bootstrap tái lập (QĐ30–QĐ32). "
        "Hoặc tải các zip về máy rồi chạy `python scripts/make_survey_tables.py`."),
    Experiment(
        "kaggle_panels.ipynb", "Panel công bố (v3.1.7, QĐ32: Girard 2016, Wilkerson 2013)",
        "Hai panel cố định trong `survey_v2/panels/published_panels.csv` chạy qua E1 pha A (panel cố định, gen thiếu ghi vào config), "
        "pha B (LR-L2) và E1b, nối vào kết quả đã có; mọi phương pháp khác giữ nguyên (`--redo-methods` chỉ hai panel).\n"
        "Nghiệm thu: `JOB_OK`; config.json của E1_feature_selection/E1_classification/E1b có `methods_in_results` chứa "
        "`panel_girard2016`, `panel_wilkerson2013` và `redo_methods` = hai panel; các dòng khác trùng từng byte với `redo_svm_rfe_B`.",
        ["panels"],
        "Một phiên (< 0,5 giờ). Khôi phục output `redo_svm_rfe_B` (đã có E1 pha A/B, E1b, E4/E5, wrapper). Sau đó chạy lại `tables` "
        "(notebook E6) với `redo_svm_rfe_B` rồi `panels` ở cuối danh sách khôi phục."),
    Experiment(
        "kaggle_E7_transfer.ipynb", "E7 — chuyển giao đơn giản TCGA → GEO",
        "4 chiến lược (source_only, select_on_source, pooled, target_only) × 5 k × 20 lần chạy × n_T ∈ {5, 10, 20} trên 4 cohort "
        "→ `E7/<cohort>/`. Dùng cùng bản chia few-shot với E1b.",
        ["e7"], "Một phiên (2–4 giờ), đợt 3 (08/11), sau khi E1b đã chạy."),
    Experiment(
        "kaggle_redo_svm_rfe.ipynb", "Chọn lại SVM-RFE (v3.1.6, QĐ25: random_state)",
        "SVM-RFE của các phiên 29–30/10 không tái lập (LinearSVC dual không có random_state). Hai việc chọn lại và đánh giá lại "
        "CHỈ svm_rfe bằng `--redo-methods svm_rfe`; mọi phương pháp khác lấy từ checkpoint/dòng đã có trong zip khôi phục.\n"
        "Nghiệm thu: `JOB_OK`; config.json của mỗi thư mục có `redo_methods = ['svm_rfe']` và `checkpoints_reused_from_commit` / "
        "`rows_reused_from_commit` = commit của đợt 1 (9b2d6a5 hoặc 68e7c7d).",
        ["redo_svm_rfe_A", "redo_svm_rfe_B"],
        "1. `redo_svm_rfe_A` — khôi phục output `e1_phaseA_geo`, `e1_phaseB_tcga`, `ablation_phaseA_geo`, `ablation_phaseA_tcga` "
        "(cần thư mục `checkpoints/` của pha A; ~1 giờ, 4 tiến trình).\n"
        "2. `redo_svm_rfe_B` — khôi phục output `redo_svm_rfe_A`, `e1_phaseB_wrappers` (3 cohort GPL570 đã có GA/BPSO), `e1_phaseB_geo` "
        "(gse42127), `e1_phaseB_tcga`, `fewshot_e1b`, `e4e5_gse37745`, `e4_tcga` (2–3 giờ). Sau đó `ablation_phaseB` và `tables`."),
    Experiment(
        "kaggle_ablation_no_IHC.ipynb", "Ablation — bỏ 4 gen IHC (NKX2-1, TP63, NAPSA, KRT5)",
        "Lặp lại pha A + pha B E1 khi loại 4 gen IHC → `E1_feature_selection_noIHC/`, `E1_classification_noIHC/`. "
        "Phục vụ khoá luận (ưu tiên 2), hạn 10/11.",
        ["ablation_phaseA_geo", "ablation_phaseA_tcga", "ablation_phaseB"],
        "1. `ablation_phaseA_geo` và `ablation_phaseA_tcga` (2 phiên song song; TCGA có thể quá 12 giờ → chạy lại cùng JOB).\n"
        "2. `ablation_phaseB` — khôi phục output cả hai phiên pha A."),
]

JOB_LINE_PREFIX = "JOB = "


def _src(cell) -> str:
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def _lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _job_table(exp: Experiment, jobs) -> str:
    rows = ["| JOB | Đợt · hạn | Khôi phục (Input) | Giờ | Nội dung |", "|---|---|---|---|---|"]
    for name in exp.jobs:
        j = jobs[name]
        flag = " **(một mình)**" if j.exclusive else ""
        rows.append(f"| `{name}`{flag} | {j.batch} | {', '.join(f'`{r}`' for r in j.restore_from) or '—'} | {j.hours} | "
                    f"{j.what.replace('|', '/')} |")
    return "\n".join(rows)


def build(exp: Experiment, runner: dict, jobs) -> dict:
    cells = copy.deepcopy(runner["cells"])
    assert cells[0]["cell_type"] == "markdown" and JOB_LINE_PREFIX in _src(cells[1]), "runner đổi cấu trúc — sửa generator"
    head = (f"# Kaggle survey v3 — {exp.title}\n\n"
            f"Notebook sinh bởi `scripts/make_kaggle_notebooks.py` — **không sửa tay**; chỉ sửa ô cấu hình trước mỗi phiên.\n"
            f"Notebook chỉ chọn việc và gọi `scripts/kaggle_jobs.py` (M18). Hướng dẫn chung: `docs/survey_v2/KAGGLE_RUN_PLAN.md`.\n\n"
            f"## Thí nghiệm\n\n{exp.intro}\n\n## Các việc (mỗi phiên Kaggle = một JOB)\n\n{_job_table(exp, jobs)}\n\n"
            f"## Thứ tự phiên\n\n{exp.order}\n\n"
            "## Mỗi phiên\n\n"
            "1. Add Input: **experiment-data-v3** (bản 29/10 sau QC) + output các việc ở cột *Khôi phục*.\n"
            "2. Ô cấu hình: `JOB`, `RESTORE_INPUTS` (**đúng thứ tự cột Khôi phục** — zip sau ghi đè zip trước, output mới nhất cuối), "
            "`REPO_COMMIT` (giữ một commit cho mọi phiên của một đợt). `kaggle_jobs.py run` dừng (mã 2) nếu config.json của đầu vào "
            "không đúng bản (`requires_cfg`, v3.1.7).\n"
            "3. Save Version → Save & Run All. Output: `/kaggle/working/<JOB>_results.zip` → lưu thành Dataset `kltn-<job>`.\n"
            "4. Mã 124 (quá `MAX_HOURS`): phiên mới cùng JOB, khôi phục zip vừa lưu — script tự tiếp tục từ checkpoint.\n")
    cells[0]["source"] = _lines(head)
    cfg = _src(cells[1]).split("\n")
    i = next(k for k, line in enumerate(cfg) if line.startswith(JOB_LINE_PREFIX))
    allowed = ", ".join(repr(n) for n in exp.jobs)
    cfg[i:i + 1] = [f"JOBS_IN_NOTEBOOK = [{allowed}]",
                    f"JOB = {exp.jobs[0]!r}" + "  # chọn một trong JOBS_IN_NOTEBOOK (bảng ở đầu notebook)"]
    cfg.append("assert JOB in JOBS_IN_NOTEBOOK, f'{JOB} không thuộc notebook này — dùng notebook của thí nghiệm tương ứng'")
    cells[1]["source"] = _lines("\n".join(cfg))
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb = {k: copy.deepcopy(v) for k, v in runner.items() if k != "cells"}
    nb["cells"] = cells
    return nb


def render(nb: dict) -> str:
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra notebook đã khớp, không ghi")
    args = ap.parse_args(argv)
    jobs = _load_jobs()
    covered = [n for e in EXPERIMENTS for n in e.jobs]
    missing, dup = set(jobs) - set(covered), {n for n in covered if covered.count(n) > 1}
    unknown = set(covered) - set(jobs)
    if missing or dup or unknown:
        print(f"danh mục lệch: thiếu {sorted(missing)}, trùng {sorted(dup)}, không có {sorted(unknown)}")
        return 1
    runner = json.loads(RUNNER.read_text(encoding="utf-8"))
    stale = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for exp in EXPERIMENTS:
        text, path = render(build(exp, runner, jobs)), OUT_DIR / exp.file
        current = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else None
        if current != text:
            stale.append(path.name)
            if not args.check:
                path.write_text(text, encoding="utf-8", newline="\n")
    extra = sorted(p.name for p in OUT_DIR.glob("*.ipynb") if p.name not in {e.file for e in EXPERIMENTS})
    if args.check:
        if stale or extra:
            print("notebook lệch (chạy python scripts/make_kaggle_notebooks.py):", stale, "thừa:", extra)
            return 1
        print(f"OK {len(EXPERIMENTS)} notebook khớp")
        return 0
    print(f"ghi {len(stale)}/{len(EXPERIMENTS)} notebook vào {OUT_DIR.relative_to(REPO)}" + (f"; thừa (xoá tay): {extra}" if extra else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
