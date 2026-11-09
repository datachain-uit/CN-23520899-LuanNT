"""Đọc dữ liệu, chia CV theo bệnh nhân, lọc sơ bộ không dùng nhãn, lấy mẫu con, bản sao tương quan.

Hai định dạng được chấp nhận (survey v3, QĐ15):
* ``experiment-data-perm/<bộ>/``: ``expression.h5`` (key ``expression``, mẫu × gen, index = sample_id),
  ``metadata.csv`` (index sample_id; cột patient_id, class, label ∈ {0, 1}) — bộ TCGA v2 và bộ sanity.
* ``experiment-data-geo-target/<thư mục>/``: ``expression.h5`` (một node duy nhất) và ``metadata.csv`` có cột
  ``sample_id``, ``label`` (0 = LUAD, 1 = LUSC), ``class``; nếu thiếu ``patient_id`` thì patient_id = sample_id và
  config ghi ``patient_id_assumed_from_sample_id = true`` (O15: phải xác minh bằng kiểm tra mẫu trùng).
Không gian gen của cohort GEO = giao với TCGA (B8) khi gọi ``load_dataset(..., gene_space="reference")``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config as C


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Dataset:
    name: str
    X: np.ndarray          # mẫu × gen, float64, log2 gốc (KHÔNG chuẩn hóa)
    y: np.ndarray          # 0 = LUAD, 1 = LUSC (bài toán chính); sanity: 0 = normal, 1 = tumor
    groups: np.ndarray     # patient_id
    genes: np.ndarray      # tên gen (str)
    sample_ids: np.ndarray
    files_sha256: dict
    data_dir: Path
    notes: dict = field(default_factory=dict)   # patient_id_assumed_from_sample_id, gene_space, n_genes_dropped...

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def p(self) -> int:
        return self.X.shape[1]


def _read_expression(path: Path) -> pd.DataFrame:
    try:
        return pd.read_hdf(path, key="expression")
    except (KeyError, TypeError, ValueError):
        return pd.read_hdf(path)


def _read_metadata(path: Path) -> tuple[pd.DataFrame, bool]:
    """Trả về (metadata có index sample_id, patient_id_assumed)."""
    meta = pd.read_csv(path)
    if "sample_id" in meta.columns:
        meta = meta.set_index("sample_id")
    else:
        meta = pd.read_csv(path, index_col=0)
    assumed = False
    if "patient_id" not in meta.columns:
        meta["patient_id"] = meta.index.astype(str)
        assumed = True
    return meta, assumed


def resolve_dataset_dir(root: Path, name: str) -> Path:
    """Thư mục của một bộ dưới KLTN_DATA_ROOT: thư mục đầu tiên tồn tại trong C.dataset_dir_candidates(name);
    nếu chưa có thư mục nào thì trả về thư mục chính (để script tạo dữ liệu/ghi split dùng được)."""
    root = Path(root)
    candidates = [root / rel for rel in C.dataset_dir_candidates(name)]
    for d in candidates:
        if (d / "expression.h5").exists() or d.is_dir():
            return d
    return candidates[0]


def dataset_path(root: Path, name: str) -> Path:
    """Tên giữ cho đội code (commit 3059ce5): như resolve_dataset_dir()."""
    return resolve_dataset_dir(root, name)


def read_gene_list(root: Path, name: str) -> np.ndarray:
    """Danh sách gen của một bộ mà không nạp toàn bộ ma trận vào bộ nhớ lâu dài."""
    d = resolve_dataset_dir(root, name)
    x_df = _read_expression(d / "expression.h5")
    return x_df.columns.to_numpy().astype(str)


def _strict_n_enabled(strict_n: bool | None) -> bool:
    if strict_n is not None:
        return bool(strict_n)
    return os.environ.get("KLTN_STRICT_N", "").strip().lower() in {"1", "true", "yes"}


def load_dataset(root: Path, name: str, gene_space: str | None = "reference", data_dir: str | Path | None = None,
                 strict_n: bool | None = None) -> Dataset:
    """Đọc một bộ dữ liệu; kiểm tra khớp index, nhãn nhị phân, không NaN.

    gene_space = "reference": cohort khác GENE_SPACE_REFERENCE chỉ giữ gen có trong bộ tham chiếu (TCGA);
    gene_space = None: giữ nguyên. Thứ tự gen của bộ được giữ (lọc theo mặt nạ).
    strict_n (mặc định: biến môi trường KLTN_STRICT_N): dừng nếu số mẫu mỗi lớp khác C.DATASET_EXPECTED_N
    (v3.1: chặn chạy nhầm bản dữ liệu, vd. GSE30219 chưa sửa R5); luôn ghi notes["n_per_class_expected"].
    """
    d = Path(data_dir) if data_dir is not None else resolve_dataset_dir(root, name)
    x_df = _read_expression(d / "expression.h5")
    meta, assumed = _read_metadata(d / "metadata.csv")
    x_df.index = x_df.index.astype(str)
    meta.index = meta.index.astype(str)
    if list(x_df.index) != list(meta.index):
        if set(x_df.index) == set(meta.index):
            meta = meta.loc[x_df.index]
        else:
            raise ValueError(f"{name}: expression.h5 và metadata.csv không cùng tập sample_id")
    y = meta["label"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{name}: nhãn phải là 0/1, gặp {np.unique(y)}")
    genes = x_df.columns.to_numpy().astype(str)
    if len(set(genes)) != len(genes):
        raise ValueError(f"{name}: tên gen trùng lặp trong expression.h5")
    X = np.ascontiguousarray(x_df.to_numpy(dtype=np.float64))
    if np.isnan(X).any():
        raise ValueError(f"{name}: có NaN trong biểu hiện")
    notes = {"patient_id_assumed_from_sample_id": assumed, "gene_space": "all", "n_genes_before_gene_space": int(X.shape[1]),
             "data_dir": str(d)}
    n_per_class = (int((y == 0).sum()), int((y == 1).sum()))
    expected = C.DATASET_EXPECTED_N.get(name)
    notes.update({"n_per_class": list(n_per_class), "n_per_class_expected": list(expected) if expected else None,
                  "n_matches_expected": (tuple(expected) == n_per_class) if expected else None})
    if expected and tuple(expected) != n_per_class:
        msg = f"{name}: số mẫu mỗi lớp {n_per_class} khác kỳ vọng {tuple(expected)} (C.DATASET_EXPECTED_N; {d})"
        if _strict_n_enabled(strict_n):
            raise ValueError(msg + " — KLTN_STRICT_N=1: dừng")
        print("[data] CẢNH BÁO " + msg, flush=True)
    if gene_space == "reference" and name != C.GENE_SPACE_REFERENCE and name != C.SANITY_DATASET:
        ref_dir = resolve_dataset_dir(root, C.GENE_SPACE_REFERENCE)
        if (ref_dir / "expression.h5").exists():
            ref_genes = set(read_gene_list(root, C.GENE_SPACE_REFERENCE))
            keep = np.array([g in ref_genes for g in genes], dtype=bool)
            if keep.sum() == 0:
                raise ValueError(f"{name}: không có gen chung với {C.GENE_SPACE_REFERENCE}")
            X, genes = np.ascontiguousarray(X[:, keep]), genes[keep]
            notes.update({"gene_space": f"intersection with {C.GENE_SPACE_REFERENCE}",
                          "n_genes_dropped": int((~keep).sum())})
        else:
            notes["gene_space"] = f"all (reference {C.GENE_SPACE_REFERENCE} not found, intersection NOT applied)"
    groups = meta["patient_id"].astype(str).to_numpy()
    return Dataset(name=name, X=X, y=y, groups=groups, genes=genes, sample_ids=meta.index.to_numpy(), data_dir=d,
                   files_sha256={f: sha256(d / f) for f in ("expression.h5", "metadata.csv")}, notes=notes)


def drop_genes(ds: Dataset, genes) -> Dataset:
    """Bản sao của ds không có các gen trong `genes` (ablation loại 4 gen IHC — RQ1/RQ2 của khoá luận)."""
    drop = set(str(g) for g in genes)
    keep = np.array([g not in drop for g in ds.genes], dtype=bool)
    out = Dataset(name=ds.name, X=np.ascontiguousarray(ds.X[:, keep]), y=ds.y, groups=ds.groups, genes=ds.genes[keep],
                  sample_ids=ds.sample_ids, files_sha256=ds.files_sha256, data_dir=ds.data_dir, notes=dict(ds.notes))
    out.notes["excluded_genes"] = sorted(drop & set(ds.genes.tolist()))
    out.notes["excluded_genes_not_found"] = sorted(drop - set(ds.genes.tolist()))
    return out


def common_genes(a: Dataset, b: Dataset) -> np.ndarray:
    """Gen chung của hai bộ, theo thứ tự của bộ a."""
    sb = set(b.genes.tolist())
    return np.array([g for g in a.genes if g in sb])


def align_to_genes(ds: Dataset, genes) -> np.ndarray:
    """Ma trận của ds theo đúng thứ tự `genes` (mọi gen phải có trong ds)."""
    index = {g: j for j, g in enumerate(ds.genes)}
    cols = [index[g] for g in genes]
    return np.ascontiguousarray(ds.X[:, cols])


# ---------------------------------------------------------------------------------------------------
# Chia CV lặp theo bệnh nhân
# ---------------------------------------------------------------------------------------------------
def repeated_patient_splits(y: np.ndarray, groups: np.ndarray, seeds=C.REPEAT_SEEDS, n_folds: int = C.N_FOLDS):
    """Sinh các lần chia (repeat, fold, train_idx, test_idx) phân tầng theo nhãn, nhóm theo bệnh nhân.

    Nếu mỗi bệnh nhân chỉ có một mẫu (trường hợp của ba bộ giai đoạn) thì StratifiedKFold(shuffle)
    là đủ và trùng với cách chia của RQ0′; nếu có bệnh nhân nhiều mẫu thì dùng StratifiedGroupKFold.
    Luôn assert không có bệnh nhân nằm ở cả train và test.
    """
    one_sample_per_patient = len(np.unique(groups)) == len(groups)
    out = []
    for repeat, seed in enumerate(seeds):
        if one_sample_per_patient:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splitter = cv.split(np.zeros(len(y)), y)
        else:
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splitter = cv.split(np.zeros(len(y)), y, groups)
        for fold, (tr, te) in enumerate(splitter):
            assert set(groups[tr]).isdisjoint(groups[te]), f"repeat {repeat} fold {fold}: trùng bệnh nhân"
            assert len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2
            out.append((repeat, seed, fold, tr, te))
    return out


def splits_to_frame(ds: Dataset, splits) -> pd.DataFrame:
    rows = []
    for repeat, seed, fold, tr, te in splits:
        role = np.full(ds.n, "", dtype=object)
        role[tr], role[te] = "train", "test"
        rows.append(pd.DataFrame({"cohort": ds.name, "repeat": repeat, "seed": seed, "fold": fold,
                                  "sample_id": ds.sample_ids, "patient_id": ds.groups, "label": ds.y, "role": role}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------------
# Lọc sơ bộ không dùng nhãn (trong fold)
# ---------------------------------------------------------------------------------------------------
def variance_prefilter(X_train: np.ndarray, p0: int = C.PREFILTER_P0) -> np.ndarray:
    """Chỉ số của p0 gen có phương sai lớn nhất TRÊN PHẦN TRAIN (không đọc nhãn, không đọc test).

    Trả về mảng chỉ số đã sắp xếp tăng dần để giữ thứ tự cột ổn định.
    """
    if p0 is None or p0 >= X_train.shape[1]:
        return np.arange(X_train.shape[1])
    var = X_train.var(axis=0, ddof=1)
    keep = np.argpartition(-var, p0 - 1)[:p0]
    return np.sort(keep)


def ftest_prefilter(X_train: np.ndarray, y_train: np.ndarray, p0: int) -> np.ndarray:
    """Độ nhạy RQ1 (QĐ53b): p0 gen có thống kê F (ANOVA hai lớp) lớn nhất TRÊN PHẦN TRAIN của fold — có dùng nhãn nhưng nằm trong CV
    nên không rò rỉ. Hoà theo chỉ số cột (ổn định); gen hằng (F = NaN) xếp cuối. Trả về chỉ số đã sắp xếp tăng dần."""
    from sklearn.feature_selection import f_classif
    if p0 is None or p0 >= X_train.shape[1]:
        return np.arange(X_train.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        f, _ = f_classif(X_train, y_train)
    f = np.nan_to_num(f, nan=-np.inf)
    order = np.lexsort((np.arange(len(f)), -f))
    return np.sort(order[:p0])


PREFILTER_RULES = ("variance", "ftest")


def prefilter(X_train: np.ndarray, y_train: np.ndarray, p0: int, rule: str = "variance") -> np.ndarray:
    """Lọc sơ bộ trong fold: "variance" (mặc định, không dùng nhãn, M14) hoặc "ftest" (độ nhạy QĐ53b)."""
    if rule == "variance":
        return variance_prefilter(X_train, p0)
    if rule == "ftest":
        return ftest_prefilter(X_train, y_train, p0)
    raise ValueError(f"prefilter rule không hợp lệ: {rule}")


# ---------------------------------------------------------------------------------------------------
# Lấy mẫu con độc lập (E4) và bản sao tương quan (E5a)
# ---------------------------------------------------------------------------------------------------
def stratified_subsample(y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Chỉ số của một tập con phân tầng theo nhãn với tỉ lệ `fraction` số mẫu (mỗi mẫu = một bệnh nhân)."""
    idx = []
    for c in (0, 1):
        cand = np.flatnonzero(y == c)
        m = max(2, int(round(fraction * len(cand))))
        idx.append(rng.choice(cand, size=m, replace=False))
    return np.sort(np.concatenate(idx))


def add_correlated_copies(X: np.ndarray, genes: np.ndarray, rho: float, n_copies: int,
                          rng: np.random.Generator):
    """Thêm `n_copies` bản sao tương quan của các gen phương sai lớn nhất (E5a), KHÔNG dùng nhãn.

    x' = ρ·z + sqrt(1 − ρ²)·ε với z = x đã z-score trên chính ma trận X (phần train / tập con),
    ε ~ N(0, 1). Bản sao được đổi về thang của gen gốc (nhân σ, cộng μ) để các phương pháp không
    chuẩn hóa không phân biệt được bản sao chỉ qua thang đo. Trả về (X_mới, genes_mới, chỉ số gốc,
    chỉ số bản sao).
    """
    var = X.var(axis=0, ddof=1)
    src = np.argsort(-var, kind="stable")[:n_copies]
    mu, sd = X[:, src].mean(0), X[:, src].std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    z = (X[:, src] - mu) / sd
    eps = rng.standard_normal(z.shape)
    copies = (rho * z + np.sqrt(1.0 - rho**2) * eps) * sd + mu
    X_new = np.hstack([X, copies])
    copy_names = np.array([f"{genes[j]}__copy_rho{rho}" for j in src])
    genes_new = np.concatenate([genes, copy_names])
    copy_idx = np.arange(X.shape[1], X.shape[1] + n_copies)
    return X_new, genes_new, src, copy_idx
