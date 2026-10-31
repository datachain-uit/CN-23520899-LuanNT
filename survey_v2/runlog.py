"""Truy vết và ghi kết quả cho các script survey: provenance (scripts/provenance.py) + kiểm tra E2 (Y13).

Mọi script survey gọi `start_run(__file__, params)` ở đầu main():
  1. provenance(): dừng nếu không phải git clone hoặc có file đã theo dõi bị sửa chưa commit;
  2. check_traceable(): script đang chạy phải nằm trong index Git và HEAD phải có trên remote (Y13);
     tạm bỏ qua bằng KLTN_ALLOW_UNPUSHED=1 (khi đó config ghi traceable = false và KHÔNG được trích).
Kết quả ghi dưới $KLTN_DATA_ROOT/results/survey_v2/<E>/..., kèm config.json có commit hash và SHA-256 dữ liệu.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.provenance import DATA_ROOT, add_data_files, provenance, stamp  # noqa: E402

from . import config as C  # noqa: E402

ENV_ALLOW_UNPUSHED = "KLTN_ALLOW_UNPUSHED"


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def is_tracked(path, repo=REPO) -> bool:
    """Return whether *path* is tracked by the repository at *repo*.

    ``git ls-files`` is evaluated from ``repo`` and expects a repository-relative
    pathspec.  Supplying an absolute path happens to work in some Windows Git
    installations, but fails on Linux (including Kaggle), incorrectly marking
    an otherwise clean, pushed run as non-traceable.
    """
    try:
        repo = Path(repo).resolve()
        relative_path = Path(path).resolve().relative_to(repo).as_posix()
        _git("ls-files", "--error-unmatch", relative_path, cwd=repo)
        return True
    except (subprocess.CalledProcessError, OSError, ValueError):
        return False


def is_pushed(repo=REPO) -> bool:
    try:
        _git("fetch", "--quiet", cwd=repo)
    except subprocess.CalledProcessError:
        pass
    try:
        return bool(_git("branch", "-r", "--contains", "HEAD", cwd=repo))
    except subprocess.CalledProcessError:
        return False


def check_traceable(script, repo=REPO, require_push: bool = True) -> list[str]:
    """Đoạn code E2 (Review 24/10): danh sách lỗi; rỗng = truy vết được."""
    errors = []
    if not is_tracked(script, repo):
        errors.append(f"script chưa được Git theo dõi: {script} (git add + commit + push trước khi chạy)")
    if require_push and not is_pushed(repo):
        errors.append("commit HEAD chưa có trên remote (git push trước khi chạy)")
    return errors


def results_dir(*parts) -> Path:
    d = DATA_ROOT.joinpath(*C.RESULT_DIR, *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def start_run(script, params: dict | None = None, data_files=()) -> dict:
    """provenance() + check_traceable(); trả về dict provenance để stamp vào config."""
    prov = provenance(script=script, data_files=data_files, params=params or {})
    allow = os.environ.get(ENV_ALLOW_UNPUSHED, "").strip().lower() in {"1", "true", "yes"}
    errors = check_traceable(script, require_push=True) if prov.get("tracked") else []
    prov["traceable"] = (prov.get("tracked", False) and not errors)
    prov["traceable_errors"] = errors
    if errors and not allow:
        print("[traceable] DỪNG:\n  - " + "\n  - ".join(errors)
              + f"\n  (tạm bỏ qua bằng {ENV_ALLOW_UNPUSHED}=1; kết quả khi đó KHÔNG được trích)", file=sys.stderr)
        raise SystemExit(2)
    return prov


def previous_config(out_dir: Path) -> dict | None:
    """config.json đang có trong thư mục kết quả (nếu có) — dùng để ghi lại commit của lần chạy trước khi tiếp tục (resume)."""
    path = Path(out_dir) / "config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_config(out_dir: Path, prov: dict, config: dict, data_files=()) -> Path:
    if data_files:
        add_data_files(prov, data_files)
    config = dict(config)
    config["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    prev = previous_config(out_dir)
    if prev is not None and prev.get("commit_hash") and prev.get("commit_hash") != prov.get("commit_hash"):
        # v3.1.6 (QĐ28): thư mục đã có kết quả của commit khác (resume / --redo-methods) -> ghi lại để truy vết từng phần
        hist = list(prev.get("resumed_from") or [])
        hist.append({"commit_hash": prev.get("commit_hash"), "generated_at": prev.get("generated_at"),
                     "note": "các dòng/checkpoint không tính lại trong lần chạy này được tạo ở commit này"})
        config["resumed_from"] = hist
    elif prev is not None and prev.get("resumed_from"):
        config["resumed_from"] = prev["resumed_from"]
    stamp(config, prov)
    # The Kaggle acceptance gate reads these fields at the top level.  Keep the
    # full provenance record as well, but do not make callers know its nesting.
    config["traceable"] = bool(prov.get("traceable", False))
    config["traceable_errors"] = list(prov.get("traceable_errors", []))
    path = Path(out_dir) / "config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def all_versions() -> dict:
    from importlib import metadata
    out = {}
    for p in ("numpy", "pandas", "scipy", "scikit-learn", "tables", "skrebate", "joblib", "threadpoolctl", "psutil",
              "xgboost", "lightgbm", "codecarbon"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out
