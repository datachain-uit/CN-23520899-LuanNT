"""PD-04 (Review 03/11, QĐ37): notebook Kaggle R kiểm cài ptLasso, glmtrans và chạy ví dụ gốc.

Máy phát triển và CI Python không có R, nên test chỉ kiểm cấu trúc: script và notebook có đúng các bước, kết quả được lưu kể cả khi
cài lỗi, và dấu ngoặc cân bằng. Hành vi thật chỉ được xác nhận khi notebook chạy trên Kaggle (r_baseline_check.csv, sessionInfo.txt).
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "r" / "check_r_baselines.R"
NOTEBOOK = REPO / "notebooks" / "kaggle_r" / "kaggle_rq2_r_baselines.ipynb"


def _strip_r(text: str) -> str:
    """Bỏ chuỗi và chú thích R, giữ phần mã để kiểm dấu ngoặc."""
    out, i, quote = [], 0, None
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'`":
            quote = c
        elif c == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        else:
            out.append(c)
        i += 1
    assert quote is None, "chuỗi không đóng"
    return "".join(out)


def _assert_balanced(text: str) -> None:
    pairs, stack = {")": "(", "]": "[", "}": "{"}, []
    for c in _strip_r(text):
        if c in "([{":
            stack.append(c)
        elif c in pairs:
            assert stack and stack.pop() == pairs[c], f"dấu ngoặc lệch ở {c}"
    assert not stack, f"chưa đóng: {stack}"


def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return nb, ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_balance_checker_catches_broken_code():
    _assert_balanced('f <- function(x) { c("(", x) } # (')
    with pytest.raises(AssertionError):
        _assert_balanced("f <- function(x) { c(x }")


def test_script_is_balanced_and_runs_both_manual_binomial_examples():
    text = SCRIPT.read_text(encoding="utf-8")
    _assert_balanced(text)
    code = _strip_r(text)
    assert "binomial.example.data(" in code and "ptLasso::ptLasso(" in code and "glmtrans::glmtrans(" in code and "glmtrans::models(" in code
    assert 'set.seed(0, kind = "L' in text and "set.seed(1234" in text              # đúng seed của ví dụ trong manual CRAN
    assert 'family = "binomial"' in text and 'type.measure = "auc"' in text


def test_script_records_failures_instead_of_stopping_before_writing_outputs():
    code = _strip_r(SCRIPT.read_text(encoding="utf-8"))
    text = SCRIPT.read_text(encoding="utf-8")
    assert "tryCatch(" in code and "sessionInfo()" in code and "conditionMessage(" in code
    for name in ("r_baseline_check.csv", "sessionInfo.txt", "examples_output.txt"):
        assert name in text
    assert text.index("sessionInfo.txt") < text.index("quit(")                       # ghi xong mới thoát với mã lỗi


def test_script_installs_outside_output_folder():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "R_BASELINE_LIB" in text and ".libPaths(" in text and "/kaggle/" not in text


def test_notebook_uses_r_kernel_and_only_dispatches_to_script():
    nb, cells = _cells()
    assert nb["metadata"]["kernelspec"]["name"] == "ir" and nb["metadata"]["kernelspec"]["language"] == "R"
    for src in cells:
        _assert_balanced(src)
        assert "install.packages" not in src and "ptLasso::" not in src and "glmtrans::" not in src      # M18: lệnh nằm trong script đã commit
    assert all(c.get("outputs") == [] and c.get("execution_count") is None for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_pins_pushed_commit_and_repo():
    _, cells = _cells()
    code = "\n".join(cells)
    assert "https://github.com/datachain-uit/CN-23520899-LuanNT.git" in code and "'rq2-mvp'" in code
    assert "REPO_COMMIT" in code and "stop(" in code and "--contains" in code


def test_notebook_saves_results_before_reporting_failure():
    _, cells = _cells()
    code = "\n".join(cells)
    assert "scripts/r/check_r_baselines.R" in code and "r_baselines_results.zip" in code
    assert code.index("r_baselines_results.zip") < code.rindex("stop(")              # lưu zip rồi mới báo lỗi
