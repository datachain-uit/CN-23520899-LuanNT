"""PD-04 (Review 03/11, QĐ37): notebook Kaggle tự cài R, kiểm cài ptLasso, glmtrans và chạy ví dụ gốc.

Máy phát triển và CI Python không có R, nên test chỉ kiểm cấu trúc: script và notebook có đúng các bước, kết quả được lưu kể cả khi
cài lỗi, và dấu ngoặc cân bằng. Hành vi thật chỉ được xác nhận khi notebook chạy trên Kaggle (r_baseline_check.csv, sessionInfo.txt).
"""

import ast
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


def test_notebook_runs_on_python_kernel_and_is_valid_python():
    """Phiên Kaggle dùng kernel Python (ipykernel); notebook bản R báo NameError và SyntaxError ở `<-`, `1L`."""
    nb, cells = _cells()
    assert nb["metadata"]["kernelspec"]["name"] == "python3" and nb["metadata"]["kernelspec"]["language"] == "python"
    for src in cells:
        ast.parse(src)
        assert "<-" not in src
    assert all(c.get("outputs") == [] and c.get("execution_count") is None for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_only_dispatches_to_r_script():
    _, cells = _cells()
    code = "\n".join(cells)
    assert "scripts/r/check_r_baselines.R" in code and "Rscript" in code
    for command in ("install.packages", "ptLasso::", "glmtrans::"):                 # M18: lệnh nằm trong script đã commit
        assert command not in code


def test_notebook_installs_own_r_from_conda_forge():
    _, cells = _cells()
    code = "\n".join(cells)
    assert "micromamba" in code and "conda-forge" in code and "r-base" in code
    for dep in ("r-glmnet", "r-caret"):                                             # phụ thuộc nặng lấy bản biên dịch sẵn
        assert dep in code


def test_notebook_pins_pushed_commit_and_repo():
    _, cells = _cells()
    code = "\n".join(cells)
    assert "https://github.com/datachain-uit/CN-23520899-LuanNT.git" in code and "'rq2-mvp'" in code
    assert "REPO_COMMIT = '67bbbe1'" in code and "--contains" in code


def test_notebook_saves_results_before_reporting_failure():
    _, cells = _cells()
    code = "\n".join(cells)
    assert "r_baselines_results" in code and "make_archive" in code
    assert code.index("make_archive") < code.rindex("raise ")                        # lưu zip rồi mới báo lỗi
    assert "setup_error" in code                                                     # lỗi cài R vẫn được ghi vào zip
