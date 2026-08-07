import ast
import glob


def test_python_files_parse():
    """Simple smoke test: ensure all src/*.py files are syntactically valid."""
    py_files = glob.glob("src/*.py")
    assert py_files, "No python files found in src/"
    for path in py_files:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        # raises SyntaxError on failure
        ast.parse(src)
