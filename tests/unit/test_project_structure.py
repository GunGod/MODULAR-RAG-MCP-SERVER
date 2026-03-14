"""
Unit tests for project structure (Task A1)

These tests verify that the project directory structure is correctly initialized
according to the architecture specification in DEV_SPEC.md section 5.2.
"""

import os
from pathlib import Path


class TestProjectStructure:
    """Test that all required directories and files exist."""

    @staticmethod
    def test_src_directory_exists():
        """Test that src/ directory exists."""
        src_path = Path("src")
        assert src_path.exists(), "src/ directory must exist"
        assert src_path.is_dir(), "src/ must be a directory"

    @staticmethod
    def test_config_directory_exists():
        """Test that config/ directory exists."""
        config_path = Path("config")
        assert config_path.exists(), "config/ directory must exist"
        assert config_path.is_dir(), "config/ must be a directory"

    @staticmethod
    def test_data_directory_exists():
        """Test that data/ directory exists."""
        data_path = Path("data")
        assert data_path.exists(), "data/ directory must exist"
        assert data_path.is_dir(), "data/ must be a directory"

    @staticmethod
    def test_tests_directory_exists():
        """Test that tests/ directory exists."""
        tests_path = Path("tests")
        assert tests_path.exists(), "tests/ directory must exist"
        assert tests_path.is_dir(), "tests/ must be a directory"

    @staticmethod
    def test_logs_directory_exists():
        """Test that logs/ directory exists."""
        logs_path = Path("logs")
        assert logs_path.exists(), "logs/ directory must exist"
        assert logs_path.is_dir(), "logs/ must be a directory"

    @staticmethod
    def test_main_py_exists():
        """Test that main.py exists."""
        main_py = Path("main.py")
        assert main_py.exists(), "main.py must exist"
        assert main_py.is_file(), "main.py must be a file"

    @staticmethod
    def test_pyproject_toml_exists():
        """Test that pyproject.toml exists."""
        pyproject = Path("pyproject.toml")
        assert pyproject.exists(), "pyproject.toml must exist"
        assert pyproject.is_file(), "pyproject.toml must be a file"

    @staticmethod
    def test_requirements_txt_exists():
        """Test that requirements.txt exists."""
        requirements = Path("requirements.txt")
        assert requirements.exists(), "requirements.txt must exist"
        assert requirements.is_file(), "requirements.txt must be a file"

    @staticmethod
    def test_mcp_server_directory_exists():
        """Test that src/mcp_server/ directory exists."""
        mcp_server_path = Path("src/mcp_server")
        assert mcp_server_path.exists(), "src/mcp_server/ directory must exist"
        assert mcp_server_path.is_dir(), "src/mcp_server/ must be a directory"

    @staticmethod
    def test_core_directory_exists():
        """Test that src/core/ directory exists."""
        core_path = Path("src/core")
        assert core_path.exists(), "src/core/ directory must exist"
        assert core_path.is_dir(), "src/core/ must be a directory"

    @staticmethod
    def test_ingestion_directory_exists():
        """Test that src/ingestion/ directory exists."""
        ingestion_path = Path("src/ingestion")
        assert ingestion_path.exists(), "src/ingestion/ directory must exist"
        assert ingestion_path.is_dir(), "src/ingestion/ must be a directory"

    @staticmethod
    def test_libs_directory_exists():
        """Test that src/libs/ directory exists."""
        libs_path = Path("src/libs")
        assert libs_path.exists(), "src/libs/ directory must exist"
        assert libs_path.is_dir(), "src/libs/ must be a directory"

    @staticmethod
    def test_observability_directory_exists():
        """Test that src/observability/ directory exists."""
        observability_path = Path("src/observability")
        assert observability_path.exists(), "src/observability/ directory must exist"
        assert observability_path.is_dir(), "src/observability/ must be a directory"

    @staticmethod
    def test_all_package_init_files_exist():
        """Test that all Python packages have __init__.py files."""
        required_packages = [
            "src",
            "src/mcp_server",
            "src/mcp_server/tools",
            "src/core",
            "src/core/query_engine",
            "src/core/response",
            "src/core/trace",
            "src/ingestion",
            "src/ingestion/chunking",
            "src/ingestion/transform",
            "src/ingestion/embedding",
            "src/ingestion/storage",
            "src/libs",
            "src/libs/loader",
            "src/libs/llm",
            "src/libs/embedding",
            "src/libs/splitter",
            "src/libs/vector_store",
            "src/libs/reranker",
            "src/libs/evaluator",
            "src/observability",
            "src/observability/dashboard",
            "src/observability/dashboard/pages",
            "src/observability/dashboard/services",
            "src/observability/evaluation",
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/e2e",
        ]

        for package in required_packages:
            init_file = Path(package) / "__init__.py"
            assert init_file.exists(), f"{init_file} must exist"
            assert init_file.is_file(), f"{init_file} must be a file"

    @staticmethod
    def test_main_py_importable():
        """Test that main.py can be imported (syntax check)."""
        try:
            # Try to compile main.py
            with open("main.py", "r", encoding="utf-8") as f:
                compile(f.read(), "main.py", "exec")
        except SyntaxError as e:
            raise AssertionError(f"main.py has syntax error: {e}")

    @staticmethod
    def test_pyproject_toml_valid():
        """Test that pyproject.toml is valid TOML."""
        try:
            import tomli
            with open("pyproject.toml", "rb") as f:
                tomli.load(f)
        except ImportError:
            # tomli not available, skip this test
            pass
        except Exception as e:
            raise AssertionError(f"pyproject.toml is not valid TOML: {e}")
