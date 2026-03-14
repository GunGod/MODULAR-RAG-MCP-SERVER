"""
Smoke tests for package imports (A2阶段验收测试)

这些测试验证项目关键包可以被正确导入，确保基本Python包结构正确。
这是A2阶段的核心验收标准：至少1个冒烟测试。

根据A2 spec要求：只做关键包 import 校验
"""


def test_import_mcp_server():
    """测试 mcp_server 包可以导入"""
    import mcp_server
    assert mcp_server is not None


def test_import_core():
    """测试 core 包可以导入"""
    import core
    assert core is not None


def test_import_ingestion():
    """测试 ingestion 包可以导入"""
    import ingestion
    assert ingestion is not None


def test_import_libs():
    """测试 libs 包可以导入"""
    import libs
    assert libs is not None


def test_import_observability():
    """测试 observability 包可以导入"""
    import observability
    assert observability is not None
