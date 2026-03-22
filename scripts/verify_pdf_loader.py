"""
PDF Loader 验证脚本

用于测试 PdfLoader 是否能正确解析真实的 PDF 文件。

用法：
    python scripts/verify_pdf_loader.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.libs.loader import PdfLoader


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"{'-' * 80}")


def verify_pdf_file(pdf_path: str) -> bool:
    """
    验证单个 PDF 文件

    Args:
        pdf_path: PDF 文件路径

    Returns:
        True if successful, False otherwise
    """
    print_separator(f"验证文件: {pdf_path}")

    loader = PdfLoader()

    try:
        # 1. 加载 PDF
        print(f"[INFO] 正在加载 PDF...")
        document = loader.load(pdf_path)

        # 2. 基本信息
        print(f"[OK] 加载成功！")
        print(f"\n[INFO] 基本信息:")
        print(f"  - Document ID: {document.id}")
        print(f"  - 文本长度: {len(document.text):,} 字符")
        print(f"  - 文件名: {document.metadata.get('file_name', 'N/A')}")
        print(f"  - 文件大小: {document.metadata.get('file_size', 0):,} bytes")
        print(f"  - 文档类型: {document.metadata.get('doc_type', 'N/A')}")
        print(f"  - 源路径: {document.metadata.get('source_path', 'N/A')}")

        # 3. 文本内容预览
        print(f"\n[INFO] 文本内容预览 (前 500 字符):")
        preview = document.text[:500]
        print(f"  {preview}...")
        if len(document.text) > 500:
            print(f"  ... (还有 {len(document.text) - 500:,} 字符)")

        # 4. 图片信息
        images = document.metadata.get('images', [])
        print(f"\n[INFO] 图片信息:")
        if images:
            print(f"  - 提取到 {len(images)} 张图片")
            for i, img in enumerate(images[:5], 1):  # 只显示前5张
                print(f"    [{i}] ID: {img.id}")
                print(f"        页码: {img.page}")
                print(f"        路径: {img.path}")
                print(f"        文本位置: {img.text_offset}")
                if img.position:
                    print(f"        PDF坐标: {img.position}")
            if len(images) > 5:
                print(f"    ... (还有 {len(images) - 5} 张图片)")

            # 检查图片文件是否存在
            print(f"\n[INFO] 验证图片文件:")
            missing_images = []
            for img in images:
                if not Path(img.path).exists():
                    missing_images.append(img.id)

            if missing_images:
                print(f"  [WARN] 警告: {len(missing_images)} 张图片文件不存在")
                for img_id in missing_images[:3]:
                    print(f"    - {img_id}")
            else:
                print(f"  [OK] 所有图片文件都存在")
        else:
            print(f"  [INFO] 未提取到图片 (可能 PDF 不包含图片)")

        # 5. 检查图片占位符
        image_placeholders = document.text.count('[IMAGE:')
        print(f"\n[INFO] 图片占位符:")
        print(f"  - 发现 {image_placeholders} 个图片占位符")
        if image_placeholders > 0:
            import re
            placeholders = re.findall(r'\[IMAGE: ([^\]]+)\]', document.text)
            print(f"  - 图片 ID: {', '.join(placeholders[:5])}")
            if len(placeholders) > 5:
                print(f"    ... (还有 {len(placeholders) - 5} 个)")

        print_separator("验证通过 [OK]")
        return True

    except FileNotFoundError as e:
        print(f"[ERROR] 错误: 文件不存在 - {e}")
        return False
    except ValueError as e:
        print(f"[ERROR] 错误: 不支持的文件格式 - {e}")
        return False
    except RuntimeError as e:
        print(f"[ERROR] 错误: PDF 解析失败 - {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 未知错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print_separator("PDF Loader 验证工具")

    # 测试文件路径
    test_files = [
        "D:/projects/MODULAR-RAG-MCP-SERVER/docs/test/simple.pdf",
        "D:/projects/MODULAR-RAG-MCP-SERVER/docs/test/with_images.pdf",
    ]

    results = {}

    for pdf_path in test_files:
        # 检查文件是否存在
        if not Path(pdf_path).exists():
            print(f"[WARN] 跳过: 文件不存在 - {pdf_path}")
            results[pdf_path] = False
            continue

        # 验证文件
        success = verify_pdf_file(pdf_path)
        results[pdf_path] = success

    # 总结
    print_separator("验证总结")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    print(f"总测试文件: {total}")
    print(f"[OK] 通过: {passed}")
    print(f"[FAIL] 失败: {failed}")

    if failed == 0:
        print(f"\n[SUCCESS] 所有 PDF 文件解析成功！")
        return 0
    else:
        print(f"\n[WARN] 部分文件解析失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    exit(main())
