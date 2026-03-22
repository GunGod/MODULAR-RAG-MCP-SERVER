"""
C4 任务验收标准验证脚本

严格按照 C4 任务的验收标准，使用真实的 PDF 文件验证 DocumentChunker 的功能。

验收标准：
1. 配置驱动：通过修改 settings.yaml 中的 splitter 配置，产出的 chunk 数量和长度发生相应变化
2. ID 唯一性：每个 Chunk 的 ID 在整个文档中唯一
3. ID 确定性：同一 Document 对象重复切分产生相同的 Chunk ID 序列
4. 元数据完整性：Chunk.metadata 包含所有 Document.metadata 字段 + chunk_index 字段
5. 图片分发正确性：含 [IMAGE: id] 占位符的 chunk 其 metadata["images"] 仅包含该 chunk 引用的图片子集
6. 溯源链接：所有 Chunk.source_ref 正确指向父 Document.id
7. 类型契约：输出的 Chunk 对象符合 core/types.py 中的 Chunk 定义
8. 精确位置追踪：Chunk.start_offset 和 end_offset 直接使用 TextChunk 的精确位置

Author: Modular RAG MCP Server Project
License: MIT
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import load_settings
from src.libs.splitter import SplitterFactory
from src.ingestion.chunking import DocumentChunker
from src.libs.loader import PdfLoader
from src.core.types import Chunk


class C4AcceptanceVerifier:
    """C4 任务验收标准验证器"""

    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def verify(self, pdf_path: str) -> Dict[str, Any]:
        """
        验证单个 PDF 文件

        Args:
            pdf_path: PDF 文件路径

        Returns:
            验证结果字典
        """
        print(f"\n{'='*80}")
        print(f"验证文件: {pdf_path}")
        print(f"{'='*80}\n")

        results = {
            "pdf_path": pdf_path,
            "acceptance_criteria": {},
            "overall_status": "UNKNOWN"
        }

        # 加载配置
        settings = load_settings()

        # 创建 splitter 和 chunker
        splitter = SplitterFactory.create(settings)
        chunker = DocumentChunker(splitter)

        # 加载 PDF 文档
        print("1. 加载 PDF 文档...")
        loader = PdfLoader()
        document = loader.load(pdf_path)

        print(f"   [OK] 文档 ID: {document.id}")
        print(f"   [OK] 文档长度: {len(document.text)} 字符")
        print(f"   [OK] 图片数量: {len(document.metadata.get('images', []))}")

        # 验收标准 1: 配置驱动
        results["acceptance_criteria"]["1_配置驱动"] = self._verify_config_driven(
            document, pdf_path
        )

        # 验收标准 2: ID 唯一性
        results["acceptance_criteria"]["2_ID_唯一性"] = self._verify_id_uniqueness(
            document, chunker
        )

        # 验收标准 3: ID 确定性
        results["acceptance_criteria"]["3_ID_确定性"] = self._verify_id_determinism(
            document, chunker
        )

        # 验收标准 4: 元数据完整性
        results["acceptance_criteria"]["4_元数据完整性"] = self._verify_metadata_completeness(
            document, chunker
        )

        # 验收标准 5: 图片分发正确性
        results["acceptance_criteria"]["5_图片分发正确性"] = self._verify_image_distribution(
            document, chunker
        )

        # 验收标准 6: 溯源链接
        results["acceptance_criteria"]["6_溯源链接"] = self._verify_source_reference(
            document, chunker
        )

        # 验收标准 7: 类型契约
        results["acceptance_criteria"]["7_类型契约"] = self._verify_type_contract(
            document, chunker
        )

        # 验收标准 8: 精确位置追踪
        results["acceptance_criteria"]["8_精确位置追踪"] = self._verify_precise_offsets(
            document, chunker
        )

        # 计算总体状态
        all_passed = all(
            result["status"] == "PASS"
            for result in results["acceptance_criteria"].values()
        )
        results["overall_status"] = "PASS" if all_passed else "FAIL"

        return results

    def _verify_config_driven(
        self, document: Any, pdf_path: str
    ) -> Dict[str, Any]:
        """
        验收标准 1: 配置驱动
        通过修改 settings.yaml 中的 splitter 配置，产出的 chunk 数量和长度发生相应变化
        """
        print("\n[验收标准 1] 配置驱动")
        print("-" * 80)

        self.total_tests += 1

        # 测试不同的 chunk_size 配置
        test_configs = [
            {"chunk_size": 500, "chunk_overlap": 50},
            {"chunk_size": 1000, "chunk_overlap": 100},
            {"chunk_size": 2000, "chunk_overlap": 200},
        ]

        chunk_counts = []
        chunk_lengths = []

        for config in test_configs:
            from src.libs.splitter.recursive_splitter import RecursiveSplitter

            splitter = RecursiveSplitter(**config)
            chunker = DocumentChunker(splitter)
            chunks = chunker.chunk_document(document)

            chunk_counts.append(len(chunks))
            avg_length = sum(len(c.text) for c in chunks) / len(chunks)
            chunk_lengths.append(avg_length)

            print(f"   chunk_size={config['chunk_size']}: "
                  f"{len(chunks)} chunks, "
                  f"avg_length={avg_length:.0f} chars")

        # 验证：chunk_size 越大，chunk 数量应该越少
        is_decreasing = (
            chunk_counts[0] > chunk_counts[1] > chunk_counts[2]
        )

        # 验证：平均长度应该随 chunk_size 增加而增加
        is_increasing = (
            chunk_lengths[0] < chunk_lengths[1] < chunk_lengths[2]
        )

        passed = is_decreasing and is_increasing

        if passed:
            self.passed_tests += 1
            print("   [PASS] PASS: chunk 配置正确影响切分结果")
        else:
            self.failed_tests += 1
            print("   [FAIL] FAIL: chunk 配置未能正确影响切分结果")

        return {
            "status": "PASS" if passed else "FAIL",
            "details": {
                "chunk_counts": chunk_counts,
                "avg_lengths": chunk_lengths,
                "is_decreasing": is_decreasing,
                "is_increasing": is_increasing
            }
        }

    def _verify_id_uniqueness(
        self, document: Any, chunker: DocumentChunker
    ) -> Dict[str, Any]:
        """
        验收标准 2: ID 唯一性
        每个 Chunk 的 ID 在整个文档中唯一
        """
        print("\n[验收标准 2] ID 唯一性")
        print("-" * 80)

        self.total_tests += 1

        chunks = chunker.chunk_document(document)
        chunk_ids = [c.id for c in chunks]

        # 检查是否有重复的 ID
        unique_ids = set(chunk_ids)
        has_duplicates = len(unique_ids) != len(chunk_ids)

        passed = not has_duplicates

        if passed:
            self.passed_tests += 1
            print(f"   [PASS] PASS: {len(chunks)} 个 Chunk 的 ID 全部唯一")
        else:
            self.failed_tests += 1
            print(f"   [FAIL] FAIL: 发现重复的 Chunk ID")

        # 找出重复的 ID（如果有）
        duplicates = []
        if has_duplicates:
            from collections import Counter
            counter = Counter(chunk_ids)
            duplicates = [cid for cid, count in counter.items() if count > 1]
            print(f"   重复的 ID: {duplicates}")

        return {
            "status": "PASS" if passed else "FAIL",
            "details": {
                "total_chunks": len(chunks),
                "unique_ids": len(unique_ids),
                "duplicates": duplicates
            }
        }

    def _verify_id_determinism(
        self, document: Any, chunker: DocumentChunker
    ) -> Dict[str, Any]:
        """
        验收标准 3: ID 确定性
        同一 Document 对象重复切分产生相同的 Chunk ID 序列
        """
        print("\n[验收标准 3] ID 确定性")
        print("-" * 80)

        self.total_tests += 1

        # 对同一文档切分 3 次
        chunks1 = chunker.chunk_document(document)
        chunks2 = chunker.chunk_document(document)
        chunks3 = chunker.chunk_document(document)

        ids1 = [c.id for c in chunks1]
        ids2 = [c.id for c in chunks2]
        ids3 = [c.id for c in chunks3]

        # 检查三次切分的 ID 是否完全相同
        passed = (ids1 == ids2 == ids3)

        if passed:
            self.passed_tests += 1
            print(f"   [PASS] PASS: 3 次切分产生相同的 ID 序列 ({len(ids1)} 个 chunks)")
        else:
            self.failed_tests += 1
            print(f"   [FAIL] FAIL: 3 次切分的 ID 序列不一致")

        # 找出不一致的 ID（如果有）
        differences = []
        if not passed:
            for i, (id1, id2, id3) in enumerate(zip(ids1, ids2, ids3)):
                if not (id1 == id2 == id3):
                    differences.append({
                        "index": i,
                        "ids": [id1, id2, id3]
                    })
            print(f"   不一致的 ID: {len(differences)} 个")

        return {
            "status": "PASS" if passed else "FAIL",
            "details": {
                "chunk_count_1": len(ids1),
                "chunk_count_2": len(ids2),
                "chunk_count_3": len(ids3),
                "are_identical": passed,
                "differences": differences
            }
        }

    def _verify_metadata_completeness(
        self, document: Any, chunker: DocumentChunker
    ) -> Dict[str, Any]:
        """
        验收标准 4: 元数据完整性
        Chunk.metadata 包含所有 Document.metadata 字段 + chunk_index 字段
        """
        print("\n[验收标准 4] 元数据完整性")
        print("-" * 80)

        self.total_tests += 1

        chunks = chunker.chunk_document(document)

        # 获取 Document 的 metadata 字段
        doc_metadata_keys = set(document.metadata.keys())

        # 检查每个 chunk 的 metadata
        missing_metadata_chunks = []
        missing_chunk_index_chunks = []

        for i, chunk in enumerate(chunks):
            # 检查是否继承了所有 Document.metadata
            chunk_metadata_keys = set(chunk.metadata.keys())

            # 移除 chunk 特有的字段（chunk_index, chunk_size, has_images, image_refs）
            chunk_specific_fields = {
                "chunk_index", "chunk_size", "has_images", "image_refs"
            }
            inherited_keys = chunk_metadata_keys - chunk_specific_fields

            # 注意：images 字段是特殊处理的
            # - 有 [IMAGE: id] 占位符的 chunk 应该有 images 字段
            # - 没有占位符的 chunk 不应该有 images 字段
            # 所以我们在检查继承时，需要排除 images 字段
            doc_metadata_keys_without_images = doc_metadata_keys - {"images"}
            inherited_keys_without_checking_images = inherited_keys

            # 如果 chunk 有图片占位符，则应该有 images 字段
            has_image_placeholder = "[IMAGE:" in chunk.text
            if has_image_placeholder:
                # 应该有 images 字段
                if "images" not in inherited_keys:
                    missing_metadata_chunks.append({
                        "chunk_index": i,
                        "chunk_id": chunk.id,
                        "missing_fields": ["images"],
                        "reason": "有图片占位符但缺少 images 字段"
                    })
                    continue
            else:
                # 不应该有 images 字段
                if "images" in inherited_keys:
                    # 这个情况下，不认为它是错误，但跳过检查 images 字段的继承
                    pass

            # 检查其他字段是否都继承了
            if not doc_metadata_keys_without_images.issubset(inherited_keys_without_checking_images):
                missing = doc_metadata_keys_without_images - inherited_keys_without_checking_images
                missing_metadata_chunks.append({
                    "chunk_index": i,
                    "chunk_id": chunk.id,
                    "missing_fields": list(missing)
                })

            # 检查是否包含 chunk_index 字段
            if "chunk_index" not in chunk.metadata:
                missing_chunk_index_chunks.append({
                    "chunk_index": i,
                    "chunk_id": chunk.id
                })

        passed = (
            len(missing_metadata_chunks) == 0 and
            len(missing_chunk_index_chunks) == 0
        )

        if passed:
            self.passed_tests += 1
            print(f"   [PASS] PASS: 所有 {len(chunks)} 个 Chunk 的元数据完整")
            print(f"   - 继承字段: {doc_metadata_keys}")
            print(f"   - 特有字段: chunk_index, chunk_size")
        else:
            self.failed_tests += 1
            print(f"   [FAIL] FAIL: 元数据完整性检查失败")
            if missing_metadata_chunks:
                print(f"   - {len(missing_metadata_chunks)} 个 chunk 缺少继承字段")
                for mc in missing_metadata_chunks[:3]:  # 显示前 3 个
                    print(f"     Chunk {mc['chunk_index']} ({mc['chunk_id'][:20]}...): 缺少 {mc['missing_fields']}")
            if missing_chunk_index_chunks:
                print(f"   - {len(missing_chunk_index_chunks)} 个 chunk 缺少 chunk_index")

        return {
            "status": "PASS" if passed else "FAIL",
            "details": {
                "total_chunks": len(chunks),
                "doc_metadata_keys": list(doc_metadata_keys),
                "missing_metadata_chunks": missing_metadata_chunks,
                "missing_chunk_index_chunks": missing_chunk_index_chunks
            }
        }

    def _verify_image_distribution(
        self, document: Any, chunker: DocumentChunker
    ) -> Dict[str, Any]:
        """
        验收标准 5: 图片分发正确性
        含 [IMAGE: id] 占位符的 chunk 其 metadata["images"] 仅包含该 chunk 引用的图片子集
        """
        print("\n[验收标准 5] 图片分发正确性")
        print("-" * 80)

        self.total_tests += 1

        chunks = chunker.chunk_document(document)
        doc_images = document.metadata.get("images", [])

        # 如果文档没有图片，跳过此测试
        if len(doc_images) == 0:
            print("   [SKIP] SKIP: 文档不包含图片")
            return {
                "status": "SKIP",
                "details": {"reason": "文档不包含图片"}
            }

        print(f"   文档图片总数: {len(doc_images)}")

        # 检查每个 chunk 的图片分发
        incorrect_chunks = []

        for i, chunk in enumerate(chunks):
            has_image_placeholder = "[IMAGE:" in chunk.text
            has_images_field = "images" in chunk.metadata
            has_image_refs = "image_refs" in chunk.metadata

            if has_image_placeholder:
                # 有占位符的 chunk 必须有 images 和 image_refs 字段
                if not has_images_field or not has_image_refs:
                    incorrect_chunks.append({
                        "chunk_index": i,
                        "chunk_id": chunk.id,
                        "reason": "有占位符但缺少 images 或 image_refs 字段"
                    })
                    continue

                # 检查 images 字段是否只包含引用的图片
                image_refs = chunk.metadata["image_refs"]
                chunk_images = chunk.metadata["images"]

                # 提取占位符中的 image_id
                import re
                placeholders = re.findall(r'\[IMAGE:\s*([^\]]+)\]', chunk.text)
                placeholder_ids = set(placeholders)

                # image_refs 中的 id 应该与占位符一致
                ref_ids = set(ref["id"] for ref in image_refs)

                if placeholder_ids != ref_ids:
                    incorrect_chunks.append({
                        "chunk_index": i,
                        "chunk_id": chunk.id,
                        "reason": f"image_refs 与占位符不匹配",
                        "placeholder_ids": list(placeholder_ids),
                        "ref_ids": list(ref_ids)
                    })
                    continue

                # chunk_images 的数量应该与 image_refs 一致
                if len(chunk_images) != len(image_refs):
                    incorrect_chunks.append({
                        "chunk_index": i,
                        "chunk_id": chunk.id,
                        "reason": f"images 数量与 image_refs 不一致",
                        "images_count": len(chunk_images),
                        "image_refs_count": len(image_refs)
                    })
            else:
                # 没有占位符的 chunk 不应该有 images 字段
                if has_images_field:
                    incorrect_chunks.append({
                        "chunk_index": i,
                        "chunk_id": chunk.id,
                        "reason": "无占位符但有 images 字段"
                    })

        passed = len(incorrect_chunks) == 0

        if passed:
            self.passed_tests += 1

            # 统计有图片的 chunk
            chunks_with_images = sum(1 for c in chunks if "[IMAGE:" in c.text)
            print(f"   [PASS] PASS: 图片分发正确")
            print(f"   - 包含图片的 chunks: {chunks_with_images}/{len(chunks)}")
        else:
            self.failed_tests += 1
            print(f"   [FAIL] FAIL: 发现 {len(incorrect_chunks)} 个图片分发错误")
            for error in incorrect_chunks[:3]:  # 只显示前 3 个错误
                print(f"   - Chunk {error['chunk_index']}: {error['reason']}")

        return {
            "status": "PASS" if passed else "FAIL",
            "details": {
                "total_chunks": len(chunks),
                "doc_images_count": len(doc_images),
                "chunks_with_images": sum(1 for c in chunks if "[IMAGE:" in c.text),
                "incorrect_chunks": incorrect_chunks
            }
        }

    def _verify_source_reference(
        self, document: Any, chunker: DocumentChunker
    ) -> Dict[str, Any]:
        """
        验收标准 6: 溯源链接
        所有 Chunk.source_ref 正确指向父 Document.id
        """
        print("\n[验收标准 6] 溯源链接")
        print("-" * 80)

        self.total_tests += 1

        chunks = chunker.chunk_document(document)

        # 检查每个 chunk 的 source_ref
        incorrect_chunks = []

        for i, chunk in enumerate(chunks):
            if chunk.source_ref != document.id:
                incorrect_chunks.append({
                    "chunk_index": i,
                    "chunk_id": chunk.id,
                    "source_ref": chunk.source_ref,
                    "expected": document.id
                })

        passed = len(incorrect_chunks) == 0

        if passed:
            self.passed_tests += 1
            print(f"   [PASS] PASS: 所有 {len(chunks)} 个 Chunk 的 source_ref 正确")
            print(f"   - source_ref: {document.id}")
        else:
            self.failed_tests += 1
            print(f"   [FAIL] FAIL: 发现 {len(incorrect_chunks)} 个 source_ref 错误")

        return {
            "status": "PASS" if passed else "FAIL",
            "details": {
                "total_chunks": len(chunks),
                "document_id": document.id,
                "incorrect_chunks": incorrect_chunks
            }
        }

    def _verify_type_contract(
        self, document: Any, chunker: DocumentChunker
    ) -> Dict[str, Any]:
        """
        验收标准 7: 类型契约
        输出的 Chunk 对象符合 core/types.py 中的 Chunk 定义
        """
        print("\n[验收标准 7] 类型契约")
        print("-" * 80)

        self.total_tests += 1

        chunks = chunker.chunk_document(document)

        # 检查每个 chunk 是否符合 Chunk 类型
        invalid_chunks = []

        for i, chunk in enumerate(chunks):
            # 检查是否是 Chunk 类型
            if not isinstance(chunk, Chunk):
                invalid_chunks.append({
                    "chunk_index": i,
                    "reason": f"类型错误: {type(chunk)}"
                })
                continue

            # 检查必需字段
            required_fields = ["id", "text", "metadata", "source_ref"]
            missing_fields = [
                field for field in required_fields
                if not hasattr(chunk, field) or getattr(chunk, field) is None
            ]

            if missing_fields:
                invalid_chunks.append({
                    "chunk_index": i,
                    "reason": f"缺少字段: {missing_fields}"
                })

            # 检查字段类型
            if not isinstance(chunk.id, str):
                invalid_chunks.append({
                    "chunk_index": i,
                    "reason": f"id 类型错误: {type(chunk.id)}"
                })

            if not isinstance(chunk.text, str):
                invalid_chunks.append({
                    "chunk_index": i,
                    "reason": f"text 类型错误: {type(chunk.text)}"
                })

            if not isinstance(chunk.metadata, dict):
                invalid_chunks.append({
                    "chunk_index": i,
                    "reason": f"metadata 类型错误: {type(chunk.metadata)}"
                })

        passed = len(invalid_chunks) == 0

        if passed:
            self.passed_tests += 1
            print(f"   [PASS] PASS: 所有 {len(chunks)} 个 Chunk 符合类型契约")
            print(f"   - 必需字段: id, text, metadata, source_ref")
        else:
            self.failed_tests += 1
            print(f"   [FAIL] FAIL: 发现 {len(invalid_chunks)} 个类型契约错误")

        return {
            "status": "PASS" if passed else "FAIL",
            "details": {
                "total_chunks": len(chunks),
                "invalid_chunks": invalid_chunks
            }
        }

    def _verify_precise_offsets(
        self, document: Any, chunker: DocumentChunker
    ) -> Dict[str, Any]:
        """
        验收标准 8: 精确位置追踪
        Chunk.start_offset 和 end_offset 直接使用 TextChunk 的精确位置
        """
        print("\n[验收标准 8] 精确位置追踪")
        print("-" * 80)

        self.total_tests += 1

        chunks = chunker.chunk_document(document)

        # 检查每个 chunk 的位置信息
        incorrect_chunks = []

        for i, chunk in enumerate(chunks):
            # 检查是否有 start_offset 和 end_offset
            if not hasattr(chunk, "start_offset") or not hasattr(chunk, "end_offset"):
                incorrect_chunks.append({
                    "chunk_index": i,
                    "chunk_id": chunk.id,
                    "reason": "缺少 start_offset 或 end_offset 字段"
                })
                continue

            # 检查 offset 值的合理性
            if chunk.start_offset < 0 or chunk.end_offset < chunk.start_offset:
                incorrect_chunks.append({
                    "chunk_index": i,
                    "chunk_id": chunk.id,
                    "reason": f"offset 值不合理: start={chunk.start_offset}, end={chunk.end_offset}"
                })
                continue

            # 验证 offset 指向的内容是否与 chunk.text 一致
            expected_text = document.text[chunk.start_offset:chunk.end_offset]
            if expected_text != chunk.text:
                incorrect_chunks.append({
                    "chunk_index": i,
                    "chunk_id": chunk.id,
                    "reason": "offset 指向的内容与 chunk.text 不一致",
                    "expected_length": len(expected_text),
                    "actual_length": len(chunk.text)
                })

        # 检查 offsets 是否单调递增（无重叠、无间隙）
        overlapping_chunks = []
        gap_chunks = []

        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            if current.end_offset > next_chunk.start_offset:
                overlapping_chunks.append({
                    "chunk_index": i,
                    "next_chunk_index": i + 1,
                    "overlap": current.end_offset - next_chunk.start_offset
                })

            if current.end_offset < next_chunk.start_offset:
                gap_chunks.append({
                    "chunk_index": i,
                    "next_chunk_index": i + 1,
                    "gap": next_chunk.start_offset - current.end_offset
                })

        # 如果有 overlap 配置，允许重叠
        chunk_overlap = 200  # SplitterFactory 使用的默认值

        has_unexpected_overlaps = (
            len(overlapping_chunks) > 0 and
            any(o["overlap"] > chunk_overlap for o in overlapping_chunks)
        )

        passed = (
            len(incorrect_chunks) == 0 and
            not has_unexpected_overlaps
        )

        if passed:
            self.passed_tests += 1
            print(f"   [PASS] PASS: 所有 {len(chunks)} 个 Chunk 的位置精确")
            print(f"   - 位置验证: offset 内容与 chunk.text 一致")
            if len(gap_chunks) > 0:
                print(f"   - 注意: 发现 {len(gap_chunks)} 个间隙（可能由于 overlap 配置）")
        else:
            self.failed_tests += 1
            print(f"   [FAIL] FAIL: 精确位置追踪检查失败")
            if len(incorrect_chunks) > 0:
                print(f"   - {len(incorrect_chunks)} 个 chunk 的位置信息错误")
            if has_unexpected_overlaps:
                print(f"   - 发现意外的重叠")

        return {
            "status": "PASS" if passed else "FAIL",
            "details": {
                "total_chunks": len(chunks),
                "incorrect_chunks": incorrect_chunks,
                "overlapping_chunks": overlapping_chunks,
                "gap_chunks": gap_chunks,
                "chunk_overlap": chunk_overlap
            }
        }

    def print_summary(self, results: List[Dict[str, Any]]):
        """打印验证总结"""
        print(f"\n{'='*80}")
        print("验证总结")
        print(f"{'='*80}\n")

        # 总体统计
        total_passed = sum(
            1 for r in results
            if r["overall_status"] == "PASS"
        )
        total_files = len(results)

        print(f"测试文件: {total_files}")
        print(f"通过: {total_passed}")
        print(f"失败: {total_files - total_passed}")
        print(f"\n验收标准测试:")
        print(f"  总测试数: {self.total_tests}")
        print(f"  通过: {self.passed_tests} [PASS]")
        print(f"  失败: {self.failed_tests} [FAIL]")

        # 详细结果
        for result in results:
            print(f"\n{'-'*80}")
            status_icon = "[PASS]" if result["overall_status"] == "PASS" else "[FAIL]"
            print(f"{status_icon} {Path(result['pdf_path']).name}: {result['overall_status']}")

            for criterion_name, criterion_result in result["acceptance_criteria"].items():
                if criterion_result["status"] == "SKIP":
                    icon = "[SKIP]"
                elif criterion_result["status"] == "PASS":
                    icon = "[PASS]"
                else:
                    icon = "[FAIL]"
                print(f"  {icon} {criterion_name}: {criterion_result['status']}")

        print(f"\n{'='*80}\n")


def main():
    """主函数"""
    print("C4 任务验收标准验证")
    print("=" * 80)

    # PDF 文件路径（使用相对路径）
    project_root = Path(__file__).parent.parent.parent
    pdf_files = [
        str(project_root / "docs/test/simple.pdf"),
        str(project_root / "docs/test/with_images.pdf"),
    ]

    # 创建验证器
    verifier = C4AcceptanceVerifier()

    # 验证每个 PDF 文件
    results = []
    for pdf_path in pdf_files:
        if Path(pdf_path).exists():
            result = verifier.verify(pdf_path)
            results.append(result)
        else:
            print(f"\n[WARNING] 文件不存在: {pdf_path}")

    # 打印总结
    verifier.print_summary(results)

    # 返回退出码
    all_passed = all(r["overall_status"] == "PASS" for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
