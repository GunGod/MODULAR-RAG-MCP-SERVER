# C4: Splitter 集成（DocumentChunker）实现总结

## 任务概述

| 属性 | 值 |
|------|-----|
| **任务编号** | C4 |
| **任务名称** | Splitter 集成（调用 Libs） |
| **完成日期** | 2026-03-22 |
| **实现目标** | 实现 DocumentChunker 适配器层，完成 Document → List[Chunk] 的业务对象转换 |

### 核心目标

本任务实现了 `DocumentChunker` 类，作为 `libs.splitter`（纯文本切分）与 Ingestion Pipeline（业务对象处理）之间的**适配器层**。

### 验收标准达成情况

| 验收标准 | 状态 | 说明 |
|---------|------|-----|
| 配置驱动 | ✅ | 通过修改 settings.yaml 中的 splitter 配置（如 chunk_size），产出的 chunk 数量和长度发生相应变化 |
| ID 唯一性 | ✅ | 每个 Chunk 的 ID 在整个文档中唯一 |
| ID 确定性 | ✅ | 同一 Document 对象重复切分产生相同的 Chunk ID 序列 |
| 元数据完整性 | ✅ | Chunk.metadata 包含所有 Document.metadata 字段 + chunk_index 字段 |
| 图片分发正确性 | ✅ | 含 `[IMAGE: id]` 占位符的 chunk 其 `metadata["images"]` 仅包含该 chunk 引用的图片子集 |
| 溯源链接 | ✅ | 所有 Chunk.source_ref 正确指向父 Document.id |
| 类型契约 | ✅ | 输出的 Chunk 对象符合 `core/types.py` 中的 Chunk 定义 |

---

## 创建的文件

### 1. `src/ingestion/chunking/document_chunker.py`

**作用**：DocumentChunker 适配器层的核心实现，负责将 Document 对象转换为 List[Chunk] 对象。

**关键类和方法**：

#### `DocumentChunker` 类
适配器类，连接 libs.splitter 和 Ingestion Pipeline。

**初始化方法**：
```python
def __init__(self, splitter: BaseSplitter)
```
- 接收一个 `BaseSplitter` 实例（通过 SplitterFactory 创建）
- 验证 splitter 类型，确保是 BaseSplitter 的子类

**核心方法**：

1. **`chunk_document(document: Document) -> List[Chunk]`**
   - 主入口方法，执行完整的文档切分流程
   - 步骤：
     1. 调用 `splitter.split_text()` 获取文本 chunks
     2. 为每个文本 chunk 生成唯一 Chunk ID
     3. 计算每个 chunk 在原文档中的 offset 位置
     4. 继承 Document.metadata 并添加 chunk 特定字段
     5. 建立 source_ref 溯源链接

2. **`_generate_chunk_id(doc_id, chunk_index, chunk_text) -> str`**
   - 生成格式：`{doc_id}_{index:04d}_{hash_8chars}`
   - 使用 `Chunk.generate_id()` 静态方法（已实现在 core/types.py）
   - 保证唯一性和确定性

3. **`_inherit_metadata(document, chunk_index, chunk_text) -> dict`**
   - 复制 Document 的所有 metadata
   - 添加 `chunk_index` 和 `chunk_size` 字段
   - 扫描 `[IMAGE: {id}]` 占位符
   - 过滤 `Document.metadata["images"]` 为仅包含当前 chunk 引用的图片

4. **`_extract_image_refs(chunk_text) -> List[dict]`**
   - 使用正则表达式 `r"\[IMAGE:\s*([^\]]+)\]"` 扫描图片占位符
   - 返回结构化的 image_refs 列表，每个元素包含 `{'id': image_id}`

5. **`_find_chunk_offsets(document_text, chunk_text, chunk_index) -> tuple[int, int]`**
   - 计算每个 chunk 在原文档中的起止位置
   - 使用启发式算法：基于 chunk_index 估算位置，然后在文档中查找匹配

**Python 概念解释**：

- **适配器模式 (Adapter Pattern)**：DocumentChunker 将 libs.splitter 的 `str → List[str]` 接口转换为 `Document → List[Chunk]` 接口
- **正则表达式 (re.findall)**：用于扫描文本中的图片占位符
- **数据类 (dataclass)**：Chunk 是一个 dataclass，自动生成 `__init__`、`__repr__` 等方法

### 2. `src/ingestion/chunking/__init__.py`

**作用**：Chunking 模块的入口文件，导出 `DocumentChunker` 类。

**内容**：
```python
from src.ingestion.chunking.document_chunker import DocumentChunker

__all__ = ["DocumentChunker"]
```

### 3. `tests/unit/test_document_chunker.py`

**作用**：DocumentChunker 的单元测试套件，覆盖所有核心功能。

**测试结构**（20 个测试用例）：

| 测试类 | 测试数量 | 覆盖范围 |
|-------|---------|---------|
| `TestDocumentChunkerInitialization` | 3 | 初始化和类型验证 |
| `TestDocumentChunkerChunking` | 4 | 基础切分功能 |
| `TestChunkIdGeneration` | 2 | Chunk ID 格式和确定性 |
| `TestMetadataInheritance` | 4 | 元数据继承和字段添加 |
| `TestImageReferenceDistribution` | 3 | 图片引用分发逻辑 |
| `TestSourceReference` | 2 | source_ref 溯源链接 |
| `TestErrorHandling` | 2 | 错误处理和边界情况 |

**测试通过情况**：✅ 20/20 通过

---

## 关键设计解读

### Q1: 为什么需要 DocumentChunker 适配器层？

**解释**：

`libs.splitter` 只是一个**纯文本切分工具**，它只负责 `str → List[str]` 的转换，不涉及任何业务逻辑。而 Ingestion Pipeline 需要的是完整的**业务对象转换** `Document → List[Chunk]`。

**类比**：

- `libs.splitter` 就像一个**切菜刀**，只负责把一大块文本切成小块
- `DocumentChunker` 就像一个**厨师**，不仅切菜，还给每道菜加上标签（元数据）、建立溯源关系、处理配菜（图片）

**分工**：
- `libs.splitter`：关注"如何切"（算法：递归、语义、定长）
- `DocumentChunker`：关注"切完之后怎么办"（业务：ID 生成、元数据继承、图片分发）

**好处**：
- **职责分离**：libs.splitter 可以独立演化（添加新的切分算法），不影响业务逻辑
- **可测试性**：可以用 FakeSplitter 测试 DocumentChunker，无需真实切分算法
- **灵活性**：可以轻松替换底层切分算法（Recursive → Semantic），而不改变业务逻辑

### Q2: 图片引用是如何分发的？

**解释**：

当文档被切分后，每个 chunk 需要知道自己包含哪些图片。DocumentChunker 通过以下步骤实现：

1. **扫描占位符**：使用正则表达式在每个 chunk 的文本中查找 `[IMAGE: {image_id}]` 模式
2. **提取 image_id**：从占位符中提取图片 ID（如 "img_001"）
3. **过滤 Document.images**：从 Document 的完整图片列表中，筛选出当前 chunk 引用的图片
4. **写入 metadata**：
   - `chunk.metadata["image_refs"]`：包含的图片 ID 列表
   - `chunk.metadata["images"]`：完整的 ImageMetadata 对象列表（仅包含当前 chunk 引用的）

**示例**：

```python
# Document 包含 3 张图片
document = Document(
    id="doc_001",
    text="Section 1 [IMAGE: img_001] Section 2 [IMAGE: img_002] Section 3 [IMAGE: img_003]",
    metadata={
        "images": [
            ImageMetadata(id="img_001", path="/img1.png", ...),
            ImageMetadata(id="img_002", path="/img2.png", ...),
            ImageMetadata(id="img_003", path="/img3.png", ...),
        ]
    }
)

# 切分后（假设切分为 2 个 chunks）
chunk1 = chunks[0]
# chunk1.text = "Section 1 [IMAGE: img_001] Section 2"
# chunk1.metadata["image_refs"] = [{"id": "img_001"}]
# chunk1.metadata["images"] = [ImageMetadata(id="img_001", ...)]  # 只有 img_001

chunk2 = chunks[1]
# chunk2.text = "[IMAGE: img_002] Section 3 [IMAGE: img_003]"
# chunk2.metadata["image_refs"] = [{"id": "img_002"}, {"id": "img_003"}]
# chunk2.metadata["images"] = [ImageMetadata(id="img_002", ...), ImageMetadata(id="img_003", ...)]
```

**好处**：
- **精准定位**：每个 chunk 只知道自己包含的图片，避免处理无关图片
- **性能优化**：下游的 ImageCaptioner 只需要处理当前 chunk 的图片，而非整个文档的图片
- **可追溯**：通过 `image_refs` 可以快速找到 chunk 引用的所有图片

### Q3: Chunk ID 为什么设计为 `{doc_id}_{index}_{hash}` 的格式？

**解释**：

Chunk ID 的设计遵循三个原则：**唯一性**、**确定性**、**可读性**。

**格式解析**：
- `{doc_id}`：父文档的 ID，表明 chunk 属于哪个文档
- `{index:04d}`：chunk 在文档中的序号（0-based，4 位补零），表明 chunk 的顺序
- `{hash_8chars}`：chunk 内容的 SHA256 哈希（前 8 位），用于内容去重和校验

**唯一性保证**：
- 同一文档的不同 chunk：index 不同
- 不同文档的 chunk：doc_id 不同
- 内容不同的 chunk：hash 不同

**确定性保证**：
- 同一文档、同一位置、相同内容的 chunk，无论切分多少次，ID 都相同
- 这对于**增量更新**和**幂等写入**至关重要

**可读性**：
- 可以从 ID 中直接读出：这是哪个文档的、第几个 chunk
- 例如：`abc123_0005_a1b2c3d4` → 文档 abc123 的第 5 个 chunk

**示例**：
```python
# 第一次切分
chunks1 = chunker.chunk_document(document)  # ["abc123_0000_a1b2c3", "abc123_0001_b2c3d4"]

# 第二次切分（相同文档）
chunks2 = chunker.chunk_document(document)  # ["abc123_0000_a1b2c3", "abc123_0001_b2c3d4"]

# IDs 完全相同 → 幂等性
assert [c.id for c in chunks1] == [c.id for c in chunks2]
```

---

## 测试验证

### 测试方法
```bash
pytest tests/unit/test_document_chunker.py -v
```

### 测试结果
```
======================= 20 passed, 34 warnings in 0.07s =======================
```

### 测试覆盖场景

#### 1. 初始化测试 (3/3)
- ✅ 有效 splitter 初始化
- ✅ 无效 splitter 类型抛出 TypeError
- ✅ None splitter 抛出 TypeError

#### 2. 基础切分测试 (4/4)
- ✅ 简单文档切分
- ✅ 带图片的文档切分
- ✅ 空文档抛出 ValueError
- ✅ chunk 数量正确性验证

#### 3. Chunk ID 测试 (2/2)
- ✅ ID 格式符合 `{doc_id}_{index:04d}_{hash}` 规范
- ✅ ID 确定性：重复切分产生相同 ID

#### 4. 元数据继承测试 (4/4)
- ✅ 继承 Document 的所有 metadata
- ✅ 添加 `chunk_index` 字段
- ✅ 添加 `chunk_size` 字段
- ✅ `has_images` 字段正确性

#### 5. 图片引用分发测试 (3/3)
- ✅ 提取 `[IMAGE: {id}]` 占位符
- ✅ 过滤 `images` 字段为仅包含引用的图片
- ✅ 无图片的 chunk 不包含 `images` 字段

#### 6. 溯源链接测试 (2/2)
- ✅ `source_ref` 指向父 Document.id
- ✅ `source_ref` 为可选字段

#### 7. 错误处理测试 (2/2)
- ✅ splitter 失败时正确传播错误
- ✅ splitter 返回空列表时抛出 RuntimeError

---

## 任务完成度

### 满足任务要求清单

| 要求 | 状态 | 实现方式 |
|------|------|---------|
| Chunk ID 生成 | ✅ | 使用 `Chunk.generate_id()` 实现 `{doc_id}_{index:04d}_{hash}` 格式 |
| 元数据继承 | ✅ | `_inherit_metadata()` 复制 Document.metadata |
| 添加 chunk_index | ✅ | `_inherit_metadata()` 添加 `metadata["chunk_index"]` |
| 建立 source_ref | ✅ | `chunk_document()` 设置 `chunk.source_ref = document.id` |
| 图片引用分发 | ✅ | `_extract_image_refs()` 扫描占位符，过滤 Document.images |
| 类型转换 | ✅ | 将 `List[str]` 转换为 `List[Chunk]` |

### 验收标准达成

- ✅ **配置驱动**：通过 settings.yaml 配置 splitter 参数（chunk_size、chunk_overlap）
- ✅ **ID 唯一性**：每个 Chunk ID 在整个文档中唯一
- ✅ **ID 确定性**：重复切分产生相同的 ID 序列
- ✅ **元数据完整性**：所有 Document.metadata 字段 + chunk_index 都被继承
- ✅ **图片分发正确性**：每个 chunk 只包含自己引用的图片
- ✅ **溯源链接**：所有 chunk.source_ref 指向父 document.id
- ✅ **类型契约**：输出符合 core/types.py 的 Chunk 定义

---

## 项目进度

### 阶段 C：Ingestion Pipeline MVP 进度

| 任务 | 状态 | 完成日期 |
|------|------|---------|
| C1: FileIntegrity 抽象基类 | ✅ | 2026-03-13 |
| C2: SQLiteIntegrityChecker 实现 | ✅ | 2026-03-13 |
| C3: Loader 抽象基类 + PDF Loader | ✅ | 2026-03-22 |
| **C4: Splitter 集成（DocumentChunker）** | ✅ | **2026-03-22** |
| C5: Transform 抽象基类 + ChunkRefiner | ⬜ | - |
| C6: Embedding 抽象基类 + DenseEncoder | ⬜ | - |
| C7: ImageCaptioner（图片描述生成） | ⬜ | - |
| C8: VectorUpsert（向量库写入） | ⬜ | - |
| C9: Pipeline 编排（整合 C1-C8） | ⬜ | - |

---

## 使用示例

### 基础用法

```python
from src.libs.splitter import SplitterFactory
from src.core.settings import load_settings
from src.ingestion.chunking import DocumentChunker
from src.libs.loader import PdfLoader

# 1. 加载配置并创建 splitter
settings = load_settings()
splitter = SplitterFactory.create(settings)

# 2. 创建 DocumentChunker
chunker = DocumentChunker(splitter)

# 3. 加载文档
loader = PdfLoader()
document = loader.load("/path/to/document.pdf")

# 4. 切分文档
chunks = chunker.chunk_document(document)

# 5. 验证结果
print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.id}")
    print(f"  Length: {len(chunk.text)}")
    print(f"  Images: {chunk.metadata.get('image_refs', [])}")
```

### 配置 splitter（settings.yaml）

```yaml
# 添加 splitter 配置
splitter:
  provider: recursive  # or fake, semantic, fixed
  chunk_size: 1000
  chunk_overlap: 200
  separators:
    - "\n\n"
    - "\n"
    - " "
    - ""
```

---

## 技术亮点

### 1. 适配器模式应用

DocumentChunker 是适配器模式的典型应用，将底层工具的简单接口转换为上层业务所需的复杂接口：

- **底层**：`libs.splitter.split_text(text: str) -> List[str]`
- **上层**：`DocumentChunker.chunk_document(doc: Document) -> List[Chunk]`

### 2. 启发式位置计算

`_find_chunk_offsets()` 方法使用启发式算法计算每个 chunk 在原文档中的位置：

- 对于第一个 chunk，从位置 0 开始
- 对于后续 chunk，基于 `chunk_index * chunk_size` 估算位置
- 使用 `str.find()` 查找 chunk text 在文档中的实际位置

### 3. 正则表达式图片占位符匹配

使用正则表达式 `r"\[IMAGE:\s*([^\]]+)\]"` 精确匹配图片占位符：

- `\[IMAGE:\s*`：匹配 `[IMAGE:` 后跟可选空白
- `([^\]]+)`：捕获组，匹配非 `]` 的字符（图片 ID）
- `\]`：匹配闭合的 `]`

### 4. 测试驱动的开发

本任务完全遵循 TDD 方法，先编写测试用例，再实现功能：

- 20 个测试用例覆盖所有关键场景
- 使用 FakeSplitter 隔离测试，无需外部依赖
- 测试通过后，再进行集成测试

---

## 依赖关系

### 前置依赖

- ✅ `C1-C3`：FileIntegrity、Loader 等前置任务已完成
- ✅ `B7.5`：libs.splitter 模块已实现（FakeSplitter、RecursiveSplitter）
- ✅ `core/types.py`：Chunk、Document 等核心数据类型已定义

### 后续任务

- ⬜ `C5`：Transform 抽象基类 + ChunkRefiner（依赖 DocumentChunker 产生的 Chunk）
- ⬜ `C9`：Pipeline 编排（将整合 DocumentChunker）

---

## 总结

C4 任务成功实现了 DocumentChunker 适配器层，完成了以下关键功能：

1. **接口适配**：将 `libs.splitter` 的 `str → List[str]` 转换为 `Document → List[Chunk]`
2. **ID 生成**：实现唯一且确定性的 Chunk ID 格式
3. **元数据管理**：继承 Document metadata，添加 chunk 特定字段
4. **图片分发**：智能分发图片引用到相关 chunk
5. **溯源链接**：建立 chunk 到 document 的溯源关系

测试覆盖完整（20/20 通过），代码质量高，为后续的 Transform、Embedding 等模块奠定了坚实基础。
