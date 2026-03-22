# C3 Loader 抽象基类与 PDF Loader - 实现总结

## 任务概述

**任务编号**: C3
**任务名称**: Loader 抽象基类与 PDF Loader
**目标**: 在 Libs 中定义 `BaseLoader`，并实现 `PdfLoader` 的最小行为
**完成日期**: 2026-03-22

### 验收标准
- [x] 基础要求：对 PDF 文件能产出 Document，metadata 至少含 `source_path`
- [x] 图片处理要求（遵循 C1 定义的契约）：
  - [x] 若 PDF 包含图片，应提取图片并保存到 `data/images/{doc_hash}/` 目录
  - [x] 在 `Document.text` 中，图片位置插入占位符：`[IMAGE: {image_id}]`
  - [x] 在 `metadata.images` 中记录图片信息（格式见 C1 规范）
  - [x] 若 PDF 无图片，`metadata.images` 可为空列表或省略该字段
- [x] 降级行为：图片提取失败不应阻塞文本解析，可在日志中记录警告

---

## 创建的文件

### 1. `src/libs/loader/base_loader.py`
**作用**: 定义文档加载器的抽象基类

**关键内容解读**:
- **BaseLoader 抽象类**: 定义了所有 Loader 必须实现的接口
  - `load(file_path: str) -> Document`: 抽象方法，子类必须实现文档加载逻辑
  - `supports_format(file_path: str) -> bool`: 检查是否支持该文件格式
  - `supported_extensions`: 抽象属性，返回支持的文件扩展名列表

- **辅助方法**:
  - `_extract_image_id()`: 生成唯一的图片 ID（格式：`{doc_hash}_{page:04d}_{index:04d}`）
  - `_create_image_metadata()`: 创建 ImageMetadata 对象，封装图片元数据

**Python 概念说明**:
- **@abstractmethod**: 装饰器，标记抽象方法，强制子类必须实现
- **ABC (Abstract Base Class)**: 抽象基类，不能直接实例化，只能继承
- **property**: 装饰器，将方法转换为属性访问，无需括号调用

### 2. `src/libs/loader/pdf_loader.py`
**作用**: PDF 文档加载器实现，使用 MarkItDown 转换 PDF 为 Markdown

**关键内容解读**:

- **PdfLoader 类**: 继承 BaseLoader，实现 PDF 加载逻辑
  - **SUPPORTED_EXTENSIONS**: 只支持 `.pdf` 文件
  - **IMAGES_BASE_DIR**: 图片存储目录（`data/images/{doc_id}/`）
  - **IMAGE_PLACEHOLDER_FORMAT**: 图片占位符格式（`[IMAGE: {image_id}]`）

- **PDF → Markdown 转换**:
  ```python
  def load(self, file_path: str) -> Document:
      # 1. 使用 MarkItDown 转换 PDF
      result = self.converter.convert(str(path))
      text_content = result.text_content

      # 2. 提取图片（如果有）
      images = self._extract_images_from_pdf(...)
      if images:
          text_content = self._insert_image_placeholders(text_content, images)

      # 3. 返回 Document 对象
      return Document(id=doc_id, text=text_content, metadata=metadata)
  ```

- **图片提取流程**:
  1. 创建图片存储目录：`data/images/{doc_id}/`
  2. 使用 PyPDF2 提取 PDF 中的图片（可选依赖）
  3. 保存图片到磁盘
  4. 创建 ImageMetadata 对象记录图片信息
  5. 在文本中插入占位符：`[IMAGE: {image_id}]`

- **降级机制**:
  - PyPDF2 未安装时，记录警告日志，但不阻塞文本解析
  - 图片提取失败时，返回空列表，继续处理文本内容

**关键方法**:
- `_extract_images_from_pdf()`: 图片提取主入口（优先 pdfplumber，降级 pypdf）
- `_extract_images_with_pdfplumber()`: 使用 pdfplumber 提取图片（推荐，精确坐标）
- `_extract_images_with_pypdf2()`: 使用 pypdf 提取图片（降级方案）
- `_extract_image_data_from_page()`: 从 PDF stream 提取图片数据
- `_calculate_text_offset_from_bbox()`: 根据 bbox 计算文本偏移
- `_insert_image_placeholders()`: 在文本中插入图片占位符

### 3. `src/libs/loader/__init__.py`
**作用**: 导出 Loader 相关类

**导出内容**:
- `BaseLoader`: 抽象基类
- `PdfLoader`: PDF 加载器实现
- `FileIntegrityChecker`, `SQLiteIntegrityChecker`, `IntegrityRecord`: 文件完整性检查相关

### 4. `tests/unit/test_loader_pdf_contract.py`
**作用**: PDF Loader 的单元测试

**测试覆盖** (16 个测试用例):

| 测试类 | 测试内容 | 数量 |
|--------|---------|------|
| **TestPdfLoaderContract** | 合规性测试 | 4 |
| - `test_supported_extensions` | 验证支持的扩展名列表 | 1 |
| - `test_supports_format` | 验证格式检测功能 | 1 |
| - `test_load_file_not_found` | 验证文件不存在错误处理 | 1 |
| - `test_load_unsupported_format` | 验证不支持的格式错误处理 | 1 |
| **TestPdfLoaderWithMockMarkItDown** | 加载功能测试 | 6 |
| - `test_load_simple_pdf_success` | 测试简单 PDF 加载成功 | 1 |
| - `test_load_pdf_with_images` | 测试带图片 PDF 加载 | 1 |
| - `test_load_pdf_image_extraction_graceful_degradation` | 测试图片提取降级行为 | 1 |
| - `test_load_pdf_markitdown_failure` | 测试 MarkItDown 失败处理 | 1 |
| - `test_metadata_completeness` | 测试元数据完整性 | 1 |
| - `test_document_id_generation` | 测试文档 ID 生成 | 1 |
| **TestPdfLoaderImageExtraction** | 图片提取测试 | 5 |
| - `test_extract_image_id_format` | 测试图片 ID 格式 | 1 |
| - `test_create_image_metadata` | 测试 ImageMetadata 创建 | 1 |
| - `test_insert_image_placeholders` | 测试占位符插入 | 1 |
| - `test_estimate_image_position_with_page_marker` | 测试带页码标记的位置估算 | 1 |
| - `test_estimate_image_position_without_page_marker` | 测试无页码标记的位置估算 | 1 |
| **TestPdfLoaderImageSaving** | 图片保存测试 | 1 |
| - `test_images_directory_creation` | 测试图片目录创建 | 1 |

---

## 关键设计解读

### Q1: 为什么使用抽象基类 (BaseLoader)？

**解释**: 抽象基类定义了所有 Loader 必须遵循的**契约（接口）**，确保不同文档格式的加载器有一致的行为。

**类比**: 就像 USB 接口标准 - 无论是鼠标、键盘、U盘，只要遵循 USB 标准，就能插入电脑使用。同样，无论是 PDF、Markdown、Word Loader，只要遵循 BaseLoader 接口，就能被 Ingestion Pipeline 统一调用。

**好处**:
- **可扩展性**: 未来添加新的文档格式（如 DocxLoader、MarkdownLoader）时，只需继承 BaseLoader 并实现 `load()` 方法
- **可替换性**: 可以在配置中动态切换不同的 Loader，无需修改调用代码
- **类型安全**: IDE 可以提供代码提示和类型检查

### Q2: 图片占位符 `[IMAGE: {image_id}]` 的作用是什么？

**解释**: 占位符有两个作用：
1. **标记图片位置**: 保留图片在原文档中的上下文位置
2. **支持后续处理**: 在 Splitter、Embedding、Rerank 阶段可以识别图片引用

**示例**:
```
原文档内容:
介绍 TensorFlow 的核心概念

[IMAGE: tensor_flow_graph_0001_0000]

上图展示了计算图的构建流程
```

**后续使用场景**:
- **ImageCaptioner**: 读取占位符中的 `image_id`，找到对应图片，生成描述
- **Rerank**: 识别查询是否与图片相关，提升召回精度
- **MCP Response**: 返回结果时，将占位符替换为实际的图片 Base64 数据

### Q3: 为什么图片提取失败要降级而不是抛出异常？

**解释**: 文档的**文本内容**通常比图片更重要，如果因为图片提取失败导致整个文档无法处理，会丢失大量有价值的信息。

**设计哲学**: **"宽进严出"**
- **宽进**: 尽可能多地接受输入，即使某些功能失败也不阻塞主流程
- **严出**: 对于输出结果，确保数据质量（如 Document 必须包含 source_path）

**降级行为**:
```python
try:
    import pypdf
    images = self._extract_images_with_pypdf2(...)
except ImportError:
    self.logger.warning("PyPDF2 not installed. Image extraction skipped.")
    images = []  # 返回空列表，不阻塞文本解析
```

**好处**:
- 用户可以快速使用 PDF 文档，即使没有安装 PyPDF2
- 避免因为可选依赖缺失导致系统崩溃
- 提供明确的日志提示，告知用户图片提取失败的原因

---

## 测试验证

**16 个测试全部通过**，验证了：

✅ **接口合规性**
- PdfLoader 正确报告支持的文件格式（`.pdf`）
- 格式检测功能正常工作（大小写不敏感）

✅ **错误处理**
- 文件不存在时抛出 `FileNotFoundError`
- 不支持的格式抛出 `ValueError`
- MarkItDown 转换失败抛出 `RuntimeError`

✅ **文档加载**
- 简单 PDF 能成功加载为 Document 对象
- Document ID 基于文件名生成（无扩展名）
- 元数据包含 `source_path`、`doc_type`、`file_name`、`file_size`

✅ **图片提取**
- 图片 ID 格式正确：`{doc_id}_{page:04d}_{index:04d}`
- 图片占位符正确插入到文本中
- 图片元数据包含所有必需字段（id, path, page, text_offset, text_length）

✅ **降级机制**
- 图片提取失败不阻塞文本解析
- 图片目录不存在时自动创建

✅ **边界条件**
- 无页码标记时的位置估算
- 多个占位符的正确插入顺序
- 图片文件的正确保存

---

## 项目进度

**阶段 C：Ingestion Pipeline MVP**
- C1: ✅ 已完成
- C2: ✅ 已完成
- **C3**: ✅ **已完成**
- C4: ⏳ 待开始

---

## 使用示例

```python
from src.libs.loader import PdfLoader

# 创建 PDF 加载器
loader = PdfLoader()

# 加载 PDF 文档
document = loader.load("/path/to/document.pdf")

# 查看文档内容
print(f"Document ID: {document.id}")
print(f"Text length: {len(document.text)}")
print(f"Source: {document.metadata['source_path']}")

# 检查图片
if "images" in document.metadata:
    print(f"Extracted {len(document.metadata['images'])} images")
    for image_meta in document.metadata["images"]:
        print(f"  - {image_meta.id}: {image_meta.path}")
else:
    print("No images found")
```

**输出示例**:
```
Document ID: research_paper
Text length: 15234
Source: /path/to/document.pdf
Extracted 3 images
  - research_paper_0001_0000: data/images/research_paper/research_paper_0001_0000.png
  - research_paper_0002_0000: data/images/research_paper/research_paper_0002_0000.png
  - research_paper_0005_0000: data/images/research_paper/research_paper_0005_0000.png
```

---

## 技术亮点

- **MarkItDown 集成**: 使用微软的 MarkItDown 库将 PDF 转换为 Markdown
- **pdfplumber 图片提取**（推荐）: 使用 pdfplumber 进行精确的图片提取，提供完整的 bbox 坐标信息
  - 优先级：pdfplumber → pypdf（降级）
  - 支持获取完整的 PDF 坐标（x0, y0, x1, y1, width, height）
  - 自动识别图片格式（JPEG、PNG 等）
- **pypdf 6.x+ API 兼容**: 修复了新版 pypdf 的 API 兼容性问题（使用 `.image` 属性）
- **优雅降级**: 图片提取失败不阻塞文本解析，支持多级降级策略
- **占位符机制**: `[IMAGE: {image_id}]` 格式支持多模态 RAG
- **抽象接口**: BaseLoader 易于扩展其他文档格式
- **完整测试**: 16 个测试用例覆盖主要场景
- **真实 PDF 验证**: 成功解析真实 PDF 文件，提取 32 张图片（simple.pdf: 1张，with_images.pdf: 31张）

---

## 图片提取实现细节

### 优先级策略

```
pdfplumber（首选）
    ↓ 失败
pypdf（降级）
    ↓ 失败
跳过图片提取（优雅降级）
```

### pdfplumber vs pypdf 对比

| 特性 | pdfplumber | pypdf 6.x+ |
|------|-------------|-------------|
| **图片数据获取** | `stream.get_data()` | `image_object.image` |
| **坐标信息** | ✅ 完整 bbox + 宽高 | ✅ bbox |
| **PDF 兼容性** | 更好（处理复杂 PDF） | 依赖 PDF 版本 |
| **性能** | 稍慢 | 更快 |
| **API 稳定性** | 稳定 | 经常变化 |

### 实际验证结果

**测试文件**：`docs/test/simple.pdf` 和 `docs/test/with_images.pdf`

| 文件 | 文本长度 | 提取图片 | 图片文件 | 占位符 |
|------|---------|---------|---------|--------|
| **simple.pdf** | 1,721 字符 | 1 张 | ✅ 存在 | ✅ 1 个 |
| **with_images.pdf** | 4,866 字符 | 31 张 | ✅ 全部存在 | ✅ 31 个 |

**提取的图片信息示例**：
```
[1] ID: with_images_0002_0000
    页码: 2
    路径: D:\projects\MODULAR-RAG-MCP-SERVER\data\images\with_images\with_images_0002_0000.jpg
    文本位置: 2000
    PDF坐标: {'x0': 17.13, 'y0': 95.57, 'x1': 942.87, 'y1': 444.43,
              'width': 925.74, 'height': 348.86}
```

---

---

## 设计决策：为什么选择 Markdown 作为 PDF 转换目标格式？

### 问题背景

在实现 PDF Loader 时，一个核心的架构决策是：**将 PDF 转换成什么格式**？这直接影响到后续的 Splitter、Embedding、Rerank 等所有环节的质量。

### 三种提取方式对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PDF 内容提取的三种方式                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  方式1: 纯文本提取 (PyPDF2.extract_text())                          │
│  ─────────────────────────────────────────                         │
│  原始 PDF:                                                           │
│  ┌─────────────┐                                                    │
│  │ 3.1 引言    │  ← 标题（大字体、加粗）                             │
│  ├─────────────┤                                                    │
│  │ 本文讨论...  │                                                    │
│  │ • 要点一     │  ← 列表                                            │
│  │ • 要点二     │                                                    │
│  └─────────────┘                                                    │
│                                                                     │
│  提取结果:                                                           │
│  "3.1 引言 本文讨论... 要点一 要点二"                               │
│                                                                     │
│  ❌ 问题: 丢失了标题、列表、表格等结构信息                           │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  方式2: 保留格式 (pdfplumber + 布局分析)                            │
│  ─────────────────────────────────────────                         │
│  提取结果:                                                           │
│  [{                                                                │
│    "type": "heading",                                              │
│    "text": "3.1 引言",                                             │
│    "font": "Arial-Bold", "size": 18,                               │
│    "position": (100, 200)                                          │
│  }, {                                                              │
│    "type": "list_item",                                            │
│    "text": "• 要点一",                                             │
│    "indent": 20                                                    │
│  }]                                                                │
│                                                                     │
│  ⚠️ 问题: 结构复杂，难以直接用于 LLM 和向量检索                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  方案3: Markdown 转换 (MarkItDown) ✅ 当前方案                      │
│  ─────────────────────────────────────────                         │
│  提取结果:                                                           │
│  "## 3.1 引言                                                       │
│                                                                     │
│   本文讨论...                                                       │
│   - 要点一                                                          │
│   - 要点二"                                                         │
│                                                                     │
│  ✅ 优点: 保留了结构，又是纯文本，便于后续处理                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Markdown 在 RAG 系统中的关键优势

#### 1. 为 Splitter 提供语义切分点

**Markdown 格式**（保留结构）:
```markdown
# 深度学习基础

## 1.1 神经网络简介

神经网络是...

### 1.1.1 感知机

感知机是...

## 1.2 卷积神经网络

CNN 专门用于图像处理...
```

**纯文本格式**（丢失结构）:
```
深度学习基础 1.1 神经网络简介 神经网络是... 1.1.1 感知机 感知机是...
1.2 卷积神经网络 CNN 专门用于图像处理...
```

**Splitter 的优势**:
- 按 `##` 标题切分，每个 Chunk 是一个完整的小节
- 每个 Chunk 保留标题信息，便于检索时理解上下文
- 避免在句子中间切分，保持语义完整性

#### 2. 为 Embedding 提供结构感知

**对比示例**:

| 格式 | 内容 | Embedding 理解 |
|------|------|----------------|
| **Markdown** | `## 性能对比\n\nTensorFlow 在分布式训练方面表现更好...` | 能识别 `##` 是标题，向量包含"结构 + 语义" |
| **纯文本** | `性能对比 TensorFlow 在分布式训练方面表现更好...` | 只是一串文字，丢失了"这是一个小节"的信息 |

**检索质量提升**:
```python
query = "TensorFlow 和 PyTorch 在性能上有什么区别？"

# Markdown 版本更容易被召回，因为:
# - "## 性能对比" 与 query 中的 "性能" 语义匹配
# - 标题结构提供了额外的上下文信息
```

#### 3. 为 LLM 生成提供格式化输入

**检索到的上下文**（Markdown 格式）:
```markdown
# ResNet 架构

## 残差连接

ResNet 的核心创新是残差连接：

```
H(x) = F(x) + x
```

其中 F(x) 是要学习的残差映射。

## 网络结构

ResNet-50 包含...
```

**LLM 生成的回答**:
```
根据文档，ResNet 的核心创新是**残差连接**，公式为：

```
H(x) = F(x) + x
```

这个设计的优势是...
```

**Markdown 的优势**:
- LLM 训练数据中包含大量 Markdown，理解力更好
- LLM 可以在回答中保留代码块、列表等格式
- 生成的答案更易读、更专业

#### 4. 表格和代码块的处理

**Markdown 保留的结构**:
```markdown
# Markdown 保留的表格结构:

| 模型 | 参数量 | Top-1 准确率 |
|------|--------|--------------|
| ResNet-50 | 25.6M | 76.15% |
| ViT-B | 86M | 77.9% |

# 代码块:

```python
import torch
model = torchvision.models.resnet50(pretrained=True)
```
```

**纯文本丢失信息**:
```
模型 参数量 Top-1 准确率 ResNet-50 25.6M 76.15% ViT-B 86M 77.9%
import torch model = torchvision.models.resnet50(pretrained=True)
```

**问题**：
- 表格变成了一团乱麻
- 代码和正文混在一起，无法区分

### PDF 内容在 RAG 系统中的完整流转

```
PDF 原始文件
├─ 文本（复杂布局）
├─ 图片（嵌入数据）
└─ 表格（多列布局）
        ↓
MarkItDown 转换引擎
        ↓
Markdown 文本（结构化纯文本）
├─ ## 表1: 模型性能对比
├─ | 模型 | 准确率 | ... |  ← 表格结构保留
└─ ![IMAGE: doc_0001_0000]  ← 图片占位符
        ↓
Splitter（按标题切分）
├─ Chunk 1: "## 表1: 模型性能对比\n\n| 模型 | 准确率..."
└─ Chunk 2: "![IMAGE: doc_0001_0000]\n\n上图展示了..."
        ↓
Embedding（向量化）
        ↓
向量数据库（每个 Chunk 对应一个向量）
        ↓
用户查询："ResNet 和 ViT 谁更准确？"
        ↓
召回 Chunk 1（包含表格，标题中有"性能对比"）
        ↓
LLM 生成回答
        ↓
"根据表格数据，ViT 的准确率（78%）高于 ResNet（76%）。"
```

### 对比其他方案的优劣

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **纯文本** | 简单、快速 | ❌ 丢失结构<br>❌ 表格变乱码<br>❌ 代码无法识别 | 简单文档，无格式 |
| **Markdown** | ✅ 保留结构<br>✅ 适合 LLM<br>✅ 便于切分 | ⚠️ 依赖转换质量 | **RAG 系统（推荐）** |
| **HTML/XML** | ✅ 保留完整格式 | ❌ 太复杂<br>❌ LLM 难处理<br>❌ 向量化困难 | 网页抓取 |
| **LaTeX** | ✅ 学术文档标准 | ❌ 解析困难<br>❌ 非通用格式 | 学术论文专用 |
| **JSON结构化** | ✅ 结构最完整 | ❌ 实现复杂<br>❌ 难以通用 | 特定格式文档 |

### 为什么不用其他格式？

#### Q: 为什么不用 HTML？

HTML 看起来更强大，但有问题：

```html
<h2>3.1 Feedforward Networks</h2>
<p>A feedforward network is defined as...</p>
<ul>
  <li>Input layer</li>
  <li>Hidden layer</li>
</ul>
```

**问题**：
1. **标签噪音**：`<h2>`, `<p>`, `<ul>` 等标签会干扰 Embedding
2. **复杂性**：需要处理 CSS、嵌套标签
3. **LLM 理解**：LLM 对 Markdown 的理解比 HTML 更好

#### Q: 为什么不用 JSON 结构化？

```json
{
  "type": "section",
  "heading": "3.1 Feedforward Networks",
  "content": "A feedforward network is defined as...",
  "items": ["Input layer", "Hidden layer"]
}
```

**问题**：
1. **不通用**：每种 PDF 格式需要不同的解析规则
2. **复杂度高**：难以处理嵌套结构、表格、图片
3. **不便于 LLM**：LLM 需要额外的提示才能理解 JSON 结构

### Markdown 的独特优势总结

| 特性 | 说明 |
|------|------|
| **结构保留** | 标题、列表、表格、代码块都被保留 |
| **LLM 友好** | 大多数 LLM 训练数据包含 Markdown |
| **便于切分** | Splitter 可以基于标题进行语义切分 |
| **向量化友好** | 纯文本，适合 Embedding 模型 |
| **可读性强** | 人类可以直接阅读和调试 |
| **通用性** | Markdown 是事实上的文档标准格式 |
| **工具生态** | 大量 Markdown 处理工具可用 |

### 一句话总结

> Markdown 是**结构化**和**简单性**的最佳平衡点——既保留了文档的语义结构，又保持了纯文本的易处理性，是 RAG 系统的理想中间格式。

---

## 依赖说明

**必需依赖**:
- `markitdown[pdf]`: PDF → Markdown 转换（包含 pdfminer-six, pdfplumber）

**可选依赖**（图片提取）:
- `pdfplumber`: 推荐的图片提取库（已包含在 `markitdown[pdf]` 中）
  - 提供精确的 bbox 坐标
  - 更好的 PDF 兼容性
- `pypdf`: 降级方案，新版 API 支持（6.9.1+）

**安装命令**:
```bash
# 推荐方式：安装完整的 PDF 支持
pip install "markitdown[pdf]"

# 这会自动安装：
# - markitdown
# - pdfminer-six
# - pdfplumber (图片提取)
# - 其他依赖
```

**依赖关系图**:
```
PdfLoader
├── markitdown[pdf] (必需)
│   ├── markitdown (PDF → Markdown)
│   ├── pdfminer-six (PDF 解析)
│   └── pdfplumber (图片提取，首选)
└── pypdf (图片提取，降级方案)
```

**关于 pypdf 版本**:
- 当前代码已修复 pypdf 6.x+ API 兼容性
- 使用 `.image` 属性访问 PIL Image 对象
- 自动转换为字节数据保存

---

---

## 代码重构与优化记录

### 重构 1：清理未使用参数（方案1）

**问题**：`_extract_images_from_pdf` 方法接收了 `text_content` 和 `markitdown_result` 参数但未使用。

**改进**：
- 移除了未使用的参数
- 添加了清晰的 MVP 实现说明
- 为未来改进留下指引（Future Improvements）

**结果**：
- 接口更简洁，意图明确
- 代码可读性提升

### 重构 2：切换到 pdfplumber（方案3）

**问题**：
- pypdf API 兼容性问题导致图片提取失败
- 错误：`'ImageFile' object has no attribute 'get_data'`

**改进**：
1. **优先级调整**：pdfplumber → pypdf（降级）
2. **修复 pypdf 6.x API**：使用 `.image` 属性而不是 `.get_data()`
3. **修复路径问题**：使用绝对路径避免 `.relative_to()` 错误
4. **改进错误处理**：详细的调试日志

**结果**：
- 图片提取成功率：0% → 100%
- 从 32 张图片全部失败 → 32 张全部成功
- 提供完整的 PDF 坐标信息（bbox + 宽高）
- 支持多级降级：pdfplumber → pypdf → 跳过

**代码变更摘要**：
- 新增：`_extract_images_with_pdfplumber()` 方法（约 90 行）
- 新增：`_extract_image_data_from_page()` 方法（约 120 行）
- 新增：`_calculate_text_offset_from_bbox()` 方法（约 25 行）
- 修改：`_extract_images_from_pdf()` 方法（优先级逻辑）
- 修改：`_extract_images_with_pypdf2()` 方法（修复 API）
- 修复：路径计算（使用绝对路径）

**测试验证**：
- ✅ 16/16 单元测试通过
- ✅ simple.pdf：1 张图片成功提取
- ✅ with_images.pdf：31 张图片成功提取
- ✅ 所有图片文件验证存在
- ✅ 图片占位符正确插入

---

## 后续扩展

### 已实现 ✅

- ✅ **pdfplumber 图片提取**：精确的图片定位和坐标信息
- ✅ **多级降级策略**：pdfplumber → pypdf → 跳过
- ✅ **完整坐标支持**：bbox + 宽高信息
- ✅ **pypdf 6.x+ 兼容**：修复新版 API

### 未来优化方向

基于 BaseLoader 抽象，可以轻松添加新的文档格式支持：

- **DocxLoader**: 加载 Word 文档（`.docx`）
- **MarkdownLoader**: 加载 Markdown 文件（`.md`）
- **TextLoader**: 加载纯文本文件（`.txt`）
- **HtmlLoader**: 加载 HTML 网页（`.html`）

### 图片提取增强（可选）

当前实现已经很好，但未来可以考虑：

1. **精确的图片位置定位**：
   - 分析 MarkItDown 输出的 Markdown 文本
   - 根据文本内容匹配图片位置
   - 提升占位符插入的准确性

2. **使用 PyMuPDF (fitz)**：
   - 性能更优（C 语言实现）
   - 功能更全（支持几乎所有 PDF 特性）
   - 适合批量处理场景

3. **图片预处理**：
   - 自动压缩超大图片
   - 格式统一（全部转为 JPEG/PNG）
   - 去重检测

每个新 Loader 只需：
1. 继承 `BaseLoader`
2. 实现 `load()` 方法
3. 定义 `supported_extensions` 属性
