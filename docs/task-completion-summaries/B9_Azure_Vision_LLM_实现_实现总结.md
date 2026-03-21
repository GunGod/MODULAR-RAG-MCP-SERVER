# B9 Azure Vision LLM 实现 - 实现总结

## 任务概述

**任务编号**: B9
**任务名称**: Azure Vision LLM 实现
**目标**: 实现 `AzureVisionLLM`，支持通过 Azure OpenAI 调用 GPT-4o 进行图像理解
**完成日期**: 2026-03-21

### 验收标准
- [x] 实现 `chat()` 方法调用 Azure OpenAI Vision API
- [x] 支持 URL/base64/PATH 三种图片输入方式
- [x] 图片过大时自动压缩至 max_image_size 配置的尺寸（默认 2048px）
- [x] API 调用失败时抛出清晰错误，包含 Azure 特有错误码
- [x] Mock 测试覆盖：正常调用、图片压缩、超时、认证失败等场景

---

## 创建的文件

### 1. `src/libs/llm/azure_vision_llm.py`
**作用**: Azure OpenAI Vision LLM 的具体实现

**关键内容解读**:

#### `AzureVisionLLM` 类
- **继承**: `BaseVisionLLM` 抽象基类
- **API 端点格式**: `https://{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={version}`
- **HTTP 客户端**: 使用 `httpx.Client`，超时设置为 120 秒（图片处理较慢）

**核心方法**:

1. **`chat(messages)`** - 发送多模态对话请求
   - 将 `VisionMessage` 转换为 Azure API 格式
   - 支持文本+图片混合内容
   - 处理单张/多张图片
   - 完整的错误处理（HTTP 错误、网络错误、认证失败）

2. **`validate_image_support()`** - 验证视觉能力支持
   - 发送简单文本请求测试端点连通性
   - 短超时（5 秒）快速验证

3. **`_prepare_image_for_api(image)`** - 图片预处理
   - URL: 直接使用
   - BASE64: 确保 data URI 前缀正确
   - PATH: 读取文件、压缩、转换为 base64

4. **`_resize_image_if_needed(img)`** - 图片压缩
   - 保持宽高比
   - 使用 LANCZOS 高质量重采样
   - 最大尺寸限制（默认 2048px）

5. **`_get_mime_type(format)`** - MIME 类型检测
   - 支持 JPEG、PNG、GIF、WEBP、BMP
   - 默认为 PNG

**Python 概念解释**:
- **PIL.Image**: Python 图像处理库，用于打开、调整大小、格式化图片
- **base64.b64encode**: 将二进制数据编码为 base64 字符串
- **BytesIO**: 内存中的字节流，避免临时文件
- **httpx**: 现代化的 HTTP 客户端，支持同步和异步请求

### 2. `src/libs/llm/__init__.py` (已修改)
**作用**: 更新 LLM 包导出，添加 AzureVisionLLM

### 3. `pyproject.toml` (已修改)
**作用**: 添加 Pillow 依赖到主依赖列表

**修改内容**:
```toml
# Document processing
"markitdown>=0.0.1",
"pillow>=10.0.0",  # Image processing for Vision LLM
```

**说明**: 原本 Pillow 在可选依赖 `[vision]` 中，由于 Vision LLM 是核心功能，移到主依赖

### 4. `tests/unit/test_azure_vision_llm.py`
**作用**: 完整的单元测试覆盖

**测试覆盖** (29 个测试用例):
- **初始化测试** (6 个): 参数验证、默认值、端点处理
- **chat 方法测试** (9 个): 纯文本、单张图片、多张图片、base64、错误处理
- **图片预处理测试** (5 个): URL、base64、PATH 三种类型
- **图片压缩测试** (3 个): 不需要压缩、宽度过大、高度过大
- **验证测试** (2 个): 成功和失败场景
- **MIME 类型测试** (3 个): JPEG、PNG、默认
- **便捷方法测试** (1 个): `describe_image()` 方法

---

## 关键设计解读

### Q1: 为什么需要图片压缩功能？

**解释**:
Azure OpenAI Vision API 对图片大小有限制：
- 过大的图片会导致 API 请求失败
- 图片越大，处理时间越长，成本越高
- 大多数情况下，压缩后的图片足以满足理解需求

**实现策略**:
```python
def _resize_image_if_needed(self, img: Image.Image) -> Image.Image:
    width, height = img.size

    if width <= self.max_image_size and height <= self.max_image_size:
        return img  # 不需要压缩

    # 保持宽高比计算新尺寸
    if width > height:
        new_width = self.max_image_size
        new_height = int(height * (self.max_image_size / width))
    else:
        new_height = self.max_image_size
        new_width = int(width * (self.max_image_size / height))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
```

**类比**:
就像压缩照片发朋友圈，虽然会损失一点细节，但传输更快，而且不影响看清楚内容。

**好处**:
- 降低 API 成本（按 token 计费）
- 提高响应速度
- 避免请求失败

### Q2: 为什么图片需要转换为 base64 格式？

**解释**:
Azure OpenAI Vision API 支持两种图片输入方式：

| 方式 | 格式 | 适用场景 |
|-----|------|---------|
| **URL** | `https://example.com/image.png` | 公网可访问的图片 |
| **Base64** | `data:image/png;base64,iVBORw0KG...` | 本地图片或需要预处理的图片 |

**Base64 转换过程**:
```python
# 1. 打开图片文件
with Image.open(image_path) as img:
    # 2. 压缩（如果需要）
    img = self._resize_image_if_needed(img)

    # 3. 转换为字节流
    buffered = BytesIO()
    img.save(buffered, format=img.format or "PNG")
    img_bytes = buffered.getvalue()

    # 4. Base64 编码
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # 5. 添加 data URI 前缀
    return f"data:{mime_type};base64,{img_base64}"
```

**类比**:
就像把图片"印"在信纸上，不需要对方联网就能看到图片。

**好处**:
- 支持本地文件
- 可以预处理（压缩、格式转换）
- 不依赖外部网络

### Q3: 为什么使用 `httpx` 而不是 `requests`？

**解释**:
`httpx` 是现代化的 HTTP 客户端，相比 `requests` 有以下优势：

| 特性 | httpx | requests |
|-----|-------|----------|
| 异步支持 | ✅ 原生支持 | ❌ 需要其他库 |
| HTTP/2 | ✅ 支持 | ❌ 不支持 |
| 类型提示 | ✅ 完整 | ❌ 不完整 |
| 维护状态 | ✅ 活跃维护 | ⚠️ 维护缓慢 |

**代码示例**:
```python
self._client = httpx.Client(
    headers={
        "Authorization": f"Bearer {self.api_key}",
        "api-key": self.api_key,  # Azure 特有
        "Content-Type": "application/json",
    },
    timeout=120.0,  # 图片处理需要更长时间
)
```

**好处**:
- 现代化的 API 设计
- 更好的错误处理
- 为未来异步扩展做准备

---

## 测试验证

**29 个测试用例全部通过**，验证了：

✅ **初始化功能**:
- 所有参数正确传递
- 默认值正确设置
- 端点 URL 正确处理（去除尾部斜杠）
- 缺少必需参数时抛出清晰错误

✅ **chat 方法功能**:
- 纯文本消息正常处理
- URL 图片正确格式化
- Base64 图片正确添加前缀
- 多张图片正确处理
- 空文本但带图片的合法消息
- 错误处理（HTTP 错误、网络错误、Azure 错误码）

✅ **图片预处理**:
- URL 直接使用
- Base64 自动添加前缀
- PATH 读取并转换
- 文件不存在时抛出错误

✅ **图片压缩**:
- 小图片不压缩
- 宽度过大时正确压缩
- 高度过大时正确压缩
- 保持宽高比

✅ **边界情况**:
- 空 messages 列表
- 非 VisionMessage 类型
- 各种 MIME 类型

**测试命令**:
```bash
python -m pytest tests/unit/test_azure_vision_llm.py -v
```

**测试结果**:
```
============================= 29 passed in 0.18s ==============================
```

---

## 任务完成度

### 满足的任务要求

- [x] **AzureVisionLLM 类实现**: 继承 BaseVisionLLM，实现所有抽象方法
- [x] **Azure API 调用**: 正确格式化请求，解析响应
- [x] **支持多种图片输入**: URL、base64、PATH 三种类型
- [x] **图片压缩功能**: 自动压缩超过 max_image_size 的图片
- [x] **完整错误处理**: HTTP 错误、网络错误、认证失败、Azure 错误码
- [x] **验证方法**: validate_image_support() 检查端点连通性
- [x] **测试覆盖**: 29 个测试用例，覆盖所有场景

### 超出任务范围的额外功能

- [x] **便捷方法**: `describe_image()` 简化单张图片描述
- [x] **MIME 类型检测**: 自动从 PIL Image 格式检测 MIME 类型
- [x] **日志记录**: 详细的调试日志，方便问题排查
- [x] **更长的超时**: 120 秒超时，适应图片处理需求

---

## 项目进度

### 阶段 B：Libs 可插拔层

| 子任务 | 状态 |
|-------|------|
| B7.1-B7.8 | ✅ 已完成 |
| B8 | ✅ 已完成 |
| **B9** | ✅ **已完成** |

**🎉 阶段 B 完成！**

所有 Libs 层的可插拔组件已实现完毕：
- ✅ LLM: OpenAI, Azure, DeepSeek, Ollama, GLM
- ✅ Embedding: OpenAI, Azure, Ollama
- ✅ Splitter: RecursiveCharacter
- ✅ VectorStore: ChromaDB
- ✅ Reranker: None, LLM, Cross-Encoder
- ✅ **Vision LLM: BaseVisionLLM + AzureVisionLLM** 🆕

**下一步**: 阶段 C - Ingestion Pipeline MVP

---

## 使用示例

### 示例 1: 使用 URL 描述图片

```python
from src.libs.llm.azure_vision_llm import AzureVisionLLM
from src.libs.llm.base_vision_llm import VisionMessage, ImageContent, ImageType

# 创建 Azure Vision LLM 实例
vision_llm = AzureVisionLLM(
    model="gpt-4o",
    api_key="your-azure-api-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    api_version="2024-02-15-preview"
)

# 创建包含 URL 图片的消息
image = ImageContent(
    data="https://example.com/image.png",
    image_type=ImageType.URL
)
message = VisionMessage(
    role="user",
    content="请描述这张图片的内容",
    images=[image]
)

# 发送请求
response = vision_llm.chat([message])
print(response.content)  # LLM 生成的描述
print(response.images_processed)  # 1
```

### 示例 2: 使用本地文件

```python
# 使用本地图片文件
image = ImageContent(
    data="/path/to/local/image.jpg",
    image_type=ImageType.PATH,
    mime_type="image/jpeg"
)
message = VisionMessage(
    role="user",
    content="这是什么？",
    images=[image]
)

response = vision_llm.chat([message])
print(response.content)
```

### 示例 3: 使用 Base64 编码的图片

```python
# Base64 编码的图片
image = ImageContent(
    data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ...",
    image_type=ImageType.BASE64,
    mime_type="image/png"
)
message = VisionMessage(
    role="user",
    content="分析这个图表",
    images=[image]
)

response = vision_llm.chat([message])
print(response.content)
```

### 示例 4: 使用便捷方法

```python
# describe_image 便捷方法
image = ImageContent(
    data="https://example.com/photo.jpg",
    image_type=ImageType.URL
)

response = vision_llm.describe_image(
    image,
    prompt="详细描述这张照片"
)
print(response.content)
```

### 示例 5: 处理多张图片

```python
# 多张图片对比
images = [
    ImageContent(data=f"https://example.com/product{i}.jpg", image_type=ImageType.URL)
    for i in range(1, 4)
]

message = VisionMessage(
    role="user",
    content="比较这三张产品的图片，说明它们的主要区别",
    images=images
)

response = vision_llm.chat([message])
print(response.content)
print(response.images_processed)  # 3
```

### 示例 6: 自定义图片大小限制

```python
# 创建自定义最大图片尺寸的实例
vision_llm = AzureVisionLLM(
    model="gpt-4o",
    api_key="your-azure-api-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    max_image_size=1024  # 最大 1024px（默认 2048px）
)

# 大于 1024px 的图片会自动压缩
image = ImageContent(
    data="/path/to/large-image.jpg",  # 假设是 3000x2000
    image_type=ImageType.PATH
)

response = vision_llm.describe_image(image)
# 图片会被自动压缩到 1024x683
```

---

## 技术亮点

### 1. 完整的图片类型支持
- **URL**: 公网图片直接使用
- **Base64**: 支持带或不带 data URI 前缀
- **PATH**: 本地文件自动读取和处理

### 2. 智能图片处理
- **自动压缩**: 超过限制自动调整大小
- **保持宽高比**: 避免图片变形
- **高质量重采样**: 使用 LANCZOS 算法
- **MIME 类型检测**: 自动识别图片格式

### 3. 健壮的错误处理
- **HTTP 错误**: 提取 Azure 错误码和消息
- **网络错误**: 清晰的连接错误提示
- **认证失败**: 明确的 API 密钥错误
- **文件错误**: 文件不存在时的清晰提示

### 4. 长超时支持
- **120 秒超时**: 适应图片处理需求
- **验证短超时**: validate 使用 5 秒快速验证
- **可配置**: 通过 kwargs 覆盖默认超时

### 5. 完整的测试覆盖
- **29 个测试用例**: 覆盖所有功能点
- **Mock HTTP 请求**: 避免真实 API 调用
- **边界情况测试**: 空、无效、极端输入
- **错误场景测试**: 各种失败情况

---

## 依赖变更

### 新增依赖
- **pillow>=10.0.0**: 图片处理库

**变更原因**:
- 原本在可选依赖 `[vision]` 中
- Vision LLM 是核心功能，移到主依赖
- 用于图片压缩、格式转换、base64 编码

**Pillow 功能说明**:
```python
from PIL import Image

# 打开图片
with Image.open("path/to/image.jpg") as img:
    # 获取尺寸
    width, height = img.size

    # 调整大小
    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 获取格式
    format = img.format  # "JPEG", "PNG", etc.

    # 保存到字节流
    from io import BytesIO
    buffered = BytesIO()
    img.save(buffered, format="PNG")
```

---

## 已知限制和未来改进

### 当前限制
1. **仅支持 Azure OpenAI**: 未来可添加 OpenAI Vision、DashScope 等
2. **同步 API**: 未来可添加异步支持
3. **本地文件读取**: PATH 类型需要本地文件系统访问

### 未来改进方向
1. **OpenAI Vision 实现**: 添加 OpenAIVisionLLM 类
2. **异步支持**: 使用 httpx.AsyncClient
3. **图片缓存**: 避免重复处理相同图片
4. **批量处理**: 支持批量图片描述生成
5. **更多图片格式**: 支持 TIFF、SVG 等

---

## 集成到项目

### 配置示例

`config/settings.yaml`:
```yaml
vision_llm:
  provider: azure
  model: gpt-4o
  azure_endpoint: "https://your-resource.openai.azure.com"
  api_key: "${AZURE_OPENAI_API_KEY}"
  api_version: "2024-02-15-preview"
  temperature: 0.7
  max_tokens: 2048
  timeout_seconds: 30
```

### 在 Ingestion Pipeline 中使用

未来的 `transform/image_captioner.py` 可以这样使用：

```python
from src.libs.llm.azure_vision_llm import AzureVisionLLM
from src.core.settings import load_settings

# 加载配置
settings = load_settings()

# 创建 Vision LLM（未来需要 VisionLLMFactory）
vision_llm = AzureVisionLLM(
    model=settings.vision_llm.model,
    api_key=settings.vision_llm.api_key,
    azure_endpoint=settings.vision_llm.azure_endpoint,
    api_version=settings.vision_llm.api_version
)

# 为图片生成描述
image = ImageContent(data=image_path, image_type=ImageType.PATH)
response = vision_llm.describe_image(image, prompt="描述这张图片")

# 将描述注入到 Chunk
chunk.metadata["image_caption"] = response.content
```

---

## 总结

B9 任务成功实现了 Azure Vision LLM 提供商，完成了以下目标：

✅ **完整的 Azure OpenAI Vision API 集成**
✅ **支持三种图片输入方式（URL/Base64/PATH）**
✅ **自动图片压缩和格式转换**
✅ **健壮的错误处理和清晰的错误信息**
✅ **全面的测试覆盖（29 个测试用例）**
✅ **便捷的开发体验（describe_image 方法）**

这为后续的 **C7: Image Captioner** 任务奠定了坚实基础，实现了 RAG 系统的多模态能力。
