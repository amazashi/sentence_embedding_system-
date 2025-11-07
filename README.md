# 句子嵌入批处理和索引构建系统

基于 `all-MiniLM-L6-v2` 模型的句子嵌入批处理系统，支持高效的相似性搜索和索引构建。

## 功能特性

- 🚀 **批量处理**: 支持单文件和目录批量处理Markdown文档
- 🧠 **智能嵌入**: 使用 `all-MiniLM-L6-v2` 模型生成高质量句子嵌入
- 🗄️ **数据库存储**: SQLite数据库存储嵌入向量和元数据
- 🔍 **快速搜索**: 基于FAISS的高效相似性搜索
- 📊 **统计分析**: 提供详细的数据库和索引统计信息
- 🛠️ **命令行界面**: 简洁易用的CLI工具

## 项目结构

```
sentence_embedding_system/
├── venv/                    # Python虚拟环境
├── database.py             # 数据库管理模块
├── embedding_processor.py  # 嵌入处理器
├── search_index.py         # 搜索索引模块
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖包列表
├── README.md              # 项目文档
├── embeddings.db          # SQLite数据库文件（运行后生成）
├── similarity_index.faiss # FAISS索引文件（运行后生成）
└── embedding_system.log   # 系统日志文件（运行后生成）
```

## 安装和设置

### 1. 环境要求

- Python 3.8+
- Windows 10/11 (已在Windows环境下测试)

### 2. 安装依赖

项目已配置独立的Python虚拟环境，直接激活并安装依赖：

```bash
# 进入项目目录
cd sentence_embedding_system

# 激活虚拟环境
.\venv\Scripts\activate

# 安装依赖包
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python main.py --help
```

## 使用指南

### 基本命令

系统提供以下主要命令：

- `process`: 处理文件生成嵌入
- `build-index`: 构建搜索索引
- `search`: 搜索相似句子
- `stats`: 显示统计信息
- `clear`: 清空所有数据

### 1. 处理文件

#### 处理单个文件
```bash
python main.py process --file "path/to/document.md"
```

#### 批量处理目录
```bash
python main.py process --directory "path/to/documents" --pattern "*.md"
```

#### 自定义批处理大小
```bash
python main.py process --file "document.md" --batch-size 64
```

### 2. 构建搜索索引

```bash
# 使用默认的平面索引（适合小规模数据）
python main.py build-index

# 使用IVF索引（适合大规模数据）
python main.py build-index --index-type ivf
```

### 3. 相似性搜索

```bash
# 基本搜索
python main.py search --query "机器学习算法"

# 自定义返回结果数量
python main.py search --query "深度学习" --top-k 10

# 设置相似度阈值
python main.py search --query "神经网络" --top-k 5 --threshold 0.3
```

### 4. 查看统计信息

```bash
python main.py stats
```

### 5. 清空数据

```bash
# 交互式清空（会提示确认）
python main.py clear

# 直接清空（跳过确认）
python main.py clear --confirm
```

## 高级配置

### 自定义模型

```bash
python main.py --model "sentence-transformers/paraphrase-MiniLM-L6-v2" process --file document.md
```

### 自定义数据库和索引文件

```bash
python main.py --database "custom.db" --index-file "custom_index.faiss" process --file document.md
```

### 日志级别设置

```bash
python main.py --log-level DEBUG process --file document.md
```

## 系统架构

### 核心模块

1. **EmbeddingDatabase** (`database.py`)
   - 管理SQLite数据库
   - 存储文档、句子和嵌入向量
   - 提供数据查询和统计功能

2. **SentenceEmbeddingProcessor** (`embedding_processor.py`)
   - 加载和管理句子嵌入模型
   - 处理Markdown文件，提取和清理句子
   - 批量生成嵌入向量

3. **SimilaritySearchIndex** (`search_index.py`)
   - 基于FAISS的高效搜索索引
   - 支持多种索引类型（Flat, IVF）
   - 提供相似性搜索功能

### 数据流程

1. **文档处理**: Markdown文件 → 句子提取 → 文本清理
2. **嵌入生成**: 清理后的句子 → 模型编码 → 嵌入向量
3. **数据存储**: 嵌入向量 + 元数据 → SQLite数据库
4. **索引构建**: 数据库中的嵌入 → FAISS索引
5. **相似性搜索**: 查询文本 → 嵌入编码 → 索引搜索 → 结果排序

## 性能优化

### 批处理大小

- 小文件（<1000句子）: `--batch-size 32`
- 中等文件（1000-5000句子）: `--batch-size 64`
- 大文件（>5000句子）: `--batch-size 128`

### 索引类型选择

- **Flat索引**: 精确搜索，适合<10万向量
- **IVF索引**: 近似搜索，适合>10万向量，速度更快

### 内存优化

- 使用CPU版本的PyTorch和FAISS
- 批量处理避免内存溢出
- 及时释放不需要的数据

## 故障排除

### 常见问题

1. **模型下载失败**
   ```bash
   # 手动下载模型
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

2. **虚拟环境激活失败**
   ```bash
   # 设置执行策略
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **依赖包安装失败**
   ```bash
   # 升级pip
   python -m pip install --upgrade pip
   # 重新安装
   pip install -r requirements.txt
   ```

### 日志分析

系统会生成详细的日志文件 `embedding_system.log`，包含：
- 模型加载信息
- 文件处理进度
- 索引构建状态
- 错误和警告信息

## 扩展开发

### 添加新的文档格式

在 `embedding_processor.py` 中扩展 `extract_sentences_from_markdown` 方法。

### 自定义相似性度量

在 `search_index.py` 中修改FAISS索引配置。

### 集成其他嵌入模型

修改 `SentenceEmbeddingProcessor` 类的模型加载逻辑。

## 许可证

本项目采用 GNU GPL v3.0 许可证。

你可以自由复制、分发和修改本项目，但必须在同一许可证下发布你的修改版本，并保留原有的版权和许可证声明。该许可证不提供任何形式的担保。详情请参阅项目根目录的 `LICENSE` 文件或访问 https://www.gnu.org/licenses/gpl-3.0.html。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 更新日志

### v1.0.0 (2024-10-30)
- 初始版本发布
- 支持Markdown文件批处理
- 集成all-MiniLM-L6-v2模型
- 实现SQLite数据库存储
- 提供FAISS索引搜索
- 完整的CLI界面