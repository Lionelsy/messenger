# Messager：每日论文抓取、解读与 RSS 发布

本项目用于**按研究主题每日抓取论文**（arXiv / HuggingFace Daily），自动完成：

- **基础解读**（摘要结构化 + 相关性判断）
- **下载 PDF + OCR 解析**
- **深度解读**
- **生成/更新 RSS**（`arxiv.rss`）
- **清理旧 RSS 条目**
- **可选：推送 RSS 到子目录仓库 `Messager/` 并 `git push`**

---

## 目录结构（核心）

- `config/`
  - `ai.py`：LLM/OCR 客户端与环境变量读取
  - `prompt.py`：提示词模板（基础摘要/相关性/深度解读/修正）
  - `topic.yml`：抓取主题与查询语句
- `pipeline/`
  - `fetch_arxiv.py`：按 `topic.yml` 抓取 arXiv → `storage/fetch-arxiv/arxiv_data_{date}.csv`
  - `fetch_hf_daily.py`：抓取 HuggingFace Daily → `storage/fetch-hf-daily/hf_papers_{date}.csv`
  - `update_paper_list.py`：合并为 master 表 → `storage/papers_master.csv`
  - `analyze_01_base.py`：基础分析 → `storage/analysis/base/*.json` 并更新 master 的 `base_analysis/relevance`
  - `analyze_02_parse.py`：下载 PDF + OCR 解析 → `storage/papers/pdfs/`、`storage/papers/parse/` 并更新 master 的 `download`
  - `analyze_03_deep.py`：深度解读 → `storage/analysis/deep/*.json` 并更新 master 的 `deep_analysis`
  - `publish_add_new_items.py`：把已 deep 的论文追加到 `arxiv.rss`，并更新 master 的 `publish`
  - `publish_delete_old_items.py`：删除 RSS 中超过 N 天的条目
- `scripts/`
  - `publish_rss.sh`：拷贝根目录 `arxiv.rss` 到子目录 `Messager/arxiv.rss`，并在子目录执行 `git add/commit/push`
- `storage/`
  - `papers_master.csv`：主状态表（流程驱动核心）
  - `analysis/base/`：基础解读产物
  - `papers/pdfs/`：PDF 文件
  - `papers/parse/`：OCR 解析产物（含 `md_content`）
  - `analysis/deep/`：深度解读产物
  - `logs/`：运行日志（`run_daily.py` / `bash_test.sh`）

---

## Master 表字段说明（`storage/papers_master.csv`）

- `base_analysis`：是否完成基础解读
- `relevance`：是否相关（True/False）
- `download`：是否已下载 PDF（并且 parse 可生成）
- `deep_analysis`：是否完成深度解读
- `publish`：是否已发布到 RSS

---

## 环境变量（LLM / OCR）

项目通过 `config/ai.py` 从环境变量读取配置，常用项：

- **OpenAI 兼容**
  - `LLM_PROVIDER=openai_compat`
  - `LLM_BASE_URL=http://host:port/v1`
  - `LLM_MODEL=gpt-oss:120b`（以你的 `/v1/models` 为准）
  - `LLM_API_KEY=...`（可为空）
- **智谱（Zhipu）**
  - `LLM_PROVIDER=zhipu`
  - `ZHIPU_API_KEY=...`
  - `ZHIPU_MODEL=glm-4.5-flash`
- **MinerU OCR**
  - `OCR_ENABLED=True`
  - `MINERU_OCR_URL=http://host:port/file_parse`
  - `MINERU_OCR_FILE_FIELD=files`

如果你使用 `.env`，请记得在 shell 中 export：

```bash
set -a; source .env; set +a
```

---

## 运行方式

### 方式 A：调度器（推荐）

周一到周六 **14:01** 自动触发一次，且会把“当日日期/时间戳”统一传给抓取与发布脚本，避免前后不一致。

```bash
cd /home/shuyu/workplace/others/Messager
/home/shuyu/anaconda3/envs/workplace/bin/python run_daily.py
```

- 日志输出：`storage/logs/run_daily-YYYYmmdd-HHMMSS.log`

调试立刻跑一次：

```bash
/home/shuyu/anaconda3/envs/workplace/bin/python run_daily.py --once
```

### 方式 B：手动逐个执行（`bash_test.sh`）

```bash
cd /home/shuyu/workplace/others/Messager
./bash_test.sh
```

- 日志输出：`storage/logs/bash_test-YYYYmmdd-HHMMSS.log`

### 方式 C：单脚本运行（按需）

示例：

```bash
python pipeline/analyze_01_base.py
python pipeline/analyze_02_parse.py
python pipeline/analyze_03_deep.py
python pipeline/publish_add_new_items.py
python pipeline/publish_delete_old_items.py --days 30
```

---

## RSS 相关

- RSS 输出文件：项目根目录 `arxiv.rss`
- `publish_add_new_items.py` 使用 CDATA 包裹 HTML，RSS 阅读器可直接渲染
- 推送到子目录仓库：`scripts/publish_rss.sh`

---

## 常见问题

- **`.env source 了但 Python 读不到变量**：请使用 `set -a; source .env; set +a` 或 `export $(grep -v '^#' .env | xargs)`
- **LLM 报 model not found**：用 `curl $LLM_BASE_URL/models` 查真实 `id` 并设置 `LLM_MODEL`
- **OCR 422 缺字段**：MinerU 通常要求字段名 `files`（`MINERU_OCR_FILE_FIELD=files`）

