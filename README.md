# Messenger：每日论文抓取、解读与 RSS 发布

本项目用于**按研究主题每日抓取论文**（arXiv / HuggingFace Daily），自动完成：

- **基础解读**（摘要结构化 + 相关性判断）
- **下载 PDF + OCR 解析**（支持 MinerU 官方 API 与自建服务双后端）
- **深度解读**
- **生成/更新 RSS**（`arxiv.rss` / `SpatialAI.rss` / `Huggingface.rss`）
- **清理旧 RSS 条目**
- **推送 RSS 到子目录仓库 `messenger/` 并 `git push`**

---

## 目录结构（核心）

- `config/`
  - `ai.py`：LLM/OCR 客户端与环境变量读取（LLM 双提供商 + OCR 双后端）
  - `prompt.py`：提示词模板（基础摘要/相关性/深度解读/修正）
  - `topic.yml`：抓取主题与查询语句
- `pipeline/`（`_pro` 为多线程版本，生产使用；顺序版为备选）
  - `fetch_arxiv.py`：按 `topic.yml` 抓取 arXiv → `storage/fetch-arxiv/arxiv_data_{date}.csv`
  - `fetch_hf_daily.py`：抓取 HuggingFace Daily → `storage/fetch-hf-daily/hf_papers_{date}.csv`
  - `update_paper_list.py`：合并为 master 表 → `storage/papers_master.csv`
  - `analyze_01_base_pro.py`：基础分析 → `storage/analysis/base/*.json` 并更新 master 的 `base_analysis/relevance`
  - `analyze_02_parse_pro.py`：下载 PDF + OCR 解析 → `storage/papers/pdfs/`、`storage/papers/parse/` 并更新 master 的 `download`
  - `analyze_03_deep_pro.py`：深度解读 → `storage/analysis/deep/*.json` 并更新 master 的 `deep_analysis`
  - `publish_add_new_items.py`：把已 deep 的论文追加到对应 RSS，并更新 master 的 `publish`
  - `publish_delete_old_items.py`：删除 RSS 中超过 N 天的条目
  - `run_daily.sh` / `run_daily.py`：完整流水线调度入口
- `scripts/`
  - `publish_rss.sh`：拷贝三个 RSS 到子目录 `messenger/`，并在子目录执行 `git add/commit/push`
- `storage/`
  - `papers_master.csv`：主状态表（流程驱动核心）
  - `analysis/base/`：基础解读产物
  - `papers/pdfs/`：PDF 文件
  - `papers/parse/`：OCR 解析产物（含 `md_content`）
  - `analysis/deep/`：深度解读产物
  - `logs/`：运行日志（`run_pipeline-*.log` 由 run_daily.sh 产生）

---

## Master 表字段说明（`storage/papers_master.csv`）

- `base_analysis`：是否完成基础解读
- `relevance`：是否相关（True/False）
- `download`：是否已下载 PDF（并且 parse 可生成）
- `deep_analysis`：是否完成深度解读
- `publish`：是否已发布到 RSS

---

## 环境变量（LLM / OCR）

项目通过 `config/ai.py` 从环境变量读取配置（推荐 `.env` + `set -a; source .env; set +a`，`run_daily.sh` 会自动加载）：

- **OpenAI 兼容**
  - `LLM_PROVIDER=openai_compat`
  - `LLM_BASE_URL=http://host:port/v1`
  - `LLM_MODEL=...`（以你的 `/v1/models` 为准）
  - `LLM_API_KEY=...`（可为空）
  - `LLM_MAX_TOKENS`（含思考 token，默认 8192）/ `LLM_THINKING`（`""`=默认 / `off` / `on`）
- **智谱（Zhipu）**
  - `LLM_PROVIDER=zhipu`；`ZHIPU_API_KEY=...`；`ZHIPU_MODEL=glm-4.5-flash`；`ZHIPU_MAX_TOKENS`（默认 2400）
- **OCR（双后端，`OCR_PROVIDER` 二选一，默认 cloud）**
  - `OCR_PROVIDER=cloud`：MinerU 官方 API（`MINERU_KEY` 必填；可选 `MINERU_CLOUD_MODEL/LANGUAGE/POLL_INTERVAL/TIMEOUT` 等）
  - `OCR_PROVIDER=local`：自建 MinerU HTTP（`MINERU_OCR_URL=http://host:port/file_parse`、`MINERU_OCR_FILE_FIELD=files`）
  - `OCR_ENABLED=True/False` 为总开关
- `INTEREST_DESCRIPTION`：研究方向描述，用于相关性筛选（未设置时使用代码内置兜底值）

---

## 运行方式

### 方式 A：完整流水线（生产，cron 周一至周六 22:50 自动执行）

```bash
bash run_daily.sh
```

- 自动加载 `.env`、设置 NO_PROXY、**预检 LLM/OCR 服务**（不可达立即失败，避免整夜空转）
- 按顺序执行：抓取 → 合并 → 基础分析 → PDF+OCR → 深度分析 → 发布 → 清理 → 推送 RSS
- 日志输出：`storage/logs/run_pipeline-YYYYmmdd-HHMMSS.log`

`run_daily.py` 为 Python 版调度器（`python run_daily.py --once`），功能等价但**不会自动加载 `.env`**，使用前请先 `set -a; source .env; set +a`。

### 方式 B：单脚本运行（按需，先加载环境变量）

```bash
set -a; source .env; set +a
.venv/bin/python pipeline/analyze_01_base_pro.py    # 或顺序版 analyze_01_base.py
.venv/bin/python pipeline/analyze_02_parse_pro.py
.venv/bin/python pipeline/analyze_03_deep_pro.py --limit 5 --workers 2
.venv/bin/python pipeline/publish_add_new_items.py --run_pubdate "Wed, 01 Jan 2025 14:01:00 +0800"
.venv/bin/python pipeline/publish_delete_old_items.py --days 14
```

---

## RSS 相关

- RSS 输出文件（项目根目录、已 gitignore）：
  - `arxiv.rss`：arXiv 来源的相关论文
  - `SpatialAI.rss`：空间 AI 子主题
  - `Huggingface.rss`：HuggingFace Daily 论文（含深度解读）
- `publish_add_new_items.py` 将 HTML 写入 `<description>`（CDATA/实体转义），RSS 阅读器可直接渲染
- 推送到子目录仓库：`scripts/publish_rss.sh`（复制 3 个 feed 到 `messenger/` 并 `git add/commit/push`）

---

## 常见问题

- **`.env` source 了但 Python 读不到变量**：请使用 `set -a; source .env; set +a`（run_daily.sh 已内置）
- **LLM 报 model not found**：用 `curl $LLM_BASE_URL/models` 查真实 `id` 并设置 `LLM_MODEL`
- **OCR 422 缺字段 / local 模式连不上**：MinerU 通常要求字段名 `files`（`MINERU_OCR_FILE_FIELD=files`）；检查 `MINERU_OCR_URL`
- **OCR_PROVIDER=cloud 报 Token 错误**：检查 `MINERU_KEY` 是否有效、当日额度（每账号每天 1000 页高优先级）
- **`python run_daily.py` 跑完没产出**：该入口不加载 `.env`，先 `set -a; source .env; set +a`
