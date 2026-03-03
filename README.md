# 📂 Web File Analyzer Agent

An AI-powered agent that takes a URL, scans the page for downloadable files (PDF, DOCX, XLSX, CSV), extracts their content, and produces concise summaries — all presented through a conversational Streamlit interface with human-in-the-loop review.

> **Case Study** — This repository was developed for **Tanı Pazarlama ve İletişim Hizmetleri A.Ş.** as described below.

<details>
<summary><strong>📄 Case Study Brief (Turkish)</strong></summary>

**Senaryo**

Birçok kurum ve kuruluş, düzenli olarak yayımladıkları raporları, veri setlerini ve dökümanları web siteleri üzerinden indirilebilir dosyalar halinde sunmaktadır. Bu dosyalar farklı formatlarda (Excel, Word, PDF vb.) olabilir ve genellikle belirli bir hiyerarşi içinde (dönem, kategori, konu vb.) organize edilmiştir.

Sizden beklenen, kullanıcının verdiği bir URL'yi girdi olarak alan ve aşağıdaki adımları gerçekleştiren bir sistem tasarlamanızdır:

1. **Sayfayı anlama** — Sayfaya gidip içeriğini analiz eder, sayfanın hangi kuruma/sektöre ait olduğunu ve ne tür içerik sunduğunu anlar.
2. **Yapıyı keşfetme** — Sayfadaki içerik organizasyonunu (dönemsel gruplandırma, kategori yapısı vb.) tespit eder.
3. **Dosya tespiti** — İndirilebilir dosya linklerini bulur, dosya türlerini ve isimlerini belirler.
4. **İçerik analizi** — Dosya içeriklerini okuyarak kısa ve anlamlı bir özet üretir (Excel'de sheet adları ve sütun başlıkları, Word/PDF'te ana başlıklar ve konular gibi).
5. **Yapılandırılmış çıktı** — Tüm sonuçları anlamlı bir hiyerarşi içinde kullanıcıya sunar.

Sistem, kullanıcının ihtiyacına göre çeşitli filtreleme parametreleri kabul edebilmelidir (dönem, kategori, dosya türü vb.).

Yukarıdaki akış temel beklentidir. Bunun ötesinde, senaryonun genel yapısını bozmadan sistemi daha iyi hale getireceğini düşündüğünüz iyileştirmeler (örneğin human-in-the-loop adımları, doğrulama mekanizmaları, akış optimizasyonları vb.) varsa bunları uygulayıp uygulamamak sizin inisiyatifinize kalmıştır.

</details>

---

## ✨ What Can You Query?

Simply type a natural language message containing a URL and any combination of filters:

| Filter | Example |
|---|---|
| **Time period** | *"Find 2025 files from this website: https://example.com/reports"* |
| **Categories** | *"Find files about Automotive industry on this website: …"* |
| **File type** | *"Only PDFs and XLSX files from …"* |
| **Combined** | *"Find 2024–2025 Retail reports (xlsx) from …"* |

**Supported file formats:** PDF · DOCX · XLSX · CSV

---

## 🏗️ Architecture

The system is built as a **LangGraph** state machine with five nodes and conditional routing:

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph Flow                           │
│                                                                 │
│  START                                                          │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────┐   no URL    ┌─────┐                      │
│  │ query_extractor   │───────────▶│ END │                      │
│  └──────────────────┘             └─────┘                      │
│    │ has URL                                                    │
│    ▼                                                            │
│  ┌──────────────────┐  tool_call  ┌───────┐                    │
│  │structure_analyzer │◀──────────▶│ tools │ (firecrawl MCP)    │
│  └──────────────────┘             └───────┘                    │
│    │ analysis ready (or error/no files → END)                   │
│    ▼                                                            │
│  ┌──────────────────┐  ◀── interrupt_before (human-in-the-loop)│
│  │  human_review     │                                          │
│  └──────────────────┘                                          │
│    │ accept          │ reject                                   │
│    ▼                  ▼                                          │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  file_reader      │  │ query_extractor   │ (re-analyze)      │
│  └──────────────────┘  └──────────────────┘                    │
│    │                                                            │
│    ▼                                                            │
│  END (final output with summaries)                              │
└─────────────────────────────────────────────────────────────────┘
```

### Nodes

| Node | Agent | What it does |
|---|---|---|
| `query_extractor` | `QueryExtractorAgent` | Parses the user's natural-language input into structured `Filters` (URL, file types, categories, date range) using **structured output** with an LLM. Rejects irrelevant queries gracefully. |
| `structure_analyzer` | `StructureAnalyzerAgent` | Calls the **Firecrawl MCP** scrape tool to fetch the page, then asks the LLM to extract every downloadable file link grouped by section. Applies user filters. Handles errors (unreachable URL, no files found). |
| `tools` | `ToolNode` (LangGraph) | Executes the Firecrawl `firecrawl_scrape` tool call generated by the structure analyzer. |
| `human_review` | *(interrupt)* | The graph **pauses** here. The user sees the extracted file structure in the Streamlit UI and can **Accept** or **Reject** (with feedback). |
| `file_reader` | `FileReaderAgent` | Downloads each file, extracts content with type-specific readers, and generates a 1–2 sentence LLM summary per file. Runs up to 2 files concurrently via `asyncio.Semaphore`. |

### Human-in-the-Loop

After the structure analyzer finds files, the graph **interrupts before `human_review`**. The Streamlit UI displays the extracted file structure with title, description, applied filters, and file list. The user can:

- **✅ Accept** → proceeds to `file_reader` which downloads and summarizes every file.
- **❌ Reject** → opens a feedback text box. The feedback is routed back to `query_extractor` so the LLM re-analyzes the cached page content with the new instructions (e.g. *"remove files with these URLs"*, *"only include 2024 reports"*).

### MCP Tool — Firecrawl

Web scraping is handled by the official **[Firecrawl MCP server](https://docs.firecrawl.dev/mcp-server)**. The `langchain-mcp-adapters` library spawns `npx firecrawl-mcp` as a stdio subprocess and exposes it as a LangChain `BaseTool`. The tool is bound to the structure analyzer's LLM so it can call `firecrawl_scrape` with parameters like `formats: ["markdown"]`, `onlyMainContent: true`, and `waitFor: 30000`.

### Cache Mechanism

A **SQLite-backed cache** (`src/cache.py`) stores:

| Cache | Key | Value | Benefit |
|---|---|---|---|
| **Page cache** | URL | Raw markdown from Firecrawl | Skips the entire scrape + tool-call round-trip on repeated URLs |
| **File summary cache** | File download URL | `FileSummary` (name, type, summary) | Skips download + LLM summarization for already-processed files |

Hit/miss counters are persisted in the same SQLite database and displayed in the Streamlit sidebar. A "Clear All Caches" button resets everything.

When running with Docker, the cache database is stored in a **named volume** (`cache-data`) so it survives container restarts.

### File Reader Tools

Each supported format has a dedicated reader tool:

| Tool | Library | Extraction strategy |
|---|---|---|
| `pdf_reader_tool` | `pdfplumber` | First 3 pages, repeated headers stripped, truncated to ~4 000 chars |
| `docx_reader_tool` | `python-docx` | All paragraph text, truncated to ~4 000 chars |
| `excel_reader_tool` | `openpyxl` | Per-sheet: column headers, row count, first 5 sample rows |
| `csv_reader_tool` | `pandas` | Column headers, row count, first 5 sample rows |

All tools use a shared `_download_file()` helper (`httpx`, 50 MB limit, follow redirects).

### FAISS Vector Store (Optional)

After summarization, file summaries are optionally stored in a **FAISS** in-memory vector store (`langchain-community` + `OpenAIEmbeddings`). This enables semantic similarity search across file summaries for future features.

---

## 📁 Project Structure

```
Web-File-Analyzer-Agent/
├── docker-compose.yml          # Docker Compose (web-app + cache volume)
├── requirements.in             # Direct dependencies
├── requirements.txt            # Pinned dependencies (pip-compile output)
├── .env                        # API keys (not committed)
├── README.md
│
└── src/
    ├── Dockerfile              # Python 3.11 + Node.js 20 image
    ├── main.py                 # Entry point — launches Streamlit
    ├── app.py                  # Streamlit UI (chat, review, sidebar)
    ├── cache.py                # SQLite cache manager (singleton)
    │
    ├── models/
    │   └── chat_models.py      # LLM factory (OpenAI wrapper)
    │
    └── agents/
        ├── base_agent.py               # Abstract base class
        ├── state.py                    # Pydantic models + TypedDict state
        ├── graph.py                    # LangGraph workflow definition
        ├── query_extractor_agent.py    # NLU → Filters
        ├── structure_analyzer_agent.py # Page scrape → file structure
        ├── file_reader_agent.py        # Download → summarize files
        │
        ├── prompts/
        │   ├── query_extractor_prompts.py
        │   ├── structure_analyzer_prompts.py
        │   └── file_reader_prompts.py
        │
        └── tools/
            ├── fetch_page_tool.py      # Firecrawl MCP client
            ├── download_utils.py       # Shared HTTP download helper
            ├── pdf_reader_tool.py
            ├── docx_reader_tool.py
            ├── excel_reader_tool.py
            └── csv_reader_tool.py
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Web-File-Analyzer-Agent.git
cd Web-File-Analyzer-Agent
```

### 2. Create your `.env` file

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=YOUR-OPENAI-API-KEY
FIRECRAWL_API_KEY=YOUR-FIRECRAWL-API-KEY
```

- **OpenAI API Key** — get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Firecrawl API Key** — get one for free at [firecrawl.dev/app/api-keys](https://www.firecrawl.dev/app/api-keys) (gives **500 free credits**, no payment required — more than enough for this case study)

### 3a. Run with Docker (recommended)

```bash
docker compose up --build
```

The Streamlit app will be available at **http://localhost:8501**.

The Docker image installs Python 3.11 and Node.js 20 (needed for the Firecrawl MCP server via `npx`). A named volume `cache-data` persists the SQLite cache across container restarts.

### 3b. Run locally (without Docker)

**Prerequisites:** Python 3.11+, Node.js 18+ (for `npx`)

```bash
pip install -r requirements.txt
python src/main.py
```

The Streamlit app will be available at **http://localhost:8501**.

> **⚠️ Note for `pip-compile` users:** If you regenerate `requirements.txt` with `pip-compile`, the `mcp` package may pull in `pywin32` as a transitive dependency. **Remove the `pywin32` line** from `requirements.txt` before building the Docker image — it is a Windows-only package and will fail the Linux container build.

---

## 📸 Screenshots

### Category-Based Search
![Category Based Search](docs/screenshots/category%20based%20search.png)

### Human-in-the-Loop Review — File Structure Output
![Human Review](docs/screenshots/file%20structure%20human%20in%20the%20loop%20output.png)

### Rejected Analysis — Re-analysis After Feedback
![Rejected Feedback](docs/screenshots/file%20structure%20human%20in%20the%20loop%20output-after%20feedback.png)

### Human-in-the-Loop — Accept / Reject Flow
![Human in the Loop](docs/screenshots/file%20structure%20human%20in%20the%20loop.png)

### Cache Sidebar — Stored Items & Hit/Miss Stats
![Cache Sidebar](docs/screenshots/cache%20stats%20after%20repeating%20request.png)

### Error Handling — Unreachable URL, Irrelevant Query, No Files Found
![Error Handling](docs/screenshots/error%20handling.png)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM integration | [LangChain](https://github.com/langchain-ai/langchain) + OpenAI |
| Web scraping | [Firecrawl MCP](https://docs.firecrawl.dev/mcp-server) via `langchain-mcp-adapters` |
| File parsing | `pdfplumber`, `python-docx`, `openpyxl`, `pandas` |
| Vector store | FAISS (optional) |
| Caching | SQLite (WAL mode) |
| UI | [Streamlit](https://streamlit.io/) |
| Containerization | Docker + Docker Compose |
