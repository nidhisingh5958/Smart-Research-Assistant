# 🧠 Smart Research Assistant (Agentic AI)

An autonomous AI tool that searches, summarizes, and generates structured reports from research papers. Just give it a goal — the agent does the rest.

## 🚀 Features
- Goal-based research planning
- Auto paper search (arXiv, Semantic Scholar)
- Summarization via GPT-4 / LLMs
- Report generation with citations
- Memory for tracking processed data

## 🏗 Tech Stack
- Python + LangChain / CrewAI
- OpenAI / Hugging Face LLMs
- FAISS / Chroma for memory
- arXiv & Semantic Scholar APIs
- Optional: Streamlit / React / Electron UI

## ⚙️ Setup

```bash
git clone https://github.com/nidhisingh5958/Smart-Research-Assistant.git
cd Smart-Research-Assistant
pip install -r requirements.txt

## 📂 Project Structure

```bash
smart-research-assistant/
├── agent/
│   ├── planner.py          # Converts goal into sub-tasks
│   ├── tools.py            # Search + summarization tools
│   ├── memory.py           # Vector database (FAISS/Chroma)
│   └── agent.py            # Main agent loop
├── ui/
│   ├── streamlit_app.py    # (or Electron/React frontend)
├── reports/
│   └── output_*.md         # Generated reports
├── requirements.txt
└── README.md
