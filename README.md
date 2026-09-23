# AI Research Agent

A Python AI agent that researches any topic using web search and Wikipedia, returns structured results, and can save them to a file. Built with LangChain.

Based on Tech With Tim's tutorial: https://youtu.be/bTMPwUgLZf0
Original repo: https://github.com/techwithtim/PythonAIAgentFromScratch

This version adds support for free models (Google Gemini, Groq, Ollama) in addition to OpenAI and Anthropic.

## Features

- Research any topic via conversational prompt
- Web search (DuckDuckGo) + Wikipedia integration
- Structured output with Pydantic (`topic`, `summary`, `sources`, `tools_used`)
- Custom save-to-file tool (`research_output.txt`)
- Swappable LLM providers via `.env`
- Verbose agent execution for learning/debugging

## How It Works

1. User enters a research query
2. Agent picks tools (search, Wikipedia, save) based on the query
3. LLM generates a structured JSON response validated by Pydantic
4. Result is printed and optionally saved to `research_output.txt`

## Tech Stack

- Python 3.10+
- LangChain + LangChain Community + LangChain Classic
- Pydantic
- DuckDuckGo Search (`ddgs`), Wikipedia API
- Google Gemini / Groq / Ollama / OpenAI / Anthropic

## Project Structure

```
Ai_agent/
├── main.py            # Agent setup, prompt, parser, executor
├── tools.py           # search_tool, wiki_tool, save_tool
├── requirements.txt   # Dependencies
├── sample.env         # Env template (copy to .env)
├── .env               # Your API keys (not committed)
└── research_output.txt # Generated output (created at runtime)
```

## Installation

```bash
git clone <your-repo-url>
cd Ai_agent

python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Copy the template and add your key:

```bash
cp sample.env .env
```

Open `.env` and set one of these:

| Provider | Variable | Where to get key | Cost |
|----------|----------|------------------|------|
| Gemini (recommended) | `GOOGLE_API_KEY` | https://aistudio.google.com/app/apikey | Free |
| Groq | `GROQ_API_KEY` | https://console.groq.com/keys | Free |
| Ollama | — (no key needed) | Install from https://ollama.com + `ollama pull llama3.2` | Free / Local |
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys | Paid |
| Anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys | Paid |

Optional model overrides in `.env`:

```
GEMINI_MODEL="gemini-3-flash-preview"
GROQ_MODEL="llama-3.3-70b-versatile"
OLLAMA_MODEL="llama3.2"
```

## Usage

```bash
python main.py
```

Example session:

```
What can I help you research? Tell me about LangChain and save to file
```

Example structured output:

```python
topic='LangChain'
summary='LangChain is a framework for building LLM-powered applications...'
sources=['https://python.langchain.com/', 'https://en.wikipedia.org/wiki/LangChain']
tools_used=['search', 'save_text_to_file']
```

## Tools

Defined in `tools.py`:

- `search` — DuckDuckGo web search for current/general info
- `WikipediaQueryRun` — Wikipedia lookup (top 1 result, 2000 chars)
- `save_text_to_file` — Custom tool, appends research with timestamp to `research_output.txt`

## Requirements

See `requirements.txt`. Key packages:

- `langchain`, `langchain-community`, `langchain-classic`
- `langchain-google-genai`, `langchain-groq`, `langchain-ollama`, `langchain-openai`, `langchain-anthropic`
- `python-dotenv`, `pydantic`, `wikipedia`, `duckduckgo-search`, `ddgs`

## Notes

- Never commit `.env` or `venv/` to GitHub. This repo includes a `.gitignore` for that.
- Free-tier APIs have rate limits. If a model name returns 404, check the provider docs for the current model name.
- Verbose mode (`AgentExecutor(verbose=True)`) prints the agent's reasoning. Set to `False` for clean output.

## Credits

- Tutorial: Tech With Tim — Build an AI Agent From Scratch in Python
- Framework: LangChain

## License

MIT
