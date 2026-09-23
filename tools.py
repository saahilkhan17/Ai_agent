from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime


def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()


def safe_search(query: str) -> str:
    for attempt in range(3):
        try:
            return search.run(query)
        except Exception as e:
            if attempt == 2:
                return f"Web search unavailable right now ({e}). Use your own knowledge instead."
            import time as _time

            _time.sleep(2)


search_tool = Tool(
    name="search",
    func=safe_search,
    description="Search the web for information. Useful for current events, facts, and general research.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
_wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


def safe_wiki(query: str) -> str:
    try:
        return _wiki.run(query)
    except Exception as e:
        return f"Wikipedia unavailable right now ({e}). Use web search results instead."


wiki_tool = Tool(
    name="wikipedia",
    func=safe_wiki,
    description="Look up facts on Wikipedia. If unavailable, use web search instead.",
)
