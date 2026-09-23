import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


def get_llm():
    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        print("Using Google Gemini (free tier): gemini-3-flash-preview")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
            google_api_key=google_key,
        )
    if groq_key:
        from langchain_groq import ChatGroq

        print("Using Groq (free tier): qwen/qwen3.8-27b")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            groq_api_key=groq_key,
        )
    if openai_key:
        from langchain_openai import ChatOpenAI

        print("Using OpenAI: gpt-4o-mini")
        return ChatOpenAI(model="gpt-4o-mini")
    if anthropic_key:
        from langchain_anthropic import ChatAnthropic

        print("Using Anthropic (original video): claude-3-5-sonnet-20241022")
        return ChatAnthropic(model="claude-3-5-sonnet-20241022")

    try:
        from langchain_ollama import ChatOllama

        print("No API key found. Trying Ollama locally (free, no key)...")
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"))
    except Exception:
        pass

    raise ValueError(
        "No LLM configured! Add GOOGLE_API_KEY (free: https://aistudio.google.com/app/apikey) "
        "or GROQ_API_KEY to your .env file."
    )


llm = get_llm()
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=8)

if __name__ == "__main__":
    while True:
        try:
            query = input("\nWhat can I help you research? (type exit to quit) ")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        if query.strip().lower() in ("exit", "quit", "q", "no", "n"):
            print("Goodbye!")
            break
        if not query.strip():
            continue
        try:
            raw_response = agent_executor.invoke({"query": query})
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                print("\nGoogle free quota exhausted for today. Try again tomorrow.")
            else:
                print("Agent error:", e)
            print("\nDone with your output. Do you want any other research? (type exit to quit)")
            continue
        output = raw_response.get("output", "")
        if isinstance(output, list):
            output_text = output[0].get("text", "") if isinstance(output[0], dict) else str(output[0])
        else:
            output_text = output
        try:
            structured_response = parser.parse(output_text)
            print("\n=== Structured Response ===")
            print(structured_response)
        except Exception as e:
            print("Error parsing response:", e)
            print("Raw Response:", output_text)
        print("\nDone with your output. Do you want any other research? (type exit to quit)")
