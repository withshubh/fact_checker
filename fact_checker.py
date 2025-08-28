import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
memory = InMemorySaver()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

llm = init_chat_model("google_genai:gemini-2.0-flash")
tavily = TavilySearch(max_results=3)

# ---------------------------
# State definition
# ---------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    claim: str
    evidence: str
    verdict: str
    sources: list

# ---------------------------
# Graph setup
# ---------------------------
graph_builder = StateGraph(State)

def claim_node(state: State):
    """Capture the claim from the user."""
    claim = state["messages"][-1].content
    return {"claim": claim}

graph_builder.add_node("claim", claim_node)

def search_node(state: State):
    """Always search Tavily for the claim."""
    claim = state["claim"]
    results = tavily.invoke({"query": claim})
    
    # Collect evidence text and top 3 sources
    evidence = "\n".join([r["content"] for r in results["results"] if r.get("content")])
    sources = [{"title": r["title"], "url": r["url"]} for r in results["results"][:3]]

    return {"evidence": evidence, "sources": sources}

graph_builder.add_node("search", search_node)

def verdict_node(state: State):
    """LLM compares claim against Tavily evidence and gives final verdict."""
    claim = state["claim"]
    evidence = state["evidence"]
    sources = state.get("sources", [])

    prompt = [
        {"role": "system", "content": "You are a fact-checking assistant. Always use the provided evidence to judge the claim. Respond with a verdict: TRUE, FALSE, or PARTIALLY TRUE, followed by a detailed explanation citing the evidence."},
        {"role": "user", "content": f"Claim: {claim}\n\nEvidence:\n{evidence}"}
    ]

    verdict = llm.invoke(prompt).content
    return {
        "verdict": verdict,
        "sources": sources,
        "messages": [{"role": "assistant", "content": verdict}],
    }

graph_builder.add_node("verdict", verdict_node)

# ---------------------------
# Graph edges
# ---------------------------
graph_builder.add_edge(START, "claim")
graph_builder.add_edge("claim", "search")
graph_builder.add_edge("search", "verdict")
graph_builder.add_edge("verdict", END)

graph = graph_builder.compile(checkpointer=memory)

# ---------------------------
# Interactive Runner
# ---------------------------
config = {"configurable": {"thread_id": "factcheck-1"}}

def run_fact_checker():
    while True:
        user_input = input("\nEnter a claim to fact-check (or 'quit'): ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Run the graph fully, not step-by-step
        events = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config)

        verdict = events.get("verdict")
        sources = events.get("sources", [])

        if verdict:
            print("\n✅ Fact-Check Result:\n", verdict)
            if sources:
                print("\n🔗 Sources:")
                for s in sources:
                    print(f"- {s['title']}: {s['url']}")
        else:
            print("\n⚠️ No verdict was generated. Check your graph logic.")

if __name__ == "__main__":
    run_fact_checker()
