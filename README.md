# Fact Checker

A command-line fact-checking assistant that uses LLMs and web search to verify user claims. Built with LangGraph, and Tavily Search.

## Features

- Accepts user claims interactively via the terminal
- Searches the web for evidence using Tavily
- Uses a Google Gemini LLM to compare the claim against evidence
- Returns a verdict: TRUE, FALSE, or PARTIALLY TRUE, with detailed explanation and sources

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up API keys**
   - Create a `.env` file in the project root with the following:
     ```env
     GOOGLE_API_KEY="<your-google-api-key>"
     TAVILY_API_KEY="<your-tavily-api-key>"
     ```

## Usage

Run the fact checker from the terminal:

```bash
python3 fact_checker.py
```

Enter a claim when prompted. The assistant will search for evidence and return a verdict with sources.

Type `quit` or `exit` to stop.

## Requirements

- Python 3.8+
- [langchain](https://python.langchain.com/)
- [langgraph](https://github.com/langchain-ai/langgraph)
- [langchain-tavily](https://github.com/tavily/langchain-tavily)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

## License

MIT
