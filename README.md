# D2C AI Support Agent

A customer-support AI agent prototype for a small direct-to-consumer (D2C) online store. The agent uses the Google Gemini API with **function calling (tool use)** to look up orders, search products and find store policies from CSV data, and a small **evaluation suite** measures how reliable its answers are.

> **Note:** This is a learning/portfolio prototype. All data in the CSV files is fictional sample data, and the app is not connected to a real store.

## Features

- **Tool calling:** Gemini decides when to call three tools: `lookup_order`, `search_products` and `get_store_policy`. Facts always come from the data, not from the model's memory.
- **Order tracking:** ask about an order ID (e.g. `ORD-1001`) and get the product, status and estimated delivery date. Only safe fields are exposed to the model.
- **Product help:** natural-language requests such as "I need something to listen to music while running" return matching products with price and stock status.
- **Store policies:** returns, shipping time, payment and cancellation answers come from `faqs.csv`.
- **Guardrails:** the agent must not invent orders, prices or policies. It says it doesn't know, or asks for the order ID, when the data doesn't answer the question.
- **Multi-language and multi-turn:** replies in the customer's language (English or Bangla) and remembers earlier messages, so "When will it arrive?" works after an order ID was given.
- **Transparency in the UI:** every answer shows which tool was called, the response time and the model used.
- **Reliable fallback:** if the AI is unavailable (rate limit, server overload, no key), the app retries, tries a backup model, and finally falls back to simple rule-based CSV answers.
- **Evals:** `evals.py` runs test questions against the real agent and reports accuracy, tool-use accuracy and response time.

## Tech stack

- Python
- Streamlit (web UI)
- pandas (CSV data handling)
- Google Gemini API via `google-genai` (free tier is enough for the demo)
- python-dotenv (loading the API key from a `.env` file)

## How it works

1. The customer types a question in the chat box (`app.py`).
2. `agent.py` sends the conversation to Gemini together with three tool definitions.
3. Gemini decides which tool to call (for example `lookup_order("ORD-1001")`). The tool reads the CSV data and returns facts.
4. Gemini writes a short, polite reply using only those facts.
5. If the AI call fails, the agent retries, switches to a backup model, and finally uses rule-based answers.

## Project structure

```
d2c-ai-support-agent/
├── app.py              # Streamlit web UI
├── agent.py            # Data loading, tools, Gemini call with retries, fallback logic
├── evals.py            # Evaluation script (run it to measure answer quality)
├── eval_results.csv    # Output of the latest evaluation run
├── orders.csv          # Sample orders (order_id, product_name, status, estimated_delivery)
├── products.csv        # Sample products (name, category, description, price_bdt, in_stock)
├── faqs.csv            # Sample FAQs (keywords, answer)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for your API key file
├── .gitignore          # Keeps .env and .venv out of Git
└── docs/screenshot.png # Screenshot used in this README
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/shafiul414/d2c-ai-support-agent.git
cd d2c-ai-support-agent
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

python -m pip install -r requirements.txt
```

### 3. Add your Gemini API key

1. Create a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Copy `.env.example` to a new file named `.env` and put your key in it:

```
GEMINI_API_KEY=your_key_here
```

Never commit your `.env` file or share your key.

### 4. Run the app

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser. A green "AI mode is on" box means Gemini is connected. Without a key the app still works in basic mode using only the CSV files.

## Evaluation

Run the test suite (needs a working API key):

```bash
python evals.py          # all cases
python evals.py --quick  # first 6 cases only
```

The script builds test cases from your own CSV files (orders, product prices and stock) plus fixed cases, and checks each answer for the correct facts, forbidden content and correct tool use. Cases include:

- order lookups, product stock and price questions
- return policy and shipping time
- a natural-language product request with no product keyword
- an unknown order ID (the agent must not invent a status)
- a missing order ID (the agent must ask for it, not guess)
- a product that is not in the catalogue
- a Bangla question and a follow-up question in the same conversation

Results are saved to `eval_results.csv` (answer, tools used, time and failure reasons for every case).

**Latest results (17 test cases):** 17/17 passed, correct tool use 16/16, average response time 2.6s (model: gemini-3.5-flash-lite, free tier). The suite is small and written by me, so it is a regression check, not proof of production readiness. Full answers for every case are in `eval_results.csv`.

The free tier has rate limits, so the script pauses between cases and retries when the AI is temporarily unavailable. Cases that could not be graded are reported as skipped, not counted as failures.

## Configuration (optional)

Add these to your `.env` file to change behaviour:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.8-flash` | Main model to use |
| `GEMINI_BACKUP_MODEL` | `gemini-3.7-flash` | Model to try if the main one is overloaded |
| `GEMINI_THINKING_LEVEL` | `low` | Lower is faster; set to `medium` for more careful reasoning |

Model names change over time. If a model name is rejected, check the model list in Google AI Studio.

## Example questions

- Where is my order ORD-1001?
- What is the status of ORD-1002?
- What is your return policy?
- How long does shipping take?
- Recommend headphones
- Do you have a laptop stand?
- I need something to listen to music while running
- Do you sell shoes? (not in the data, so the agent should say it doesn't know)
- আমার ORD-1002 অর্ডারটা কবে আসবে?

## Screenshot

     ![D2C AI Support Agent](images/screenshot.png)
## Limitations and ideas for improvement

- Uses small sample CSV files; a real store would need a database or store API (for example Shopify).
- Order lookup is by order ID only, with no customer login or verification.
- Product search is keyword-based. For a large catalogue, retrieval with embeddings (RAG) would work better.
- No WhatsApp or helpdesk integration yet.
- The free Gemini tier has rate limits, and free-tier requests may be used by Google to improve its products, so do not send real customer data to it.
- Possible next steps: RAG over a larger catalogue, customer verification, cost and latency logging, more eval cases, and deployment to Streamlit Community Cloud.

## Author

**Md. Shafiul Mursalin**
GitHub: [shafiul414](https://github.com/shafiul414) · LinkedIn: [shafiul-mursalin](https://www.linkedin.com/in/shafiul-mursalin-0620251b9)
