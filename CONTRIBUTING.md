# Contributing to FinSights

Thanks for your interest in contributing to FinSights.

FinSights is an open-source AI-powered financial document analysis platform built with a FastAPI backend and a React frontend. It provides intelligent section-based summarization and RAG-powered chat capabilities for financial documents. We welcome improvements across the codebase, documentation, bug reports, design feedback, and feature enhancements.

Before you start, read the relevant section below. It helps keep contributions focused, reviewable, and aligned with the current project setup.

---

## Quick Setup Checklist

Before you dive in, make sure you have these installed:

```bash
# Check Python (3.11+ recommended)
python --version

# Check Node.js (18+ recommended)
node --version

# Check npm
npm --version

# Check Docker
docker --version
docker compose version

# Check Git
git --version
```

New to contributing?

1. Open an issue or pick an existing one to work on.
2. Sync your branch from `finsights/dev`.
3. Follow the local setup guide below.
4. Run the app locally and verify your change before opening a PR.

## Table of Contents

- [How do I...?](#how-do-i)
  - [Get help or ask a question?](#get-help-or-ask-a-question)
  - [Report a bug?](#report-a-bug)
  - [Suggest a new feature?](#suggest-a-new-feature)
  - [Set up FinSights locally?](#set-up-finsights-locally)
  - [Start contributing code?](#start-contributing-code)
  - [Improve the documentation?](#improve-the-documentation)
  - [Submit a pull request?](#submit-a-pull-request)
- [Code guidelines](#code-guidelines)
- [Pull request checklist](#pull-request-checklist)
- [Branching model](#branching-model)
- [Thank you](#thank-you)

---

## How do I...

### Get help or ask a question?

- Start with the main project docs in [`README.md`](./README.md), [`backend/.env.example`](./backend/.env.example), and the inline documentation in service files.
- If something is unclear, open a GitHub issue with your question and the context you already checked.

### Report a bug?

1. Search existing issues first.
2. If the bug is new, open a GitHub issue.
3. Include your environment, what happened, what you expected, and exact steps to reproduce.
4. Add screenshots, logs, request details, or response payloads if relevant.

### Suggest a new feature?

1. Open a GitHub issue describing the feature.
2. Explain the problem, who it helps, and how it fits FinSights.
3. If the change is large, get alignment in the issue before writing code.

### Set up FinSights locally?

#### Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- Git
- Docker with Docker Compose v2
- One LLM provider:
  - An OpenAI-compatible API key (OpenAI, Groq, OpenRouter), or
  - Enterprise inference endpoint (GenAI Gateway, APISIX), or
  - Ollama installed locally on the host machine

#### Option 1: Local Development

##### Step 1: Clone the repository

```bash
git clone https://github.com/cld2labs/FinSights.git
cd FinSights
```

##### Step 2: Configure environment variables

Create an `.env` file in the `backend/` directory from the example:

```bash
cp backend/.env.example backend/.env
```

At minimum, configure your LLM provider. Example for OpenAI:

```env
API_ENDPOINT=
API_TOKEN=sk-...
MODEL_NAME=gpt-4o-mini
PROVIDER_NAME=OpenAI
```

Example for local Ollama:

```env
API_ENDPOINT=http://localhost:11434
API_TOKEN=not-needed
MODEL_NAME=llama3.2:3b
PROVIDER_NAME=Ollama
```

##### Step 3: Install backend dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

##### Step 4: Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

##### Step 5: Start the backend

```bash
cd backend
source .venv/bin/activate
python server.py
```

The backend runs at `http://localhost:8000`.

##### Step 6: Start the frontend

Open a second terminal:

```bash
cd frontend
npm run dev
```

The Vite dev server runs at `http://localhost:5173`.

##### Step 7: Access the application

- Frontend: `http://localhost:5173`
- Backend health check: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`

#### Option 2: Docker

From the repository root:

```bash
cp backend/.env.example backend/.env
# Edit backend/.env with your LLM provider settings
docker compose up --build
```

This starts:

- Frontend on `http://localhost:5173`
- Backend on `http://localhost:8000`

#### Common Troubleshooting

- If ports `5173` or `8000` are already in use, stop the conflicting process before starting FinSights.
- If LLM requests fail, confirm your `.env` values are correct for the selected provider.
- If you use Ollama with Docker, make sure Ollama is running on the host and reachable from the container.
- If Docker fails to build, rebuild with `docker compose up --build`.
- If Python packages fail to install, confirm you are using a supported Python version (3.11+).

### Start contributing code?

1. Open or choose an issue.
2. Create a feature branch from `finsights/dev`.
3. Keep the change focused on a single problem.
4. Run the app locally and verify the affected workflow.
5. Update docs when behavior, setup, configuration, or architecture changes.
6. Open a pull request back from your feature branch into `finsights/dev`.

### Improve the documentation?

Documentation updates are welcome. Relevant files currently live in:

- [`README.md`](./README.md)
- [`backend/.env.example`](./backend/.env.example)
- Service docstrings in [`backend/services/`](./backend/services/)
- [`backend/config.py`](./backend/config.py)

### Submit a pull request?

Follow the checklist below before opening your PR. Your pull request should:

- Stay focused on one issue or topic.
- Explain what changed and why.
- Include manual verification steps.
- Include screenshots or short recordings for UI changes.
- Reference the related GitHub issue when applicable.

Note: pull requests should target the `finsights/dev` branch.

---

## Code guidelines

- Follow the existing project structure and patterns before introducing new abstractions.
- Keep frontend changes consistent with the React + Vite + Tailwind setup already in use.
- Keep backend changes consistent with the FastAPI service structure in [`backend`](./backend).
- Avoid unrelated refactors in the same pull request.
- Do not commit secrets, API keys, local `.env` files, or generated artifacts.
- Prefer clear, small commits and descriptive pull request summaries.
- Update documentation when contributor setup, behavior, environment variables, or service logic changes.

---

## Pull request checklist

Before submitting your pull request, confirm the following:

- You tested the affected flow locally.
- The application still starts successfully in the environment you changed.
- You removed debug code, stray logs, and commented-out experiments.
- You documented any new setup steps, environment variables, or behavior changes.
- You kept the pull request scoped to one issue or topic.
- You added screenshots for UI changes when relevant.
- You did not commit secrets, API keys, or cached documents.
- You are opening the pull request against `finsights/dev`.

If one or more of these are missing, the pull request may be sent back for changes before review.

---

## Branching model

- Base new work from `finsights/dev`.
- Open pull requests against `finsights/dev`.
- Use descriptive branch names such as `fix/rag-chat-validation` or `docs/update-contributing-guide`.
- Rebase or merge the latest `finsights/dev` before opening your PR if your branch has drifted.

---

## Thank you

Thanks for contributing to FinSights. Whether you're fixing a bug, improving the docs, adding a new feature, or refining the RAG workflow, your work helps make the project more useful and easier to maintain.
