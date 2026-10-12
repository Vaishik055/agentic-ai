# Agentic AI Assistant for Sustainable Farming Decisions

> **Final-Year Major Project (B.Tech CSE / AI-ML)**  
> An autonomous multi-agent agronomic decision-support system that personalizes crop selection, soil nutrient balancing, irrigation planning, sustainability evaluation, and market intelligence for smallholder farmers.

---

## 🌾 Project Overview

Existing agricultural advisory systems rely on static lookups or expert manual intervention to parse soil and climatic data. Smallholder farmers lack real-time, personalized, and sustainability-aware recommendations.

This project delivers an **Agentic AI System** that:
- Ingests natural language farmer queries across regional dialects and text formats.
- Autonomously extracts soil ($N, P, K, pH$) and climate features ($temperature, humidity, rainfall$).
- Asks proactive clarifying questions when required data points are absent.
- Predicts optimal crops using trained machine learning models (99.5% accuracy across 22 crops).
- Recommends tailored chemical and bio-fertilizer dosages with split application schedules.
- Connects autonomous external data agents (OpenWeatherMap, ICAR regional soil atlas, Agmarknet mandi market prices).
- Retrieves verified agronomic best practices using FAISS vector semantic search over official ICAR and KVK extension guides.
- Evaluates cost vs. profitability, yield potential, and an environmental Sustainability Score (0–100).
- Orchestrates 8 specialized agents via **LangGraph StateGraph** into a unified, explainable reasoning report.

---

## 🏗️ System Architecture (6 Phases)

1. **Phase 1 — Farmer Interaction & Input Layer:** FastAPI endpoints + Streamlit multilingual chat & parameter forms.
2. **Phase 2 — AI Farming Assistant & Query Processing:** Natural language intent classification, entity extraction, missing feature prompts.
3. **Phase 3 — Data Gathering Agents:** Autonomous external workers:
   - `WeatherAgent`: Fetches live/forecasted agro-meteorological data via OpenWeatherMap.
   - `SoilAgent`: Retrieves regional soil profiles (type, pH, organic carbon) from ICAR Atlas.
   - `MarketAgent`: Pulls real-time APMC mandi prices and market demand trends via Agmarknet.
   - `KnowledgeAgent`: FAISS dense vector search over ICAR/KVK agronomic guidelines.
4. **Phase 4 — Planning & Recommendation:**
   - `CropRecommendationAgent`: Scikit-learn Random Forest model predicting optimal crop with confidence scoring.
   - `FertilizerAgent`: ML classifier & nutrient stoichiometry calculating NPK deficits and split dosages.
   - `RotationPlannerAgent`: 3-season crop rotation planner (Kharif -> Rabi -> Zaid) maximizing nitrogen fixation.
5. **Phase 5 — Evaluation & Prediction:**
   - `SustainabilityAgent`: 0–100 index measuring water footprint, chemical intensity, soil conservation, and carbon balance.
   - `YieldProfitAgent`: Harvest yield estimation, operational cost breakdown, revenue projection, and net ROI calculation.
6. **Phase 6 — Multi-Agent Orchestration & Reasoning:**
   - `FarmOrchestrator`: LangGraph StateGraph compiling all agents into an autonomous 5-node cyclic reasoning workflow.
   - `ReasoningAgent`: Generates transparent, farmer-friendly rationales reconciling trade-offs between profit and sustainability.

---

## 📁 Repository Structure

```
sustainable-farming-ai/
├── backend/
│   ├── agents/
│   │   ├── crop_recommendation_agent.py   # Week 3: Crop Recommendation Agent + LangChain Tool
│   │   ├── fertilizer_agent.py            # Week 4: Fertilizer Advisory Agent + LangChain Tool
│   │   ├── weather_agent.py               # Week 5: Live Weather Intelligence Agent
│   │   ├── soil_agent.py                  # Week 5: Regional Soil Profile Agent
│   │   ├── market_agent.py                # Week 5: Mandi Market APMC Price Agent
│   │   ├── knowledge_agent.py             # Week 6: Agronomic RAG Agent over ICAR/KVK manuals
│   │   ├── sustainability_agent.py        # Week 7: Sustainability Scoring Agent (0-100)
│   │   ├── yield_profit_agent.py          # Week 7: Yield & Profitability Forecaster
│   │   ├── rotation_planner_agent.py      # Week 7: Multi-season Crop Rotation Planner
│   │   ├── reasoning_agent.py             # Week 8: Explainable Reasoning Synthesis Agent
│   │   └── orchestrator.py                # Week 8: Master LangGraph Multi-Agent Orchestrator
│   ├── services/
│   │   ├── crop_service.py                # Week 2: Crop ML inference engine (Random Forest)
│   │   └── fertilizer_service.py          # Week 4: Fertilizer ML & dosage calculation engine
│   ├── models/
│   │   ├── crop_model.pkl                 # Random Forest Crop Classifier (99.5% acc)
│   │   └── fertilizer_model.pkl           # Random Forest Fertilizer Classifier (96.4% acc)
│   ├── scalers/
│   │   ├── crop_scaler.pkl
│   │   └── fertilizer_scaler.pkl
│   ├── encoders/
│   │   ├── label_encoder.pkl
│   │   └── fertilizer_encoders.pkl
│   ├── schemas/
│   │   ├── crop_schemas.py                # Pydantic schemas for Crop ML & Agent
│   │   ├── fertilizer_schemas.py          # Pydantic schemas for Fertilizer ML & Agent
│   │   ├── data_schemas.py                # Pydantic schemas for Weather, Soil, Market
│   │   ├── rag_schemas.py                 # Pydantic schemas for Knowledge RAG
│   │   ├── evaluation_schemas.py          # Pydantic schemas for Sustainability, Profit, Rotation
│   │   └── orchestrator_schemas.py        # Pydantic schemas for Master Orchestrator
│   ├── rag/
│   │   ├── document_loader.py             # Agronomic guide parser & chunker
│   │   └── vector_store.py                # FAISS vector store with normalized embeddings
│   ├── utils/
│   │   └── api_clients.py                 # External API client adapters (Weather, Soil, Mandi)
│   └── main.py                            # Production FastAPI application gateway
├── datasets/
│   ├── crop_recommendation.csv            # Kaggle 22-crop benchmark dataset
│   └── fertilizer_prediction.csv          # Agronomic soil-fertilizer training dataset
├── docs/
│   └── sample_icar_guidelines.md          # Official ICAR / KVK agronomic manuals
├── tests/
│   ├── test_crop_service.py               # Crop ML inference tests
│   ├── test_crop_agent.py                 # Crop conversational agent tests
│   ├── test_fertilizer_agent.py           # Fertilizer agent & dosage tests
│   ├── test_data_agents.py                # Weather, Soil, Market agent tests
│   ├── test_rag_agent.py                  # RAG loader, vector index, and knowledge tests
│   ├── test_evaluation_agents.py          # Sustainability, Profit, and Rotation tests
│   └── test_orchestrator.py               # LangGraph multi-agent orchestrator tests
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone and Setup
```bash
git clone -b arena/01a0ccbe-agentic-ai https://github.com/Vaishik055/agentic-ai.git
cd agentic-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```
*(Note: System includes deterministic regex NLP fallbacks so all tests and agents function even without external LLM API keys.)*

### 4. Run Automated Test Suite
```bash
pytest -v
```
All 48 unit and integration tests across services, tools, agents, vector store, and FastAPI endpoints will execute.

### 5. Launch FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## 📡 API Reference

### Health Endpoints
- `GET /` — Root health check and system information.
- `GET /health` — Detailed status checking ML models and vector store load states.

### Core Recommendations & ML
- `POST /predict/crop` — Direct ML inference from structured JSON features (`N, P, K, temperature, humidity, ph, rainfall`).
- `POST /agent/crop` — Agentic conversational endpoint accepting free-text queries.
- `POST /predict/fertilizer` — Direct ML inference calculating optimal fertilizer, NPK deficit reports, and application dosage.
- `POST /agent/fertilizer` — Agentic conversational endpoint interpreting farmer questions and outputting timing tips.

### External Intelligence & RAG
- `POST /agent/weather` — Real-time weather intelligence via OpenWeatherMap.
- `POST /agent/soil` — Soil profile intelligence via ICAR Regional Soil Atlas.
- `POST /agent/market` — APMC Mandi market price trends via Agmarknet.
- `POST /agent/knowledge` — FAISS RAG vector search over ICAR/KVK agronomic manuals.

### Evaluation & Planning
- `POST /agent/sustainability` — 0–100 sustainability index evaluation.
- `POST /agent/profitability` — Economic yield, cost, revenue, and net profit forecasting.
- `POST /agent/rotation` — Multi-season crop rotation schedule planner.

### Master Multi-Agent Orchestrator
- `POST /agent/orchestrate` — LangGraph StateGraph pipeline executing entity extraction, autonomous environmental enrichment, crop & fertilizer recommendations, farm plan evaluation, and explainable farmer reasoning.

---

## 📅 Roadmap & Milestones

| Milestone | Status | Deliverables |
| :--- | :---: | :--- |
| **Week 2** | ✅ Completed | Crop Recommendation ML model (Random Forest, 99.5% acc) & FastAPI service. |
| **Week 3** | ✅ Completed | Conversational `CropRecommendationAgent`, LangChain ML tool, clarification engine. |
| **Week 4** | ✅ Completed | Fertilizer ML classifier (96.4% acc), deficit analysis service, and `FertilizerAgent`. |
| **Week 5** | ✅ Completed | Autonomous external data agents (Weather, Soil, Market Mandi Price). |
| **Week 6** | ✅ Completed | Agricultural Knowledge RAG Agent (FAISS + normalized TF-IDF embeddings). |
| **Week 7** | ✅ Completed | Sustainability (0–100), Yield, Profit, and Crop Rotation Planners. |
| **Week 8** | ✅ Completed | LangGraph Multi-Agent Orchestrator & Explainable Reasoning Engine. |
| **Week 9** | ⏳ Planned | Streamlit Interactive Multilingual Frontend Dashboard. |
| **Week 10** | ⏳ Planned | Packaging, Dockerization, Viva Documentation & Final Demo. |
