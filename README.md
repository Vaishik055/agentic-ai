# Agentic AI Assistant for Sustainable Farming Decisions

> **Final-Year Major Project (B.Tech CSE / AI-ML)**  
> An autonomous multi-agent agronomic decision-support system that personalizes crop selection, soil nutrient balancing, irrigation planning, and sustainability evaluation for smallholder farmers.

---

## 🌾 Project Overview

Existing agricultural advisory systems rely on static lookups or expert manual intervention to parse soil and climatic data. Smallholder farmers lack real-time, personalized, and sustainability-aware recommendations.

This project delivers an **Agentic AI System** that:
- Ingests natural language farmer queries across regional dialects and text formats.
- Autonomously extracts soil ($N, P, K, pH$) and climate features ($temperature, humidity, rainfall$).
- Asks proactive clarifying questions when required data points are absent.
- Predicts optimal crops using trained machine learning models (99.5% accuracy).
- Recommends tailored chemical and bio-fertilizer dosages with split application schedules.
- Connects autonomous external data agents (Weather, Soil, Market Mandi prices) and RAG over ICAR/KVK guides.
- Evaluates cost vs. profitability, yield potential, and an environmental Sustainability Score (0–100).

---

## 🏗️ System Architecture (6 Phases)

1. **Phase 1 — Farmer Interaction & Input Layer:** FastAPI endpoints + Streamlit multilingual chat & parameter forms.
2. **Phase 2 — AI Farming Assistant & Query Processing:** Natural language intent classification, entity extraction, missing feature prompts.
3. **Phase 3 — Data Gathering Agents:** Autonomous external workers (Weather, Soil, Market Mandi Price, ICAR Knowledge RAG).
4. **Phase 4 — Planning & Recommendation:** Crop Recommendation ML Agent + Fertilizer ML Agent + Irrigation / Crop Rotation Planner.
5. **Phase 5 — Evaluation & Prediction:** Yield forecasting, cost/profit ROI estimation, and 0–100 Sustainability Index.
6. **Phase 6 — Multi-Agent Orchestration & Reasoning:** LangGraph supervisor synthesizing multi-agent outputs into transparent, explainable advice.

---

## 📁 Repository Structure

```
sustainable-farming-ai/
├── backend/
│   ├── agents/
│   │   ├── crop_recommendation_agent.py   # Week 3: Crop Recommendation Agent + LangChain Tool
│   │   ├── fertilizer_agent.py            # Week 4: Fertilizer Advisory Agent + LangChain Tool
│   │   ├── weather_agent.py               # (Week 5)
│   │   ├── soil_agent.py                  # (Week 5)
│   │   ├── market_agent.py                # (Week 5)
│   │   └── orchestrator.py                # (Week 8)
│   ├── services/
│   │   ├── crop_service.py                # Week 2: Scikit-learn Crop inference engine
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
│   │   └── fertilizer_schemas.py          # Pydantic schemas for Fertilizer ML & Agent
│   ├── utils/
│   ├── rag/
│   └── main.py                            # Production FastAPI application gateway
├── datasets/
│   ├── crop_recommendation.csv            # Kaggle 22-crop benchmark dataset
│   └── fertilizer_prediction.csv          # Agronomic soil-fertilizer training dataset
├── notebooks/
│   └── train_fertilizer_model.py          # Fertilizer ML training and evaluation pipeline
├── tests/
│   ├── test_crop_service.py               # ML service validation tests
│   ├── test_crop_agent.py                 # Crop Agent conversational tests
│   └── test_fertilizer_agent.py           # Fertilizer Agent & service tests
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
All 18 unit and integration tests across services, tools, agents, and FastAPI endpoints will execute.

### 5. Launch FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## 📡 API Reference

### Health Endpoints
- `GET /` — Root health check and system information.
- `GET /health` — Detailed status checking ML model load states.

### Crop Recommendation
- `POST /predict/crop` — Direct ML inference from structured JSON features (`N, P, K, temperature, humidity, ph, rainfall`).
- `POST /agent/crop` — Agentic conversational endpoint accepting free-text queries, identifying missing parameters, and calling `predict_crop_tool`.

### Fertilizer Recommendation
- `POST /predict/fertilizer` — Direct ML inference calculating optimal fertilizer, NPK deficit reports, and application dosage.
- `POST /agent/fertilizer` — Agentic conversational endpoint interpreting farmer questions, calculating soil deficits, and outputting sustainable timing tips.

---

## 📅 Roadmap & Milestones

| Milestone | Status | Deliverables |
| :--- | :---: | :--- |
| **Week 2** | ✅ Completed | Crop Recommendation ML model (Random Forest, 99.5% acc) & FastAPI service. |
| **Week 3** | ✅ Completed | Conversational `CropRecommendationAgent`, LangChain ML tool, clarification engine. |
| **Week 4** | ✅ Completed | Fertilizer ML classifier (96.4% acc), deficit analysis service, and `FertilizerAgent`. |
| **Week 5** | 🔄 Next | Autonomous external data agents (Weather, Soil, Market Mandi Price). |
| **Week 6** | ⏳ Planned | Agricultural Knowledge RAG Agent (FAISS + sentence-transformers). |
| **Week 7** | ⏳ Planned | Sustainability, Yield, Profit, and Crop Rotation Planners. |
| **Week 8** | ⏳ Planned | LangGraph Multi-Agent Orchestrator & Reasoning Engine. |
| **Week 9** | ⏳ Planned | Streamlit Interactive Multilingual Frontend Dashboard. |
| **Week 10** | ⏳ Planned | Packaging, Dockerization, Viva Documentation & Final Demo. |
