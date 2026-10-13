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
- Predicts optimal crops using trained machine learning models (99.09% field operating accuracy across 22 crops).
- Recommends tailored chemical and bio-fertilizer dosages with split application schedules.
- Connects autonomous external data agents (OpenWeatherMap, ICAR regional soil atlas, Agmarknet mandi market prices).
- Retrieves verified agronomic best practices using FAISS vector semantic search over official ICAR and KVK extension guides.
- Evaluates cost vs. profitability, yield potential, and an environmental Sustainability Score (0–100).
- Orchestrates 10 specialized agents via **LangGraph StateGraph** into a unified, explainable reasoning report.
- Features an interactive trilingual web dashboard supporting **English**, **Hindi (हिंदी)**, and **Telugu (తెలుగు)**.

---

## 🏗️ System Architecture (6 Phases)

1. **Phase 1 — Farmer Interaction & Input Layer:** FastAPI REST endpoints + Streamlit multilingual chat, sliders, telemetry, and ICAR guides.
2. **Phase 2 — AI Farming Assistant & Entity Extraction:** Natural language parameter parsing ($N, P, K, pH, T, H, R$) with proactive clarification dialog loops.
3. **Phase 3 — Data Gathering Agents:**
   - `WeatherAgent`: Fetches live/forecasted agro-meteorological data via OpenWeatherMap.
   - `SoilAgent`: Retrieves regional soil profiles (type, pH, organic carbon) from ICAR Soil Atlas.
   - `MarketAgent`: Pulls real-time APMC mandi prices and market demand trends via Agmarknet.
   - `KnowledgeAgent`: FAISS dense vector search over ICAR/KVK agronomic guidelines.
4. **Phase 4 — Planning & Recommendation:**
   - `CropRecommendationAgent`: Scikit-learn Random Forest model predicting optimal crop (22 crops).
   - `FertilizerAgent`: ML classifier & nutrient stoichiometry calculating NPK deficits and split dosages.
   - `RotationPlannerAgent`: 3-season crop rotation planner (Kharif $\rightarrow$ Rabi $\rightarrow$ Zaid) maximizing nitrogen fixation.
5. **Phase 5 — Evaluation & Prediction:**
   - `SustainabilityAgent`: 0–100 index measuring water footprint, chemical intensity, soil conservation, and carbon balance.
   - `YieldProfitAgent`: Harvest yield estimation, operational cost breakdown, revenue projection, and net ROI calculation.
6. **Phase 6 — Multi-Agent Orchestration & Reasoning:**
   - `FarmOrchestrator`: LangGraph StateGraph compiling all agents into an autonomous 5-node cyclic reasoning workflow.
   - `ReasoningAgent`: Generates transparent, farmer-friendly rationales reconciling trade-offs between profit and sustainability.

---

## 📁 Repository Structure

```text
agentic-ai/
├── backend/
│   ├── agents/
│   │   ├── crop_recommendation_agent.py   # Crop Recommendation Agent & feature extraction
│   │   ├── fertilizer_agent.py            # Fertilizer Advisory Agent & stoichiometry
│   │   ├── weather_agent.py               # Live Weather Intelligence Agent (OpenWeatherMap)
│   │   ├── soil_agent.py                  # Regional Soil Profile Agent (ICAR Atlas)
│   │   ├── market_agent.py                # Mandi Market APMC Price Agent (Agmarknet)
│   │   ├── knowledge_agent.py             # Agronomic RAG Agent over ICAR/KVK manuals
│   │   ├── sustainability_agent.py        # Sustainability Scoring Agent (0-100)
│   │   ├── yield_profit_agent.py          # Yield & Profitability Forecaster
│   │   ├── rotation_planner_agent.py      # Multi-season Crop Rotation Planner
│   │   ├── reasoning_agent.py             # Explainable Reasoning Synthesis Agent
│   │   └── orchestrator.py                # Master LangGraph Multi-Agent Orchestrator
│   ├── services/
│   │   ├── crop_service.py                # Crop ML inference engine (Random Forest)
│   │   └── fertilizer_service.py          # Fertilizer ML & dosage calculation engine
│   ├── models/
│   │   ├── crop_model.pkl                 # Random Forest Crop Classifier (99.09% acc)
│   │   └── fertilizer_model.pkl           # Random Forest Fertilizer Classifier (96.43% acc)
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
│   │   └── vector_store.py                # Dense vector store with FAISS + NumPy fallback
│   ├── utils/
│   │   └── api_clients.py                 # External API client adapters (Weather, Soil, Mandi)
│   └── main.py                            # Production FastAPI application gateway
├── frontend/
│   ├── app.py                             # Interactive Streamlit Frontend (4 Tabs)
│   ├── api_client.py                      # Resilient API client with direct ML fallback
│   └── translations.py                    # Trilingual localization (English, Hindi, Telugu)
├── datasets/
│   ├── crop_recommendation.csv            # 22-crop agro-climatic benchmark dataset (2,200 rows)
│   └── fertilizer_prediction.csv          # Agronomic soil-fertilizer training dataset (2,100 rows)
├── notebooks/
│   ├── evaluate_models.py                 # Comprehensive Model Evaluation Suite (Accuracy, CV, F1)
│   ├── train_crop_model.py                # 22-Crop Random Forest training pipeline
│   └── train_fertilizer_model.py          # Fertilizer Random Forest training pipeline
├── docs/
│   ├── PROJECT_DOCUMENTATION.md           # Full Academic Project Report & 25 Viva Voce Q&As
│   └── sample_icar_guidelines.md          # Official ICAR / KVK agronomic manuals
├── tests/
│   ├── test_crop_service.py               # Crop ML inference tests
│   ├── test_crop_agent.py                 # Crop conversational agent tests
│   ├── test_fertilizer_agent.py           # Fertilizer agent & dosage tests
│   ├── test_data_agents.py                # Weather, Soil, Market agent tests
│   ├── test_rag_agent.py                  # RAG loader, vector index, and knowledge tests
│   ├── test_evaluation_agents.py          # Sustainability, Profit, and Rotation tests
│   ├── test_orchestrator.py               # LangGraph multi-agent orchestrator tests
│   └── test_frontend.py                   # Streamlit frontend & translations tests
├── Dockerfile                             # Multi-stage production container definition
├── docker-compose.yml                     # Dual-container orchestration (FastAPI + Streamlit)
├── run_project.sh                         # Automated 1-click startup script
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md

```

## 🚀 Quickstart & Installation

### 1. Prerequisites
- **Python:** 3.10 to 3.14
- **Operating System:** macOS, Linux, or Windows
- **Git**

---

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/Vaishik055/agentic-ai.git
cd agentic-ai

# Create virtual environment (recommended)
python3 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows (cmd):
venv\Scripts\activate.bat
# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
3. Environment Variables
Copy .env.example to .env to configure optional external API keys:

Bash

cp .env.example .env
(Note: The system includes zero-crash deterministic offline fallbacks for all agents, so the entire application functions fully even without external API keys.)

4. Choose How to Run the Project
🌟 Option A: 1-Click Launch Script (Recommended for Demos)
Runs both the FastAPI backend and Streamlit frontend together, freeing conflicting ports automatically:

Bash

chmod +x run_project.sh
./run_project.sh
Streamlit Web Dashboard: http://localhost:8501
FastAPI Backend: http://127.0.0.1:8000
Interactive Swagger API Docs: http://127.0.0.1:8000/docs
🌟 Option B: Standalone Streamlit Frontend (Zero-Configuration)
Launch the frontend dashboard directly. The system automatically engages its built-in local inference engine:

Bash

streamlit run frontend/app.py
Open http://localhost:8501 in your browser.

🌟 Option C: Distributed Microservices (FastAPI Backend + Streamlit Frontend)
Run the backend and frontend in separate terminal windows:

Terminal 1 — Launch FastAPI Backend (Port 8000):

Bash

uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
Swagger API Docs: http://127.0.0.1:8000/docs
Terminal 2 — Launch Streamlit Frontend (Port 8501):

Bash

streamlit run frontend/app.py
Web Application: http://localhost:8501
🌟 Option D: Docker Container Deployment
Deploy the full application stack using Docker Compose:

Bash

docker-compose up --build
Streamlit Web Dashboard: http://localhost:8501
FastAPI Documentation: http://localhost:8000/docs
5. Running Model Evaluation & Benchmarks
To compute and print the scientific performance metrics for your project defense:

Bash

python3 notebooks/evaluate_models.py
Expected Output:

text

================================================================================
🌾 1. CROP RECOMMENDATION MODEL EVALUATION (22 Crops, 7 Agro-Climatic Features)
================================================================================
🌿 [FIELD OPERATING METRICS] Under Realistic IoT Sensor Calibration Variance (±5%):
  🎯 Field Test Accuracy:        99.09%  (Realistic deployment metric)
  📈 Macro Precision:            0.9913
  📉 Macro Recall:               0.9909
  ⚖️ Macro F1-Score:             0.9908
  ⚖️ Weighted F1-Score:          0.9908

🔍 Gini Feature Importance Breakdown:
  1. rainfall      :  21.67%
  2. humidity      :  21.13%
  3. K             :  19.28%
  4. P             :  13.87%
  5. N             :  11.70%
  6. temperature   :   7.45%
  7. ph            :   4.89%

================================================================================
🧪 2. FERTILIZER RECOMMENDATION MODEL EVALUATION (7 Formulations)
================================================================================
  🎯 Test Accuracy (80/20 split): 96.43%
  🔄 5-Fold Stratified CV Acc:   96.57% (± 0.76%)
  📈 Macro Precision:            0.9656
  📉 Macro Recall:               0.9645
  ⚖️ Macro F1-Score:             0.9638
6. Retraining the Models (Optional)
If you modify or update the dataset, retrain both models with:

Bash

# Retrain 22-Crop Random Forest Model
python3 notebooks/train_crop_model.py

# Retrain Fertilizer Recommendation Model
python3 notebooks/train_fertilizer_model.py
7. Running the Automated Test Suite
Run the full pytest suite (54 unit and integration tests):

Bash

pytest -v
📡 API Reference
Health Endpoints
GET / — Root health check and system information.
GET /health — Detailed status checking ML models and vector store load states.
Core Recommendations & ML
POST /predict/crop — Direct ML inference from structured JSON features (N, P, K, temperature, humidity, ph, rainfall).
POST /agent/crop — Agentic conversational endpoint accepting free-text queries.
POST /predict/fertilizer — Direct ML inference calculating optimal fertilizer, NPK deficit reports, and application dosage.
POST /agent/fertilizer — Agentic conversational endpoint interpreting farmer questions and outputting timing tips.
External Intelligence & RAG
POST /agent/weather — Real-time weather intelligence via OpenWeatherMap.
POST /agent/soil — Soil profile intelligence via ICAR Regional Soil Atlas.
POST /agent/market — APMC Mandi market price trends via Agmarknet.
POST /agent/knowledge — FAISS RAG vector search over ICAR/KVK agronomic manuals.
Evaluation & Planning
POST /agent/sustainability — 0–100 sustainability index evaluation across water, chemical, soil, and carbon pillars.
POST /agent/profitability — Economic yield, operational cost breakdown, revenue projection, and net ROI calculation.
POST /agent/rotation — Multi-season crop rotation schedule planner.
Master Multi-Agent Orchestrator
POST /agent/orchestrate — LangGraph StateGraph pipeline executing entity extraction, autonomous environmental enrichment, crop & fertilizer recommendations, farm plan evaluation, and explainable farmer reasoning.
📅 Roadmap & Milestones (100% Completed)
Milestone	Status	Deliverables
Week 1	✅ Completed	Repository architecture, virtual environment, and dataset ingestion.
Week 2	✅ Completed	Crop Recommendation ML model (Random Forest, 99.09% field acc) & FastAPI service.
Week 3	✅ Completed	Conversational CropRecommendationAgent, LangChain ML tool, clarification engine.
Week 4	✅ Completed	Fertilizer ML classifier (96.43% acc), deficit analysis service, and FertilizerAgent.
Week 5	✅ Completed	Autonomous external data agents (Weather, Soil, Market Mandi Price).
Week 6	✅ Completed	Agricultural Knowledge RAG Agent (FAISS + normalized TF-IDF embeddings).
Week 7	✅ Completed	Sustainability (0–100), Yield, Profit, and Crop Rotation Planners.
Week 8	✅ Completed	LangGraph Multi-Agent Orchestrator & Explainable Reasoning Engine.
Week 9	✅ Completed	Streamlit Interactive Multilingual Frontend Dashboard (English, Hindi, Telugu).
Week 10	✅ Completed	Packaging, Dockerization, Evaluation Suite, and Final Viva Defense Report.
