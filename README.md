# Parallel World Social Simulator

**現実を食べて、未来を走る並行世界**

現実世界のニュースをリアルタイムに取り込みながら、シミュレーション内の時間を加速させ、
常に数ヶ月先の「ありえた未来」を生成し続けるソーシャルネットワークシミュレーター。

## Concept

```
現実:          ───●──────────────────→  2026/03 (now)
シミュレーション: ───●━━━━━━━━━━━━━━━━●→  2026/09 (6ヶ月先)
                   ↑ニュース投入で分岐
                   ├──→ Fork A (もしAI法案が可決されなかったら)
                   └──→ Fork B (もし気候変動法案も同時に可決されたら)
```

- **50〜200体のAIエージェント**が架空の市民として社会課題を議論
- **現実のニュースが分岐点**となり、並行世界が枝分かれする
- **現実が追いついたとき**、予測との差分（Reality Diff）が可視化される

## Quick Start

```bash
# CLI mode (rule-based, fast)
python -m src run --agents 50 --steps 14
python -m src run --agents 200 --steps 30 --no-live --speed 0

# With news injection
python -m src inject "AI規制法案が国会で可決" --sentiment 0.5

# API server mode
python -m src.visualization.server
# Then: http://localhost:8000/docs
```

## API Endpoints

```
POST /api/simulation/start     Initialize simulation
POST /api/simulation/step      Advance one step
POST /api/simulation/run       Run N steps
POST /api/news/inject          Inject news (+ optional fork)
GET  /api/agents               List all agents
GET  /api/agents/{id}          Agent detail
GET  /api/opinions             Opinion distribution
GET  /api/timelines            Parallel worlds
GET  /api/stats                Step-by-step statistics
WS   /ws/live                  Real-time updates
```

## Architecture

```
News Sources → Topic Extraction → Agent Network → Timeline Branches → Visualization
(RSS/API)       (LLM/keyword)     (LLM + Rules)   (Fork Manager)     (CLI/API/Web)
```

See [research/architecture-design.md](research/architecture-design.md) for full design.

## Project Structure

```
src/
├── core/
│   ├── config.py          # SimulationConfig, LLMConfig
│   ├── models.py          # SimTime, SimEvent, Topic
│   ├── engine.py          # SimulationEngine (hybrid LLM/rules)
│   ├── time_manager.py    # Time acceleration
│   ├── database.py        # SQLite persistence
│   └── llm_client.py      # Anthropic API wrapper
├── agents/
│   ├── models.py          # AgentProfile (Big Five, biases), AgentState
│   ├── factory.py         # Population generation
│   ├── behavior.py        # Rule-based bounded confidence model
│   ├── llm_brain.py       # Claude-powered agent cognition
│   ├── prompts.py         # LLM prompt templates
│   └── memory.py          # Agent memory management
├── network/
│   ├── models.py          # Post, Relationship
│   ├── graph.py           # Watts-Strogatz small-world network
│   └── dynamics.py        # Echo chamber detection, polarization metrics
├── news/
│   ├── models.py          # NewsItem, TopicSeed
│   ├── ingestion.py       # RSS feed + manual injection
│   └── scheduler.py       # Automated news polling
├── timeline/
│   ├── models.py          # Timeline, ForkPoint, Snapshot
│   ├── manager.py         # Snapshot/fork/comparison management
│   ├── fork.py            # ForkRunner, ParallelWorldEngine
│   └── reality_diff.py    # Prediction vs reality comparison
└── visualization/
    ├── cli.py             # Rich CLI dashboard
    ├── api.py             # FastAPI REST + WebSocket (19 endpoints)
    ├── server.py          # Uvicorn startup
    └── static/index.html  # D3.js web dashboard
```

## Key Features

| Feature | Status |
|---------|--------|
| Agent population (Big Five, biases, values) | ✅ |
| Small-world social network | ✅ |
| Rule-based bounded confidence dynamics | ✅ |
| LLM-powered agent cognition (Claude) | ✅ |
| Hybrid mode (LLM + rules) | ✅ |
| Time acceleration engine | ✅ |
| Multi-topic support | ✅ |
| News injection (manual + RSS) | ✅ |
| Timeline forking (parallel worlds) | ✅ |
| Reality Diff (prediction accuracy) | ✅ |
| Rich CLI dashboard | ✅ |
| FastAPI + WebSocket API (19 endpoints) | ✅ |
| D3.js web dashboard (network graph, real-time) | ✅ |
| Automated news scheduling (RSS polling) | ✅ |
| Echo chamber detection + polarization metrics | ✅ |
| Network evolution (homophily rewiring) | ✅ |
| Advanced memory (reflections, weighted retrieval) | ✅ |

## Tech Stack

- **Simulation**: Python 3.11+, NetworkX
- **LLM**: Claude API (Sonnet) — hybrid mode with rule-based fallback
- **Server**: FastAPI + WebSocket
- **Data**: SQLite (WAL mode)
- **CLI**: Rich
