# Parallel World Social Simulator

**現実を食べて、未来を走る並行世界**

現実世界のニュースをリアルタイムに取り込みながら、シミュレーション内の時間を加速させ、
常に数ヶ月先の「ありえた未来」を生成し続けるソーシャルネットワークシミュレーター。

## Concept

```
現実:          ───●──────────────────→  2026/03 (now)
シミュレーション: ───●━━━━━━━━━━━━━━━━●→  2026/09 (6ヶ月先)
                   ↑ニュース投入で分岐
                   ├──→ Fork A
                   └──→ Fork B
```

- **100〜200体のAIエージェント**が架空の市民として社会課題を議論
- **現実のニュースが分岐点**となり、並行世界が枝分かれする
- **現実が追いついたとき**、予測との差分（Reality Diff）が可視化される

## Architecture

```
News Sources → Topic Extraction → Agent Network → Timeline Branches → Visualization
(RSS/API)       (LLM)             (LLM Brain)     (Fork Manager)     (Web UI)
```

See [research/architecture-design.md](research/architecture-design.md) for full design.

## Project Structure

```
src/
├── core/           # Time warp engine, simulation loop
├── agents/         # Agent definitions, personality, memory, bias
├── network/        # Social graph, recommendation, echo chambers
├── news/           # News ingestion, topic extraction (Reality Anchor)
├── timeline/       # Timeline branching, fork management, Reality Diff
└── visualization/  # Web frontend, real-time dashboards
```

## Tech Stack

- **Simulation**: Python 3.11+, NetworkX
- **LLM**: Claude API (Sonnet)
- **Server**: FastAPI + WebSocket
- **Frontend**: Next.js + D3.js
- **Data**: SQLite → PostgreSQL

## Status

🔬 Research & Design Phase
