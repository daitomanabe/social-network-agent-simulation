"""Pre-built simulation scenarios for various social issues.

Each scenario defines:
- Topic and initial conditions
- A sequence of timed news events
- Expected discussion dynamics
- Agent count and configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScheduledNews:
    """A news event scheduled for a specific step."""

    step: int
    headline: str
    summary: str
    sentiment: float = 0.0
    create_fork: bool = True


@dataclass
class Scenario:
    """A complete simulation scenario."""

    id: str
    name: str
    description: str
    topics: list[str]
    agent_count: int = 100
    total_steps: int = 42  # 6 weeks
    seed: int = 2026
    news_events: list[ScheduledNews] = field(default_factory=list)

    # Narrative acts (step ranges for display)
    acts: list[dict] = field(default_factory=list)


# ════════════════════════════════════════════════════
# Scenario: AI Regulation (AI規制)
# ════════════════════════════════════════════════════

AI_REGULATION = Scenario(
    id="ai_regulation",
    name="AI規制法案をめぐる分断",
    description=(
        "AI規制法案が国会に提出され、テック業界と市民の間で激しい議論が展開される。"
        "法案可決後、企業の集団訴訟と市民の支持が対立し、社会が分極化していく。"
    ),
    topics=["AI regulation"],
    agent_count=100,
    total_steps=42,
    seed=2026,
    news_events=[
        ScheduledNews(
            step=7,
            headline="AI規制法案が国会に正式提出",
            summary="AI技術の商用利用に届出義務を課す法案が、与野党から提出された。業界団体は反発。",
            sentiment=0.2,
        ),
        ScheduledNews(
            step=14,
            headline="AI規制法案が賛成多数で可決",
            summary="AI規制法案が衆議院で賛成多数で可決。企業は6ヶ月以内に届出が必要に。",
            sentiment=0.4,
        ),
        ScheduledNews(
            step=21,
            headline="大手AI企業5社がAI規制法に対して集団訴訟を提起",
            summary="規制法は技術革新を阻害し憲法に抵触するとして、大手5社が集団で訴訟。",
            sentiment=-0.5,
        ),
        ScheduledNews(
            step=28,
            headline="AI規制の影響で国内スタートアップの海外移転が加速",
            summary="規制施行を前に、AI関連スタートアップの30%が海外拠点への移転を発表。",
            sentiment=-0.3,
        ),
        ScheduledNews(
            step=35,
            headline="世論調査：AI規制への賛成が60%に上昇",
            summary="最新の世論調査で、AI規制法への賛成が60%に。安全性への関心が高まる。",
            sentiment=0.5,
        ),
    ],
    acts=[
        {"name": "提出前の議論", "start": 1, "end": 7},
        {"name": "法案審議", "start": 8, "end": 14},
        {"name": "可決後の反応", "start": 15, "end": 21},
        {"name": "訴訟と混乱", "start": 22, "end": 28},
        {"name": "社会の変化", "start": 29, "end": 42},
    ],
)


# ════════════════════════════════════════════════════
# Scenario: Climate Policy (気候変動政策)
# ════════════════════════════════════════════════════

CLIMATE_POLICY = Scenario(
    id="climate_policy",
    name="気候変動対策をめぐる世論の変動",
    description=(
        "大型台風の被害をきっかけに気候変動対策への関心が高まるが、"
        "経済への影響を懸念する声との間で世論が揺れ動く。"
    ),
    topics=["climate change"],
    agent_count=100,
    total_steps=42,
    seed=2026,
    news_events=[
        ScheduledNews(
            step=5,
            headline="観測史上最大の台風が関東を直撃、死者50名",
            summary="カテゴリ5の超大型台風が関東地方を直撃。甚大な被害。",
            sentiment=-0.7,
        ),
        ScheduledNews(
            step=10,
            headline="政府が2030年までに排出50%削減の新目標を発表",
            summary="台風被害を受け、政府が大幅な排出削減目標を前倒し発表。",
            sentiment=0.5,
        ),
        ScheduledNews(
            step=18,
            headline="産業界が排出削減による雇用喪失30万人と試算",
            summary="経済団体が排出削減の経済影響を試算。製造業を中心に30万人の雇用に影響。",
            sentiment=-0.4,
        ),
        ScheduledNews(
            step=25,
            headline="再生可能エネルギー産業で新規雇用20万人の見通し",
            summary="グリーン産業の成長により、5年以内に20万人の新規雇用が見込まれる。",
            sentiment=0.4,
        ),
        ScheduledNews(
            step=32,
            headline="若者1万人が気候ストライキ、国会前に集結",
            summary="全国の高校生・大学生が気候変動対策の加速を求めてストライキ。",
            sentiment=0.3,
        ),
    ],
    acts=[
        {"name": "災害前", "start": 1, "end": 5},
        {"name": "危機感の高まり", "start": 6, "end": 14},
        {"name": "経済的反発", "start": 15, "end": 25},
        {"name": "新たな希望", "start": 26, "end": 35},
        {"name": "社会運動", "start": 36, "end": 42},
    ],
)


# ════════════════════════════════════════════════════
# Scenario: Digital Privacy (デジタルプライバシー)
# ════════════════════════════════════════════════════

DIGITAL_PRIVACY = Scenario(
    id="digital_privacy",
    name="監視社会 vs プライバシー権",
    description=(
        "政府による市民監視プログラムの存在が内部告発で明らかになり、"
        "安全保障とプライバシーの権利の間で社会が揺れる。"
    ),
    topics=["digital privacy"],
    agent_count=80,
    total_steps=35,
    seed=2026,
    news_events=[
        ScheduledNews(
            step=3,
            headline="内部告発者が政府の大規模市民監視プログラムを暴露",
            summary="元政府職員が、SNSデータを網羅的に収集する秘密プログラムの存在を告発。",
            sentiment=-0.6,
        ),
        ScheduledNews(
            step=10,
            headline="政府「監視プログラムはテロ防止に不可欠」と反論",
            summary="政府は監視プログラムの存在を認めつつ、国家安全保障のために必要と主張。",
            sentiment=0.1,
        ),
        ScheduledNews(
            step=17,
            headline="裁判所が監視プログラムの一部を違憲と判決",
            summary="最高裁が令状なしの通信傍受を違憲と判断。プログラムの大幅な見直しを命令。",
            sentiment=0.4,
        ),
        ScheduledNews(
            step=24,
            headline="監視プログラムがテロ未遂を事前に阻止していたことが判明",
            summary="過去3年間で5件のテロ計画を事前に発見・阻止していた事実が公表。",
            sentiment=-0.2,
        ),
        ScheduledNews(
            step=30,
            headline="新プライバシー保護法が成立、監視に裁判所の承認が必要に",
            summary="市民の権利を保護しつつ安全保障を確保する新法が成立。",
            sentiment=0.5,
        ),
    ],
    acts=[
        {"name": "暴露", "start": 1, "end": 5},
        {"name": "論争", "start": 6, "end": 15},
        {"name": "司法判断", "start": 16, "end": 22},
        {"name": "揺り戻し", "start": 23, "end": 28},
        {"name": "新たな均衡", "start": 29, "end": 35},
    ],
)

# ════════════════════════════════════════════════════
# Scenario: Labor & Automation (労働と自動化)
# ════════════════════════════════════════════════════

LABOR_AUTOMATION = Scenario(
    id="labor_automation",
    name="AIによる雇用の未来",
    description=(
        "大手企業がAIによる大規模リストラを発表し、"
        "技術進歩と雇用保護の間で社会全体の議論が沸騰する。"
    ),
    topics=["labor automation"],
    agent_count=120,
    total_steps=42,
    seed=2026,
    news_events=[
        ScheduledNews(
            step=5,
            headline="メガバンク3行がAI導入で3万人の人員削減を発表",
            summary="国内3大銀行が今後3年で合計3万人を削減。事務処理のAI化を加速。",
            sentiment=-0.6,
        ),
        ScheduledNews(
            step=12,
            headline="政府がAI失業者向けリスキリング支援を閣議決定",
            summary="AI化による失業者に対し、年間100万円のリスキリング補助金を決定。",
            sentiment=0.3,
        ),
        ScheduledNews(
            step=20,
            headline="AI活用企業の生産性が平均40%向上、利益は過去最高",
            summary="AI導入企業の調査で、生産性と利益が大幅に向上したことが判明。",
            sentiment=0.2,
        ),
        ScheduledNews(
            step=28,
            headline="ベーシックインカム導入の議論が国会で本格化",
            summary="AI時代の社会保障として、月額10万円のベーシックインカム構想が浮上。",
            sentiment=0.1,
        ),
        ScheduledNews(
            step=35,
            headline="AI共生型新産業で新規雇用50万人の試算",
            summary="AIを活用した新産業分野で、今後5年間に50万人の新規雇用が見込まれると発表。",
            sentiment=0.5,
        ),
    ],
    acts=[
        {"name": "リストラ発表", "start": 1, "end": 8},
        {"name": "政府対応", "start": 9, "end": 18},
        {"name": "恩恵と格差", "start": 19, "end": 27},
        {"name": "新たな社会設計", "start": 28, "end": 42},
    ],
)


# ════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════

ALL_SCENARIOS: dict[str, Scenario] = {
    "ai_regulation": AI_REGULATION,
    "climate_policy": CLIMATE_POLICY,
    "digital_privacy": DIGITAL_PRIVACY,
    "labor_automation": LABOR_AUTOMATION,
}


def get_scenario(scenario_id: str) -> Scenario:
    """Get a scenario by ID."""
    if scenario_id not in ALL_SCENARIOS:
        available = ", ".join(ALL_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario: {scenario_id}. Available: {available}")
    return ALL_SCENARIOS[scenario_id]


def list_scenarios() -> list[dict]:
    """List all available scenarios."""
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description[:100],
            "topics": s.topics,
            "agents": s.agent_count,
            "steps": s.total_steps,
            "news_events": len(s.news_events),
        }
        for s in ALL_SCENARIOS.values()
    ]
