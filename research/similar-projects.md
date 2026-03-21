# 類似プロジェクト・関連研究リサーチ

## 概要

AIエージェントによるソーシャルネットワークシミュレーション、特に意見分極化（polarization）に関する類似プロジェクトと関連研究をまとめる。

---

## 主要な類似プロジェクト（GitHub実装あり）

### 1. OASIS — Open Agent Social Interaction Simulations
- **GitHub**: https://github.com/camel-ai/oasis
- **論文**: https://arxiv.org/abs/2411.11581
- **概要**: 最大100万エージェントでTwitterやRedditをシミュレートする大規模ソーシャルメディアシミュレータ。LLMエージェントが23種類のアクション（フォロー、コメント、リポストなど）を実行可能。
- **特徴**:
  - スケーラブル（最大100万エージェント）
  - 興味ベース・ホットスコアベースのレコメンデーションシステム
  - TwitterとRedditの環境を再現
  - PyPIで公開、CAMEL-AIフレームワーク上に構築
- **知見**: シミュレーションが進むにつれ、ユーザーの意見がより極端になる傾向（分極化）が確認された。Uncensoredモデルではさらに顕著。
- **技術**: Python, GPT-4o-mini, CAMEL framework

### 2. LLM Agent Opinion Dynamics
- **GitHub**: https://github.com/yunshiuan/llm-agent-opinion-dynamics
- **論文**: https://arxiv.org/abs/2311.09618 (NAACL 2024)
- **概要**: LLMベースのエージェント集団によるオピニオンダイナミクスシミュレーション。確認バイアス（confirmation bias）やメモリ更新関数を操作し、意見の進化を研究。
- **特徴**:
  - ネットワーク上の多エージェント会話をシミュレート
  - 確認バイアスのプロンプトエンジニアリングによる導入
  - 複数トピック（気候変動など）での実験
- **知見**: LLMエージェントには正確な情報を生成する強い固有バイアスがあり、デフォルトではコンセンサスに収束する。確認バイアスを導入することで意見の断片化が観察された。
- **技術**: Python, OpenAI API

### 3. Casevo — Cognitive Agents and Social Evolution Simulator
- **GitHub**: https://github.com/rgCASS/casevo
- **論文**: https://arxiv.org/abs/2412.19498
- **概要**: LLMを統合した社会シミュレーション用マルチエージェントフレームワーク。離散イベントシミュレータとして設計。
- **特徴**:
  - Chain of Thoughts (CoT)、RAG、カスタマイズ可能なメモリ機構
  - Mesaフレームワーク上に構築
  - ロール挿入、長期・短期メモリメカニズム
  - 米国2020年中間選挙のTVディベートをデモとして使用
- **技術**: Python 3.11+, Mesa framework, LLM

### 4. Generative Agents (Stanford Smallville)
- **GitHub**: https://github.com/joonspk-research/generative_agents
- **論文**: https://arxiv.org/abs/2304.03442 (UIST 2023)
- **概要**: 信頼性のある人間行動をシミュレートする生成エージェントの先駆的研究。Smallvilleというサンドボックス環境で25エージェントが生活。
- **特徴**:
  - メモリストリーム（包括的な経験記録）
  - リフレクション（高次の抽象化）
  - リトリーバル（関連記憶の動的取得）
  - 計画立案と行動決定
- **意義**: 本分野の基盤となる研究。多くの後続プロジェクトがこのアーキテクチャを参照。
- **技術**: Python, OpenAI API

### 5. Opinion Polarization (Network Dynamics)
- **GitHub**: https://github.com/adamlechowicz/opinion-polarization
- **概要**: ソーシャルネットワーク上のローカルダイナミクスが意見分極化を駆動するメカニズムの研究。Friedkin-Johnsenオピニオンモデルを使用。
- **特徴**:
  - 時間発展ネットワーク
  - エッジの追加・削除による動的ネットワーク構造
  - レコメンダーシステムの影響分析

### 6. MultiAgent Social Simulation
- **GitHub**: https://github.com/ahaque2/MultiAgent-Social-Simulation
- **論文**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9859750/
- **概要**: 選択的暴露（selective exposure）と寛容性（tolerance）が分極化に与える影響をマルチエージェントシミュレーションで分析。
- **知見**: 高い寛容性は分極化を遅らせるが、ユーザー満足度を低下させる。高い選択的暴露は分極化を促進。

---

## 関連研究論文（実装非公開・進行中）

### 7. Human-Agent Interaction in Synthetic Social Networks (ICWSM/AAAI)
- **論文**: https://arxiv.org/html/2506.15866v1
- **概要**: LLMエージェントと人間参加者を統合した合成ソーシャルネットワークプラットフォーム。122名の被験者実験で、分極化した議論の特徴を再現することに成功。

### 8. RecSysLLMsP — Algorithmic Personalization and Polarization
- **論文**: https://link.springer.com/article/10.1007/s41111-025-00326-x
- **概要**: アルゴリズムによるパーソナライゼーションが分極化に与える影響を100エージェントで検証。セルビアのソーシャルメディアユーザーの心理測定・人口統計データに基づく。

### 9. Agent-Based Modelling Meets Generative AI in Social Network Simulations
- **論文**: https://arxiv.org/html/2411.16031v1
- **概要**: 2020年米国大統領選挙でのトランプ対バイデンのTwitter会話を再現。ホモフィリー、分極化、論争の分析。

### 10. Decoding Echo Chambers: LLM-Powered Simulations (COLING 2025)
- **論文**: https://aclanthology.org/2025.coling-main.264.pdf
- **概要**: エコーチャンバー効果のデコードと緩和戦略。ナッジ操作によるエコーチャンバーと分極化の軽減を実証。

### 11. FDE-LLM: Fusing Dynamics Equation with LLM-based Agents
- **論文**: https://www.nature.com/articles/s41598-025-99704-3
- **概要**: オピニオンリーダーとフォロワーに分け、LLMによるロールプレイとセルオートマトンによる意見変化の制約を組み合わせたモデル。

---

## キュレーションリスト・サーベイ

| リソース | URL | 説明 |
|---------|-----|------|
| SocialAgent | https://github.com/FudanDISC/SocialAgent | ソーシャルエージェント研究論文のコレクション |
| LLM-Agents-for-Simulation | https://github.com/giammy677dev/LLM-Agents-for-Simulation | シミュレーションとLLMエージェントの交差点に関するリソース集 |
| LLM-Agent-Based-Modeling | https://github.com/tsinghua-fib-lab/LLM-Agent-Based-Modeling-and-Simulation | 清華大学によるLLMエージェントベースモデリングのサーベイ |
| Autonomous-Agents | https://github.com/tmgthb/Autonomous-Agents | 自律エージェント研究論文の日次更新リスト |

---

## 共通の課題・知見

### LLMの固有バイアス
- LLMはRLHFにより「正しい」回答を生成するバイアスがあり、分極化や誤情報のシミュレーションが困難
- プロンプトエンジニアリングだけでは多様な視点の完全な再現は不十分

### 中立への収束
- 現在のLLMは意見が中立値に収束する傾向があり、時間経過とともに分極化が減少
- 確認バイアスの明示的な導入が分極化再現の鍵

### アルゴリズム増幅の役割
- レコメンデーションアルゴリズムがエコーチャンバーを形成・強化
- コンテンツの選択的暴露が分極化を加速

### スケーラビリティ
- 大規模シミュレーション（OASIS: 100万エージェント）からスモールスケール実験まで幅広い
- コスト面ではAPIコールの最適化が重要

---

## 本プロジェクトへの示唆

1. **アーキテクチャ参考**: OASIS（大規模）やCasevo（Mesa基盤）のアーキテクチャが参考になる
2. **メモリ機構**: Generative Agentsのメモリストリーム・リフレクション機構は意見形成の深さに寄与
3. **バイアス設計**: 確認バイアスやエコーチャンバー効果の明示的なモデリングが分極化再現に必要
4. **評価指標**: 分極化の定量的な計測方法（意見分布、ネットワーク構造、ホモフィリー指数など）
5. **トピック選定**: 気候変動、選挙、政策議論など、意見が分かれやすいトピックが研究に適する
