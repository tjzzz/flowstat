# GitHub 上 A/B 实验工具包现状（检索对比）

> 检索时间：2026-10-19
> 用途：给 FlowStat 的 A/B 模块（src/abtools.py + metrics/metrics_ab）做对标，判断「自研 vs 复用 vs 补强」

## 判断

GitHub 上 A/B 实验工具**非常多且已成熟**，分三类：
1. **大厂权威库**（生产级、持续维护、许可清晰）——适合作为严肃分析底座
2. **功能齐全的独立库**（个人/小团队，覆盖 SRM+CUPED+Bayesian+power 全套）——适合学习和轻量落地
3. **分流/开关类框架**（Django/Laravel/iOS/Rails 等 SDK）——只做实验分流与 feature flag，不做统计

我们 FlowStat 的 `abtools` 属于「轻量自研核心」，覆盖面约在独立库的 60%，**缺非参数检验、贝叶斯、Monte Carlo power、比率指标 CUPED/CUPAC、聚类/开关回设计**。

---

## 一、大厂权威库（推荐优先对标）

| 仓库 | 来源 | Stars(⭐) | 覆盖能力 | 适合 |
|---|---|---|---|---|
| **spotify/confidence** | Spotify | ⭐310 | A/B 分析（frequentist + 部分 Bayesian）、pip 可装 `confidence` | 作为生产分析底座直接调用 |
| **mozilla/mozanalysis** | Mozilla | ⭐15 | 实验分析流水线、指标定义、Bootstrap 置信区间 | 大规模在线实验 |
| **google/GeoexperimentsResearch** | Google | ⭐142 | 地理实验（Geo）方法论 |  geographic assignment 场景 |
| **google/causalexpanalysis** | Google | ⭐19 | 因果/实验分析 | 因果推断扩展 |

> 注：二者均为 public（geo 已 archived），许可可直接商用参考。

## 二、功能齐全的独立库（最贴近你记忆中的"封装好工具"）

| 仓库 | Stars(⭐) | 亮点 | 覆盖清单 |
|---|---|---|---|
| **iamvisheshsrivastava/ab-lab** | ⭐1 | 端到端 + Streamlit UI，诚实标注局限 | SRM(卡方)、A/A 校验、Welch/pooled t、Mann-Whitney U、两比例 z、Cohen's d & Hedges' g、样本量/MDE(闭式+Monte Carlo)、CUPED、Beta-Binomial 贝叶斯 |
| **ghadfield32/ab_testing** | ⭐0 | 号称 200+ 方法/13 模块/80% 覆盖 | power、frequentist(z/t)、CUPED、SRM、**novelty detection、instrumental variables、network effects、guardrail metrics** |
| **luizhsuperti/cluster-experiments** | ⭐1 | 聚类随机/开关回实验专长 | power(Monte Carlo+解析)、MDE、A/B/聚类/开关回/分层、OLS/Clustered OLS/GEE/MLM、delta method(比率指标)、CUPED/CUPAC、合成控制、scorecard 多维切片 |
| **Arek-KesizAbnousi/causal-inference-experimentation-toolkit** | ⭐1 | 紧凑教学级 | power/MDE、Welch t、两比例 z、CUPED、DiD(TWFE)、1000× A/A 校验 |
| **tlentali/leab** | ⭐75 | 轻量分析小库 | 基础 A/B 分析 |

## 三、分流/开关框架（不做统计，仅做开关）

django-experiments(mixcloud)、abrouter/laravel-abtest、recruit-mp/ABKit、express-ab、ABRouter 等——与本任务（统计分析）无关，略。

---

## 与我们 FlowStat `abtools` 的差距

**我们已实现（abtools.py）**
- `srm_test`（卡方）→ ✅ 对齐 ab-lab/ghadfield
- `welch_ttest`（含差值 CI + Cohen's d）→ ✅
- `two_proportion_test`（z 检验）→ ✅
- `min_sample_size_mean`（**输入 CV + 相对提升 delta**）→ ✅ 独特友好设计
- `min_sample_size_proportion`（基线 + 绝对 MDE）→ ✅
- `cuped_adjust`（方差缩减）→ ✅
- `cohens_d`、`benjamini_hochberg`（FDR 校正）→ ✅

**业界有、我们缺**
- Mann-Whitney U（偏态指标的非参数兜底）
- Hedges' g（小样本效应量修正）
- Bayesian Beta-Binomial（P(B beats A)）
- Monte Carlo power simulation（闭式正态近似过于乐观时）
- 比率指标 CUPED/CUPAC + delta method
- 聚类/开关回/分层随机设计（cluster-experiments）
- novelty detection / guardrail metrics / IV / network effects（ghadfield32）

---

## 补充调研：真正热门的实验项目（按 GitHub Star，2026-10-19）

> 上一轮检索偏「纯统计函数库」，star 普遍偏低（≤310）。本轮按 star 阈值补检索，发现**真正热门的是「平台 / Feature-Flag」类**，而非统计函数库。

| 排名 | 仓库 | ⭐ | 🍴 | 类型 | 说明 |
|---|---|---|---|---|---|
| 1 | posthog/posthog | 40344 | 3525 | 全栈产品分析平台 | 含 A/B、session replay；完整 SaaS 式产品 |
| 2 | Unleash/unleash | 13884 | 910 | Feature Flag 平台 | 企业级开关 + 灰度实验 |
| 3 | growthbook/growthbook | 8509 | 884 | 实验平台 | warehouse-native；统计引擎含 CUPED / Bayesian / Sequential / MAB |
| 4 | github/scientist | 7771 | 508 | 代码级对照库(Ruby) | 新旧代码分支对照，非指标统计 |
| 5 | Flagsmith/flagsmith | 6602 | 590 | Feature Flag | 远程配置 + 多元测试 |
| 6 | flipt-io/flipt | 4923 | 314 | Feature Flag 引擎(Go) | 高性能开关评估 |
| 7 | featbit/featbit | 1946 | 150 | 实验平台 | 开发者中心 |
| 8 | seatgeek/sixpack | 1754 | 178 | 语言无关 A/B | Redis + Python |
| 9 | facebookarchive/planout | 1681 | 216 | 随机化框架 | Facebook 在线实验随机化（已归档）|
| 10 | intuit/wasabi | 1140 | 234 | A/B 平台 | Intuit 高性能实验 |
| 11 | open-feature/flagd | 1008 | 142 | CNCF 标准 | Feature Flag 规范 / 守护进程 |
| 12 | spotify/confidence | 310 | 34 | 纯统计 SDK | Python 纯统计（见上一轮）|

### 洞察（对 FlowStat 定位）

- **k-star 级项目全是「平台 / Feature-Flag」**，不是统计函数库。它们把分流 + 分析 + 可视化一体化，统计内嵌（GrowthBook 明确支持 CUPED / Bayesian / Sequential / MAB）。
- **Python 纯统计函数库几乎没有高 star 开源**：scientist 是 Ruby 且偏代码对照；confidence(310) 已是该细分 top。说明多数团队直接用平台而非自研统计。
- **对 FlowStat 的两条路**：
  - 要「开箱即用严肃实验」→ 直接部署 **GrowthBook**（统计最严谨、warehouse-native），无需自研平台；
  - 坚持「轻量可插拔统计框架」（当前定位）→ 参考 **GrowthBook 的统计方法论**（补 Sequential / Bayesian / MAB）+ 把 **confidence** 当 Python 纯统计补充；FlowStat 差异点是 yaml 驱动、无平台负担、可嵌入自有管道。

---

## 建议路线（待你定）

- **A. 保持轻量自研**：FlowStat 定位「通用框架+可插拔」，abtools 已够 demo 与中小实验；缺的项按需补（性价比最高）
- **B. 对接大厂底座**：生产严肃分析时，`metrics/metrics_ab` 内封装调用 `spotify/confidence`，自研仅做调度/接口
- **C. 参考独立库补强**：把 ab-lab / ghadfield32 的 Mann-Whitney、Bayesian、Monte Carlo power 移植进 abtools（功能最全但需自己保证正确性）

下一步问你选哪条；若选 C，我建议优先补 **Mann-Whitney U + Hedges' g + Bayesian Beta-Binomial + Monte Carlo power**，这四项 ROI 最高、且不与现有框架冲突。
