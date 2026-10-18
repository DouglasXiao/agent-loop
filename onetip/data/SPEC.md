# 内容规范（Tip 写作 + 数据格式）

本文件是 `tips/*.json` 与 `sources.json` 的唯一规范。写 tip 的人和校验脚本 `pipeline/validate_tips.py` 都以此为准。

## 1. 读者与孩子

- 读者：一位家长（爸爸或妈妈），中文母语，白天上班，晚上/周末带娃。可能有老人帮忙带。
- 孩子：女儿，当前 **2 岁 11 个月（35 个月）**，App 会陪她到 4 岁左右，所以 tip 覆盖 **30–54 个月**，重点 33–42 个月。
- 文中称孩子为「她」或「孩子」，称读者为「你」。

## 2. 单条 tip 写作要求

| 字段 | 要求 |
|---|---|
| `title` | 4–14 个汉字，具体、有画面感，像一句提醒，不要「浅谈」「如何」式标题。 |
| `body` | **100–160 个字符**（含标点，`len(body)` 计）。结构：一句「为什么/什么场景」+ 具体做法 + 一句能直接照说的话术或「今天就试」的小动作。 |
| `basis` | 一句话（≤60 字）写清依据：哪个机构/研究/指南说了什么。给审核人和「查看依据」用，不是正文。 |

写作风格：

- 温暖、平视、具体。像一个懂行的朋友，不说教、不鸡汤、不制造焦虑。
- 一条只讲一件事。能当晚就用。
- 优先给「话术」：引号里写出家长可以原样说的一句话。
- 不出现药物剂量、诊断结论；涉及健康异常只说「留意…，持续/明显时问儿科医生」。
- 不推荐已被主流证据否定的做法：打骂/羞辱、哭声免疫、用食物或屏幕当奖惩、强迫分享、吓唬（「警察来抓你」）等。
- 不照抄任何来源原文（版权）。用自己的话改写，依据写在 `basis`/`source_ids`。
- 考虑中国家庭场景：老人带娃、幼儿园/托班、小区玩伴、过年走亲戚等，但不要只适用于某一个城市。

## 3. JSON 格式

`tips/<category>.json` 是一个数组，每个元素：

```json
{
  "id": "emotion-001",
  "category": "emotion",
  "title": "先接住，再讲理",
  "body": "100–160 字正文……",
  "age_min_months": 30,
  "age_max_months": 48,
  "tags": ["tantrum", "connection"],
  "source_ids": ["harvard-cdc", "gottman"],
  "basis": "Gottman 情绪辅导研究：先命名和认可情绪，孩子更快平复、日后自我调节更好。"
}
```

- `id`：`<category>-<三位序号>`，在文件内唯一。
- `category`：必须是下表之一。
- `age_min_months` / `age_max_months`：整数，24 ≤ min < max ≤ 72，且区间必须包含 35–42 中的至少一部分或在其之后（App 会按孩子当前月龄筛选）。
- `tags`：1–4 个英文小写短词，自由填写。
- `source_ids`：1–3 个，必须来自第 5 节的 ID 列表。

## 4. 分类

| category | 中文 | 目标条数 |
|---|---|---|
| `language` | 语言与沟通 | 40 |
| `emotion` | 情绪与情商 | 45 |
| `discipline` | 规则与正面管教 | 45 |
| `play` | 游戏与早期学习（阅读、数学、专注力、创造力、运动） | 50 |
| `selfcare` | 生活自理与习惯（如厕、穿衣、家务、作息） | 30 |
| `sleep` | 睡眠 | 20 |
| `food` | 饮食与营养 | 25 |
| `safety` | 健康与安全（居家/出行安全、身体保护教育、口腔、视力、生病） | 30 |
| `social` | 社交与入园（同伴、分享、幼儿园适应） | 35 |
| `family` | 亲子关系与家庭（专属时光、爸爸参与、老人带娃、二胎、家长自我照顾） | 45 |

## 5. 来源 ID 列表（`source_ids` 只能用这些）

国际 / 官方 / 学术：

| id | 来源 |
|---|---|
| `cdc-milestones` | CDC「Learn the Signs. Act Early.」3 岁/4 岁发育里程碑 |
| `cdc-parenting` | CDC Positive Parenting Tips（学龄前 3–5 岁）与 Essentials for Parenting |
| `aap` | 美国儿科学会 AAP / HealthyChildren.org（含屏幕、如厕、睡眠、安全） |
| `aap-bright-futures` | AAP Bright Futures 3 岁/4 岁体检指导 |
| `who-under5` | WHO 2019《5 岁以下儿童身体活动、久坐行为和睡眠指南》 |
| `who-unicef-nurturing-care` | WHO/UNICEF 养育照护框架（Nurturing Care）与 UNICEF Parenting |
| `harvard-cdc` | 哈佛大学儿童发展中心（serve & return、执行功能活动指南、压力与韧性） |
| `zero-to-three` | ZERO TO THREE（幼儿行为、情绪、游戏） |
| `vroom` | Vroom 大脑建构日常小活动（Bezos Family Foundation） |
| `nhs` | 英国 NHS（Start for Life、幼儿发脾气、如厕、饮食） |
| `raising-children-au` | 澳大利亚 Raising Children Network（学龄前） |
| `child-mind` | Child Mind Institute（焦虑、行为、情绪） |
| `pcit` | 亲子互动疗法 PCIT（PRIDE 技巧：表扬、复述、模仿、描述、享受） |
| `triple-p` | Triple P 积极养育项目 |
| `gottman` | Gottman 情绪辅导（Emotion Coaching） |
| `hanen` | Hanen Centre（幼儿语言发展） |
| `asha` | 美国言语听力协会 ASHA（语言发育） |
| `dialogic-reading` | 对话式阅读（Whitehurst，PEER/CROWD），What Works Clearinghouse |
| `aasm` | 美国睡眠医学会 AASM 儿童睡眠时长共识 |
| `satter` | Ellyn Satter 喂养责任分工（Division of Responsibility） |
| `naeyc` | 全美幼教协会 NAEYC 家长资源 |
| `kidshealth` | Nemours KidsHealth |
| `hirsh-pasek` | Hirsh-Pasek & Golinkoff 游戏化学习研究（《爱因斯坦从不用闪卡》《成为聪明人》） |
| `book-how-to-talk` | 《如何说孩子才会听（2–7 岁版）》Faber & King |
| `book-whole-brain` | 《全脑教养法》《不打不骂》Siegel & Bryson |
| `book-positive-discipline` | 《正面管教（学龄前）》Nelsen |
| `book-good-inside` | 《Good Inside》Becky Kennedy |
| `book-gardener-carpenter` | 《园丁与木匠》Alison Gopnik |
| `book-30m-words` | 《父母的语言》Dana Suskind |
| `book-montessori-toddler` | 《蒙特梭利幼儿教育》Simone Davies |
| `book-hunt-gather-parent` | 《Hunt, Gather, Parent》Michaeleen Doucleff |

中国官方 / 专业：

| id | 来源 |
|---|---|
| `nhc-care-guide` | 国家卫健委《3 岁以下婴幼儿健康养育照护指南（试行）》2022 |
| `nhc-psych-norm` | 国家卫健委《儿童心理保健技术规范》（含心理行为发育预警征象） |
| `moe-3-6-guide` | 教育部《3–6 岁儿童学习与发展指南》2012 |
| `cns-preschool-diet` | 中国营养学会《中国学龄前儿童膳食指南（2022）》 |
| `cn-preschool-exercise` | 《学龄前儿童（3–6 岁）运动指南》2018 专家共识 |
| `cn-myopia` | 教育部等八部门《综合防控儿童青少年近视实施方案》及近视防控适宜技术指南 |
| `cn-oral-health` | 国家卫健委 / 中华口腔医学会儿童口腔保健建议（含涂氟、窝沟） |
| `cn-child-injury` | 中国疾控 / 国家卫健委儿童伤害预防（溺水、跌落、烫伤、道路安全） |
| `dxy-mama` | 丁香妈妈（医生审核科普，仅作参考，不转载） |
