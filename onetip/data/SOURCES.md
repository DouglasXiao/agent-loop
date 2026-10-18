# 数据源清单

> 由 `pipeline/render_sources.py` 从 `data/sources/*.json` 生成，请改 JSON 再重新生成。

已收录 **40** 个来源（tip 的 `source_ids` 只能引用这些），另有 **22** 个候选来源待评估。

| 使用方式 | 数量 | 含义 |
|---|---|---|
| ✅ 可改写引用 | 9 | 公有领域 / 国家机关官方文件 / 宽松许可：可抓取、可改写，注明出处 |
| 🔗 仅参考+链接 | 23 | 有版权的网站：不抓取正文，只作依据，用自己的话写并附链接 |
| 📕 书籍，仅参考 | 8 | 有版权的书：只参考思路 |

核实状态说明：本次在云端沙箱里整理，网络策略挡住了直接打开大多数来源网页，核实主要靠搜索引擎索引和摘录；标 ⚪ 的条目是凭已有知识写的，上线前要逐条打开原页复核。

## 总览

| id | 来源 | 地区 | 层级 | 使用方式 | 核实 | 擅长 |
|---|---|---|---|---|---|---|
| `cn-child-injury` | 儿童伤害预防（中国疾控中心 / 国家卫健委） | CN | A | 🔗 仅参考+链接 | — | 安全 |
| `cn-myopia` | 综合防控儿童青少年近视实施方案（及近视防控技术指南） | CN | A | ✅ 可改写引用 | — | 安全、游戏、家庭 |
| `cn-oral-health` | 儿童口腔保健（国家卫健委 / 中华口腔医学会相关文件） | CN | A | 🔗 仅参考+链接 | — | 安全、自理、饮食 |
| `cns-preschool-diet` | 中国学龄前儿童膳食指南（2022） | CN | A | 🔗 仅参考+链接 | — | 饮食、自理、游戏 |
| `moe-3-6-guide` | 3-6岁儿童学习与发展指南 | CN | A | ✅ 可改写引用 | — | 自理、情绪、语言、社交、游戏、安全 |
| `nhc-care-guide` | 3岁以下婴幼儿健康养育照护指南（试行） | CN | A | ✅ 可改写引用 | — | 睡眠、饮食、游戏、自理、安全、家庭 |
| `nhc-psych-norm` | 儿童心理保健技术规范 | CN | A | ✅ 可改写引用 | — | 语言、情绪、社交、游戏、自理 |
| `cn-preschool-exercise` | 学龄前儿童（3～6岁）运动指南 | CN | B | 🔗 仅参考+链接 | — | 游戏、家庭 |
| `dxy-mama` | 丁香妈妈 | CN | C | 🔗 仅参考+链接 | — | 睡眠、饮食、安全、情绪、管教 |
| `aap` | American Academy of Pediatrics — HealthyChildren.org（美国儿科学会（AAP）家长网站 HealthyChildren.org） | US | A | 🔗 仅参考+链接 | 🟡 | 安全、自理、睡眠、饮食、语言、管教、家庭、游戏 |
| `aap-bright-futures` | AAP Bright Futures — 3 Year and 4 Year Visit parent handouts（美国儿科学会 Bright Futures 3 岁/4 岁体检家长指导） | US | A | 🔗 仅参考+链接 | 🟡 | 安全、饮食、自理、家庭 |
| `aasm` | American Academy of Sleep Medicine — pediatric sleep duration consensus (2016)（美国睡眠医学会 AASM 儿童睡眠时长共识（2016）） | US | A | 🔗 仅参考+链接 | ⚪ | 睡眠 |
| `asha` | American Speech-Language-Hearing Association (ASHA) — communication milestones（美国言语听力协会 ASHA（语言发育里程碑）） | US | A | 🔗 仅参考+链接 | ⚪ | 语言、社交 |
| `cdc-milestones` | CDC Learn the Signs. Act Early. — Milestones by 3 and 4 Years（美国疾控中心「早发现早行动」3 岁/4 岁发育里程碑） | US | A | ✅ 可改写引用 | ✅ | 语言、社交、情绪、游戏、自理 |
| `cdc-parenting` | CDC Positive Parenting Tips (Toddlers 2–3, Preschoolers 3–5) & Essentials for Parenting Toddlers and Preschoolers（美国疾控中心 积极养育小贴士（学步儿/学龄前）与《幼儿养育要点》） | US | A | ✅ 可改写引用 | ✅ | 管教、睡眠、安全、家庭、游戏、情绪 |
| `dialogic-reading` | Dialogic Reading (Whitehurst; PEER/CROWD) — What Works Clearinghouse（对话式阅读（Whitehurst，PEER/CROWD）） | US | A | ✅ 可改写引用 | ⚪ | 语言、游戏 |
| `harvard-cdc` | Center on the Developing Child at Harvard University（哈佛大学儿童发展中心（serve & return、执行功能、压力与韧性）） | US | A | 🔗 仅参考+链接 | ✅ | 语言、情绪、游戏、家庭 |
| `hirsh-pasek` | Hirsh-Pasek & Golinkoff — playful / guided-play learning research（Hirsh-Pasek & Golinkoff 游戏化学习研究（《爱因斯坦从不用闪卡》《成为聪明人》）） | US | A | 🔗 仅参考+链接 | ⚪ | 游戏、语言 |
| `naeyc` | NAEYC — For Families（全美幼教协会 NAEYC 家长资源） | US | A | 🔗 仅参考+链接 | ⚪ | 游戏、社交、语言、家庭 |
| `nhs` | NHS (UK) — Best Start in Life / Start for Life, toddler behaviour, potty training, feeding（英国国家医疗服务体系 NHS（Best Start in Life / Start for Life）） | UK | A | ✅ 可改写引用 | 🟡 | 情绪、自理、饮食、语言、安全、睡眠 |
| `who-under5` | WHO Guidelines on physical activity, sedentary behaviour and sleep for children under 5 years of age (2019)（世界卫生组织《5 岁以下儿童身体活动、久坐行为和睡眠指南》（2019）） | Global | A | ✅ 可改写引用 | ✅ | 游戏、睡眠、自理、家庭 |
| `who-unicef-nurturing-care` | WHO/UNICEF Nurturing Care Framework, WHO ECD & parenting guidelines, UNICEF Parenting（世卫组织/联合国儿基会「养育照护」框架与 UNICEF 育儿指导） | Global | A | 🔗 仅参考+链接 | ✅ | 家庭、情绪、游戏、饮食、安全 |
| `child-mind` | Child Mind Institute（美国儿童心智研究所 Child Mind Institute） | US | B | 🔗 仅参考+链接 | ⚪ | 情绪、管教、社交、家庭 |
| `gottman` | Gottman Emotion Coaching（Gottman 情绪辅导（Emotion Coaching）） | US | B | 🔗 仅参考+链接 | ⚪ | 情绪、家庭、管教 |
| `hanen` | The Hanen Centre（加拿大 Hanen 中心（幼儿语言发展）） | CA | B | 🔗 仅参考+链接 | ⚪ | 语言、游戏、家庭 |
| `kidshealth` | Nemours KidsHealth — for parents（Nemours KidsHealth 儿童健康网） | US | B | 🔗 仅参考+链接 | 🟡 | 安全、自理、饮食、睡眠、语言 |
| `pcit` | Parent-Child Interaction Therapy (PCIT) — PRIDE skills（亲子互动疗法 PCIT（PRIDE 技巧）） | US | B | 🔗 仅参考+链接 | ⚪ | 管教、游戏、情绪、家庭 |
| `raising-children-au` | Raising Children Network (Australia) — preschoolers（澳大利亚 Raising Children Network 育儿网（学龄前）） | AU | B | 🔗 仅参考+链接 | ⚪ | 情绪、管教、游戏、社交、睡眠、饮食、自理、家庭 |
| `satter` | Ellyn Satter Institute — Division of Responsibility in Feeding（Ellyn Satter 喂养责任分工） | US | B | 🔗 仅参考+链接 | ⚪ | 饮食、家庭 |
| `triple-p` | Triple P — Positive Parenting Program（Triple P 积极养育项目） | AU | B | 🔗 仅参考+链接 | ⚪ | 管教、家庭、情绪 |
| `vroom` | Vroom (brain-building tips)（Vroom 大脑建构日常小活动） | US | B | 🔗 仅参考+链接 | ✅ | 游戏、语言、家庭、情绪 |
| `zero-to-three` | ZERO TO THREE — resources for families（ZERO TO THREE（零到三岁）家长资源） | US | B | 🔗 仅参考+链接 | 🟡 | 情绪、管教、游戏、语言、家庭 |
| `book-30m-words` | Thirty Million Words: Building a Child's Brain (Dana Suskind, 2015)（《父母的语言》Dana Suskind） | US | C | 📕 书籍，仅参考 | ⚪ | 语言、家庭 |
| `book-gardener-carpenter` | The Gardener and the Carpenter (Alison Gopnik, 2016)（《园丁与木匠》Alison Gopnik） | US | C | 📕 书籍，仅参考 | ⚪ | 游戏、家庭 |
| `book-good-inside` | Good Inside: A Guide to Becoming the Parent You Want to Be (Becky Kennedy, 2022)（《Good Inside》Becky Kennedy） | US | C | 📕 书籍，仅参考 | ⚪ | 情绪、管教、家庭 |
| `book-how-to-talk` | How to Talk So Little Kids Will Listen: A Survival Guide to Life with Children Ages 2–7 (Joanna Faber & Julie King, 2017)（《如何说孩子才会听（2–7 岁版）》Faber & King） | US | C | 📕 书籍，仅参考 | ⚪ | 管教、情绪、语言、家庭 |
| `book-hunt-gather-parent` | Hunt, Gather, Parent (Michaeleen Doucleff, 2021)（《Hunt, Gather, Parent》Michaeleen Doucleff） | US | C | 📕 书籍，仅参考 | ⚪ | 家庭、自理、管教、情绪 |
| `book-montessori-toddler` | The Montessori Toddler (Simone Davies, 2019)（《蒙特梭利幼儿教育》Simone Davies） | NL/US | C | 📕 书籍，仅参考 | ⚪ | 自理、游戏、家庭 |
| `book-positive-discipline` | Positive Discipline for Preschoolers (Jane Nelsen, Cheryl Erwin, Roslyn Ann Duffy)（《正面管教（学龄前）》Nelsen） | US | C | 📕 书籍，仅参考 | ⚪ | 管教、家庭、自理 |
| `book-whole-brain` | The Whole-Brain Child (2011) & No-Drama Discipline (2014) — Daniel J. Siegel & Tina Payne Bryson（《全脑教养法》《不打不骂》Siegel & Bryson） | US | C | 📕 书籍，仅参考 | ⚪ | 情绪、管教、家庭 |

## 详情

### 儿童伤害预防（中国疾控中心 / 国家卫健委）

- **id**: `cn-child-injury` · **发布方**: 中国疾病预防控制中心（慢性非传染性疾病预防控制中心）；国家卫生健康委员会 · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — 国家卫健委发布的行政文件和法律条文可以改写（第五条）。中国疾控中心是事业单位，不属于第五条所指的国家机关，其技术指南和科普作品受著作权保护，只能引用。核实之前按 cite_only 处理。
- **获取**: robots 未确认
- **擅长**: 安全
- **备注**: 【未核实】原因同 cn-oral-health（检索额度用尽，中国站点不可达）。已核实的替代依据：nhc-care-guide 有「伤害预防」咨询指导章节（0–3 岁）；moe-3-6-guide 有 3–4 岁安全目标（不跟陌生人走、走失时能说出信息）。待核实清单（据既有知识）：①《中华人民共和国未成年人保护法》（2020 修订）第十八条：监护人应排除触电、烫伤、跌落等安全隐患，配备儿童安全座椅，防止溺水、动物伤害等事故（法律属第五条范围，可改写）；②国务院《中国儿童发展纲要（2021—2030 年）》「儿童与安全」领域；③中国疾控中心慢病中心编制的儿童溺水、跌落、烧烫伤、道路交通伤害、中毒干预技术指南系列（年份待核）；④国家疾控局或中国疾控中心每年发布的「预防儿童溺水」等健康提示（另见候选 ndcpa-kepu）。

### 综合防控儿童青少年近视实施方案（及近视防控技术指南）

- **id**: `cn-myopia` · **发布方**: 教育部、国家卫生健康委员会等八部门（经国务院同意）；技术指南：国家卫生健康委员会（2019）、国家疾病预防控制局（2023） · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — 教体艺〔2018〕3号（2018-08-30）；配套技术指南为国卫办疾控函〔2019〕780号、国疾控综卫免〔2023〕218号。都是国家机关行政文件，依《著作权法》第五条可改写并注明出处。
- **获取**: robots 未确认
- **擅长**: 安全、游戏、家庭
- **链接**: [1](http://www.moe.gov.cn/srcsite/A17/moe_943/s3285/201808/t20180830_346672.html)
- **和 3–4 岁相关的要点**:
  - 0～6 岁是孩子视觉发育的关键期，家长应格外重视孩子早期的视力保护与健康。
  - 家长陪伴孩子时应尽量减少自己使用电子产品，并有意识地控制孩子特别是学龄前儿童使用电子产品：非学习目的的使用单次不宜超过 15 分钟，每天累计不宜超过 1 小时；用电子产品学习 30～40 分钟后，应休息、远眺放松 10 分钟，年龄越小，连续使用时间应越短。
  - 引导孩子到户外阳光下活动，在家时每天接触户外自然光的时间达 60 分钟以上；已经近视的孩子应进一步增加户外时间。
  - 目标：到 2030 年，6 岁儿童近视率控制在 3% 左右。
  - 幼儿园应保证儿童每天 2 小时以上户外活动（寄宿制不少于 3 小时），其中体育活动不少于 1 小时（据广东省转发版原文，国家方案有对应条款）。
  - 国家疾控局 2023 年《儿童青少年近视防控公共卫生综合干预技术指南》：学龄前儿童尽量避免接触和使用视屏类电子产品。
- **备注**: 【已核实：教育部原文 URL 与标题（出现在检索结果中）、文号，以及上述家庭条款原文（多省转发版逐字一致）】本环境无法直连 moe.gov.cn 核验。技术指南只有 PDF 镜像（武汉市疾控，文号取自文件名；未放进 urls，以免 harvest 抓到 PDF 乱码）：2023 公共卫生综合干预技术指南 https://www.whcdc.org/d/file/p/2024/07-10/%E5%9B%BD%E5%AE%B6%E7%BA%A7-%E5%84%BF%E7%AB%A5%E9%9D%92%E5%B0%91%E5%B9%B4%E8%BF%91%E8%A7%86%E9%98%B2%E6%8E%A7%E5%85%AC%E5%85%B1%E5%8D%AB%E7%94%9F%E7%BB%BC%E5%90%88%E5%B9%B2%E9%A2%84%E6%8A%80%E6%9C%AF%E6%8C%87%E5%8D%97%EF%BC%88%E5%9B%BD%E7%96%BE%E6%8E%A7%E7%BB%BC%E5%8D%AB%E5%85%8D%5B2023%5D218%E5%8F%B7%EF%BC%89.pdf ；2019 适宜技术指南 https://tumour.whcdc.org/d/file/p/2024/07-10/国家级-儿童青少年近视防控适宜技术指南（国卫办疾控函[2019]780号）.pdf 。2019 适宜技术指南的具体条款本次未核实。教育部新闻稿：http://www.moe.gov.cn/jyb_xwfb/gzdt_gzdt/s5987/201808/t20180830_346673.html 。学龄前眼保健的具体检查时点见候选 nhc-eye-0-6。屏幕口径对比：本方案（学龄前非学习用途单次 ≤15 分钟、每天 ≤1 小时）与 nhc-care-guide（2 岁以上每次 ≤20 分钟、每天 ≤1 小时）单次上限不同，写 tip 时取更严的 15 分钟或分别注明来源。

### 儿童口腔保健（国家卫健委 / 中华口腔医学会相关文件）

- **id**: `cn-oral-health` · **发布方**: 国家卫生健康委员会；中华口腔医学会 · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — 视具体文件而定：国家卫健委发布的行动方案或技术规范属行政文件（第五条，可改写）；中华口腔医学会的专家共识或指南发表在期刊或学会出版物上，受著作权保护，只能引用。核实之前按 cite_only 处理。
- **获取**: robots 未确认
- **擅长**: 安全、自理、饮食
- **备注**: 【未核实】检索到这一条之前，本会话的 WebSearch 额度（所有代理共享 200 次）已经用完；WebFetch 和 curl 又无法访问 gov.cn 等中国站点。所以这里没有 URL，也没有核实过的数值。待核实清单（据既有知识，核实前不能写进 tip）：①《健康口腔行动方案（2019—2025 年）》，国家卫健委办公厅，文号可能是国卫办疾控发〔2019〕10 号；②《中国居民口腔健康指南》（原卫生部疾控局、中华口腔医学会，2009）儿童部分：含氟牙膏用量（3 岁以下米粒大小、3–6 岁豌豆大小），乳磨牙及第一恒磨牙窝沟封闭的适宜年龄；③《儿童口腔保健指导技术规范》，可能是卫办妇社发〔2013〕26 号的配套规范之一；④中华口腔医学会儿童口腔医学专委会关于局部涂氟频率的共识（据记忆为中、高龋风险儿童每 3–6 个月一次）。核实之前，刷牙类 tip 可以引用已核实的 moe-3-6-guide（3–4 岁「在提醒下每天早晚刷牙」）。

### 中国学龄前儿童膳食指南（2022）

- **id**: `cns-preschool-diet` · **发布方**: 中国营养学会 · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — 中国营养学会是社会团体，不是国家机关；指南和平衡膳食宝塔以图书形式出版（收录于《中国居民膳食指南（2022）》），受著作权保护。推荐数值属于事实，可用自己的话转述并注明出处，但不得复制原文段落或宝塔图。
- **获取**: robots 未确认
- **擅长**: 饮食、自理、游戏
- **链接**: [1](http://dg.cnsoc.org/) · [2](https://www.cnsoc.org/) · [3](http://www.nyetyxzx.com/e/action/ShowInfo.php?classid=12&id=788) · [4](https://www.thepaper.cn/newsDetail_forward_18439349)
- **和 3–4 岁相关的要点**:
  - 适用于满 2 周岁至满 5 周岁儿童。五条核心推荐：食物多样、规律就餐、自主进食、培养健康饮食行为；每天饮奶、足量饮水、合理选择零食；合理烹调、少调料少油炸；参与食物选择与制作，增进对食物的认知与喜爱；经常户外活动、定期体格测量。
  - 奶类：2～3 岁、4～5 岁都是每天 350～500 g（mL）。国家卫健委《儿童青少年肥胖食养指南（2024 年版）》同样写明 2～5 岁每天 350～500 mL 奶及奶制品。
  - 饮水：2～3 岁每天 600～700 mL，4～5 岁每天 700～800 mL。
  - 盐与烹调油：2～3 岁盐 <2 g、油 10～20 g；4～5 岁盐 <3 g、油 20～25 g。
  - 谷类：2～3 岁每天 75～125 g，4～5 岁每天 100～150 g；加餐以奶类、水果为主，配以少量松软面点。
- **备注**: 【部分核实：2022-06-01 发布时间、适用年龄、五条核心推荐和上述宝塔数值都来自多份转载（协会、医院、媒体），彼此一致；中国营养学会官网的学龄前专页 URL 未检索到】dg.cnsoc.org（中国居民膳食指南官网）和 www.cnsoc.org 是从检索到的两站文章/附件 URL 推出的根域名；第 3、4 个链接是转载页。注意不要和同期发布的《中国学龄儿童膳食指南（2022）》（6–17 岁，饮水 800–1400 mL、盐 <4 g 等）混用。添加糖：本指南的具体限量未核实；国家卫健委 2024 年食养指南和体重管理指导原则写明「2 岁及以上限制添加糖」「不喝含糖饮料」，可以一起引用（见候选 nhc-obesity-diet-2024）。宝塔里蔬菜、水果、肉蛋、大豆等其余分量未逐项核实。出版社（人民卫生出版社）据既有知识，未核实。

### 3-6岁儿童学习与发展指南

- **id**: `moe-3-6-guide` · **发布方**: 中华人民共和国教育部 · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — 教基二〔2012〕4号教育部通知的附件，属国家机关行政文件，依《著作权法》第五条不受著作权保护，可改写并注明出处。
- **获取**: robots 未确认
- **擅长**: 自理、情绪、语言、社交、游戏、安全
- **链接**: [1](http://www.moe.gov.cn/srcsite/A06/s3327/201210/t20121009_143254.html) · [2](https://www.cernet.edu.cn/html/e/etfz/health.shtml)
- **和 3–4 岁相关的要点**:
  - 生活自理（3–4 岁）：在帮助下能穿脱衣服或鞋袜；能将玩具和图书放回原处。教育建议：不论孩子做得好坏都给予肯定，不因做不好或做得慢而包办代替。
  - 情绪（3–4 岁）：情绪比较稳定，很少因一点小事哭闹不止；有比较强烈的情绪反应时，能在成人的安抚下逐渐平静下来。教育建议：孩子发脾气时不硬性压制，等冷静后再告诉他什么行为是可以接受的。
  - 卫生与用眼（3–4 岁）：愿意饮用白开水，不贪喝饮料；不用脏手揉眼睛，连续看电视等不超过 15 分钟（4–5 岁 20 分钟，5–6 岁 30 分钟）；在提醒下每天早晚刷牙、饭前便后洗手。
  - 作息与户外（3–6 岁通用教育建议）：保证每天睡 11～12 小时，其中午睡一般约 2 小时；每天户外活动一般不少于 2 小时，其中体育活动不少于 1 小时，季节交替时也要坚持。
  - 语言（3–4 岁）：能听懂日常会话；愿意表达自己的需要和想法，必要时能配以手势动作；能口齿清楚地说儿歌、童谣或复述简短的故事。教育建议：孩子表达时，成人可蹲下来、眼睛平视孩子，耐心听他把话说完。
  - 安全与同伴（3–4 岁）：不跟陌生人走，不吃陌生人给的东西；在提醒下注意安全，不做危险的事；在公共场所走失时，能向警察或有关人员说出自己和家长的名字等简单信息。在成人指导下不争抢、不独霸玩具；与同伴发生冲突时，能听从成人的劝解。
- **备注**: 【已核实：文号、moe.gov.cn 原文 URL（出现在检索结果中，也是期刊参考文献的引用地址），以及上述 3–4 岁条目（CERNET、北大附属幼儿园、各地政府转载、某幼儿园小班评价指标数据等多份文本交叉核对）】本环境无法解析 moe.gov.cn，页面未直接打开。正文落款 2012 年 9 月，官网页面路径日期为 2012-10-09。有一份检索摘要把「20 分钟」归到 3–4 岁，经核对 3–4 岁是 15 分钟、4–5 岁才是 20 分钟（2012-10 新浪报道及小班指标原文都是 15/20/30 分钟）。另外已核实的 3–4 岁数学条目：能手口一致地点数 5 个以内的物体，并能说出总数。走失条目各转载措辞略有出入（有的写「自己的名字、家庭地址、家长的名字或电话号码」），写 tip 时用意思、不用原句。指南整体立场：反对「小学化」和拔苗助长，以游戏和生活为主要学习方式，适合作为 play/discipline 类 tip 的价值依据。CERNET 链接只是健康领域分页镜像。

### 3岁以下婴幼儿健康养育照护指南（试行）

- **id**: `nhc-care-guide` · **发布方**: 国家卫生健康委员会办公厅（妇幼健康司发布） · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — 国卫办妇幼函〔2022〕409号，2022-11-19 成文、2022-11-28 发布。属国家机关具有行政性质的文件，依《著作权法》第五条不受著作权保护，可改写并注明出处（不得暗示官方背书）。期刊重刊版（《中国儿童保健杂志》2023;31(1):1-9，DOI 10.11852/zgetbjzz2022-1446）的排版受期刊权利约束，引用以政府文本为准。
- **获取**: robots 未确认
- **擅长**: 睡眠、饮食、游戏、自理、安全、家庭
- **链接**: [1](https://www.gov.cn/zhengce/zhengceku/2022-11/29/content_5729421.htm)
- **和 3–4 岁相关的要点**:
  - 屏幕：2 岁以内不建议观看或使用电子屏幕；2 岁以上每天累计不超过 1 小时，每次不超过 20 分钟。
  - 身体活动：幼儿每天身体活动至少 3 小时，其中户外活动至少 2 小时（雾霾、高温等天气可酌情减少户外时间）；避免连续久坐超过 1 小时。
  - 睡眠：幼儿期每天总睡眠 10～14 小时，夜间睡眠应达 8 小时以上；卧室室温以 20～25℃为宜，卧室不宜放置电视等视屏产品。
  - 进餐：进餐时不看电视、手机等电子产品；每次进餐 20 分钟左右，最长不宜超过 30 分钟，逐渐养成定时进餐习惯。
  - 结构：8 项基本理念（含重视早期全面发展、遵循生长发育规律、给予儿童恰当积极的回应、培养自主和自我调节能力、注重亲子陪伴和交流玩耍等）+ 6 方面咨询指导（生长发育监测、营养与喂养、交流与玩耍、生活照护、伤害预防、常见健康问题防控）；附件含《儿童心理行为发育问题预警征象筛查表》《婴幼儿亲子交流与玩耍要点》。
- **备注**: 【已核实：文号、日期、结构与上述数值，来源为检索索引中的多份政府/媒体转载全文】本环境 WebFetch/curl 无法解析或连接 gov.cn、nhc.gov.cn，所有页面均未直接打开。nhc.gov.cn 原发布页 URL 未在检索中找到；上面的 gov.cn 政策库地址取自一篇论文的参考文献，并出现在检索结果中。其他镜像（仅作人工核对用，未放进 urls，因为 harvest.py 会抓取 can_adapt 来源的全部 urls，而这些是 PDF 或媒体稿）：洛阳市卫健委转载 PDF https://oss.ly.gov.cn/lyswjw/Upload/xhe/day_230404/202304040534487296.pdf；gov.cn 政策解读 https://www.gov.cn/zhengce/2022-12/01/content_5729780.htm；人民网报道 https://health.people.com.cn/n1/2022/1129/c14739-32576782.html；期刊版 https://cjchc.xjtu.edu.cn/CN/10.11852/zgetbjzz2022-1446。适用 0–3 岁，正好覆盖 30–36 月龄；36 月龄以后改用 moe-3-6-guide / cn-preschool-exercise。睡眠口径注意：本指南写幼儿期 10–14 h，WS/T 579—2017（候选 nhc-sleep-ws579）写 3–5 岁 10–13 h。检索摘要还提到国家卫健委 2024 年发布《婴幼儿早期发展服务指南（试行）》，未核实。RSS/sitemap/robots 未能检查。

### 儿童心理保健技术规范

- **id**: `nhc-psych-norm` · **发布方**: 原卫生部办公厅（部分文献署名为国家卫生和计划生育委员会办公厅） · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — 是《关于印发儿童眼及视力保健等儿童保健相关技术规范的通知》（卫办妇社发〔2013〕26号，2013-04-09）的附件，属行政规范性文件，依《著作权法》第五条可改写并注明出处。配套解读论文《中国儿童心理行为发育问题预警征编制及释义》是期刊作品，受著作权保护，只能引用。
- **获取**: robots 未确认
- **擅长**: 语言、情绪、社交、游戏、自理
- **链接**: [1](https://www.nhc.gov.cn/wjw/gfxwj/201304/a60624807528402eb7f72ed447eb41dc/files/1733384819948_65252.doc) · [2](https://www.nhc.gov.cn/ewebeditor/uploadfile/2013/04/20130415103710390.doc) · [3](https://www.ecphf.cn/Women-children-health/2013-04/1731.htm)
- **和 3–4 岁相关的要点**:
  - 3 岁预警征象（出现任何一条都应登记并转诊评估）：不会说自己的名字；不会玩「拿棍当马骑」等假想游戏；不会模仿画圆；不会双脚跳。
  - 2 岁半预警征象：不会说 2～3 个字的短语；兴趣单一、刻板；不会示意大小便；不会跑（个别转载写作「走路经常跌倒」，以原文为准）。
  - 筛查表在 0～3 岁共设 8 个时点，每个时点 4 条；它是初筛工具，不是诊断。阳性者再用 DDST 等标准化量表进一步评估。
  - 规范目的：按儿童心理发展规律和不同年龄段的心理行为特征，定期做心理行为发育监测与评估，及早发现问题、及时干预。
- **备注**: 【已核实：文号、日期、nhc.gov.cn 附件链接，以及 3 岁/2 岁半条目（好大夫在线、搜狐、康复医院等多份转载交叉一致）】两个 nhc.gov.cn 链接是 .doc 附件（检索索引中可见，本环境无法下载），harvest.py 的 HTML 抽取器解析不了；第三个是华东公共卫生网的 HTML 全文转载。附件公告 PDF 镜像：国家心理健康和精神卫生防治中心 https://ncmhc.org.cn/channel/downfile/fmNSVkjeV。解读论文：https://cjchc.xjtu.edu.cn/CN/10.11852/zgetbjzz2018-26-01-32。4 岁一档（据记忆为：不会说带形容词的句子、不能按要求等待或轮流、不会独立穿衣、不会单脚站立）本次未核实，不要直接使用。好大夫在线的 3 岁档多列了一条「不能与其他儿童交流、游戏」，和「每个时点 4 条」对不上，可能混入了 2022 年孤独症筛查规范的扩展版（候选 nhc-asd-screening），以原文为准。写 tip 时只能用作「留意……，持续或明显时问儿科医生」式提醒，不下诊断，也不制造焦虑。

### 学龄前儿童（3～6岁）运动指南

- **id**: `cn-preschool-exercise` · **发布方**: 学龄前儿童（3～6岁）运动指南编制工作组（首都儿科研究所、北京体育大学、国家体育总局体育科学研究所联合组织；关宏岩、赵星、屈莎等） · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — 专家共识，2018 年 6 月以「专家共识版」发布，正式刊于《中国儿童保健杂志》2020;28(6):714-720，DOI 10.11852/zgetbjzz2020-0982。期刊论文受著作权保护，只能引用，数值用自己的话转述。
- **获取**: robots 未确认
- **擅长**: 游戏、家庭
- **链接**: [1](https://cjchc.xjtu.edu.cn/CN/10.11852/zgetbjzz2020-0982) · [2](http://health.people.com.cn/n1/2018/0611/c14739-30050371.html) · [3](http://medi-guide.meditool.cn/ymtpdf/F5AB1F83-3D84-D72A-6BD4-B65F2F7FA7F8.pdf)
- **和 3–4 岁相关的要点**:
  - 全天各类身体活动累计至少 180 分钟，其中中等及以上强度不少于 60 分钟（都是全天累计值，不必一次完成）。
  - 每天户外活动至少 120 分钟；雾霾、高温、高寒天气可酌情减少户外时间，但不应减少每日运动总量。
  - 屏幕时间每天累计不超过 60 分钟，越少越好；任何一次久坐都应限制在 60 分钟以内。
  - 既要有低强度活动，也要有中等及以上强度活动，并兼顾大肌肉和小肌肉动作发展；全文从运动原则、运动时间、运动类型、运动监测与评估 4 个方面提出 10 条推荐。
- **备注**: 【已核实：期刊出处、DOI、作者和单位、2018 年 6 月发布（人民网 2018-06-11 报道），以及上述数值】SPEC 写作「2018 专家共识」：2018 年发布共识版、2020 年期刊正式发表，两种说法都对。分级：它不是政府或学会文件，按 A/B 定义归为 B。不过核心数字（每天 3 h 活动、户外 2 h、屏幕 ≤1 h、久坐 ≤1 h）和国家卫健委 2022 照护指南、2024 食养指南一致，需要 A 级、可改写的依据时可以同时引用 nhc-care-guide。10 条推荐中关于具体运动类型的条目没有逐条核实。第三个链接是医脉通托管的 PDF 全文。

### 丁香妈妈

- **id**: `dxy-mama` · **发布方**: 丁香园（旗下母婴健康科普品牌） · **层级**: C 书籍/商业科普
- **使用方式**: 🔗 仅参考+链接 — 商业平台的原创内容，著作权归平台或作者所有，一般禁止转载。只能作背景参考，tip 不复制、不仿写其文章结构，basis 要追溯到它引用的一手指南。
- **获取**: robots 未确认
- **擅长**: 睡眠、饮食、安全、情绪、管教
- **链接**: [1](https://mama.dxy.com/)
- **备注**: 【未核实】URL 是既有知识中的域名；本环境的出口代理拒绝连接 mama.dxy.com（connect_rejected），所以可访问性、robots、RSS 都无法确认，检索额度用尽前也没来得及查。定位：医生审核的消费级科普，覆盖面广、可读性好，但属于二手综述。tip 的 basis 应写它引用的 AAP、WHO、国家卫健委等一手依据，不要把丁香妈妈本身当作依据。

### American Academy of Pediatrics — HealthyChildren.org（美国儿科学会（AAP）家长网站 HealthyChildren.org）

- **id**: `aap` · **发布方**: American Academy of Pediatrics · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — © American Academy of Pediatrics, all rights reserved. Terms of Use: only the Spanish version was seen (temporary single-copy download for informational use). The site's llms.txt (https://healthychildren.org/llms.txt) grants no copyright license. It allows automated systems to index, retrieve limited portions to answer a user, and cite/link the canonical page. It prohibits training or fine-tuning models on the content and republishing substantial portions.
- **获取**: robots 未确认
- **擅长**: 安全、自理、睡眠、饮食、语言、管教、家庭、游戏
- **核实状态**: 🟡 部分核实
- **链接**: [1](https://www.healthychildren.org/English/ages-stages/preschool/Pages/Developmental-Milestones-3-to-4-Year-Olds.aspx) · [2](https://www.healthychildren.org/English/ages-stages/toddler/toilet-training/Pages/The-Right-Age-to-Toilet-Train.aspx) · [3](https://www.healthychildren.org/English/family-life/Media/Pages/Helping-Kids-Thrive-in-a-Digital-World-AAP-Policy-Explained.aspx)
- **和 3–4 岁相关的要点**:
  - AAP policy statement 'Digital Ecosystems, Children, and Adolescents' (Pediatrics, Feb 2026, online 20 Jan 2026) replaces the 2016 'Media and Young Minds' guidance. It shifts focus from screen-time totals to the quality of interactions, a family media plan, and platform design (engagement-driven business models harm sleep, learning and mood). Reports disagree on whether any numeric limit for ages 2–5 remains, so don't cite 'AAP: 1 hour' as current.
  - AAP 0–5 explainer (2026): heavy solo screen use can affect language and social skills. What matters includes daily hours and whether adults use screens to soothe or entertain. Misuse is linked to poor sleep and less time for reading, moving and exploring.
  - Toilet training: signs of bladder/bowel control typically appear at 18–24 months. The average US start is 2–3 y. Readiness signs include staying dry for at least 2 h during the day or after naps. Pediatricians raise the topic at the 18-month, 2, 2½ and 3-year visits.
  - Developmental milestones at 3–4 y (AAP list): hops on one foot for up to ~5 s, uses stairs without support, pedals a tricycle, tells stories, dresses and undresses without help.
  - Family Media Plan (healthychildren.org/MediaUsePlan): an interactive tool that builds a per-child plan covering screen-free zones and times, device curfews, sleep and exercise.
- **备注**: All three URLs were confirmed in the search index. The AAP 2026 news release is at https://www.aap.org/en/news-room/news-releases/aap/2026/american-academy-of-pediatrics-calls-for-creating-a-child-friendly-digital-world/. The AAP 3–4 y milestone list (from 'Caring for Your Baby and Young Child') is not the same as the CDC 75%-threshold list. Prefer CDC for milestone claims. Superseded AAP 2016 advice (for context only): ages 2–5, ≤1 h/day of high-quality programming, co-viewed. Because llms.txt forbids training on this content, never put HealthyChildren text into fine-tuning or bulk-harvest pipelines. Use it as a reference and rewrite.

### AAP Bright Futures — 3 Year and 4 Year Visit parent handouts（美国儿科学会 Bright Futures 3 岁/4 岁体检家长指导）

- **id**: `aap-bright-futures` · **发布方**: American Academy of Pediatrics (Bright Futures National Center; HRSA-supported) · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — © American Academy of Pediatrics. The handouts carry an AAP copyright notice saying the AAP does not endorse modifications. No open license was found.
- **获取**: robots 未确认
- **擅长**: 安全、饮食、自理、家庭
- **核实状态**: 🟡 部分核实
- **链接**: [1](https://www.aap.org/en/practice-management/bright-futures/bright-futures-family-centered-care/well-child-visits-parent-and-patient-education/) · [2](https://www.aap.org/en/practice-management/bright-futures/bright-futures-materials-and-tools/bright-futures-tool-and-resource-kit/bright-futures-early-childhood-tools/)
- **和 3–4 岁相关的要点**:
  - Parent handouts exist for the 3 Year and 4 Year Visits, in English and Spanish, HTML and PDF (AAP page last updated 2024-05-30).
  - 4-year handout: brush twice a day with a pea-sized amount of fluoride toothpaste. Some versions add: spit, don't rinse with water.
  - Juice is not necessary. If given, no more than 4 oz (~120 ml) of 100% juice a day, always with a meal.
  - Car safety: stay in the forward-facing harnessed seat until the child reaches its weight/height limit (shoulders above the top harness slots, or ears at the top of the seat), then move to a belt-positioning booster. The back seat is safest until age 13.
  - Sun: sunscreen SPF 15 or higher. Avoid peak sun from 11 a.m. to 3 p.m.
- **备注**: The AAP index page was confirmed, but the exact official PDF URLs for the 3- and 4-year handouts were not surfaced. Handout content was confirmed from practice-hosted copies of the AAP handout, some of them practice-modified. Check the official AAP version before quoting exact wording. Bright Futures Guidelines 4th ed. is the underlying clinical reference.

### American Academy of Sleep Medicine — pediatric sleep duration consensus (2016)（美国睡眠医学会 AASM 儿童睡眠时长共识（2016））

- **id**: `aasm` · **发布方**: American Academy of Sleep Medicine (Journal of Clinical Sleep Medicine) · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — Not verified. Journal article © AASM, free to read. Facts (hour ranges) can be restated with citation.
- **获取**: robots 未确认
- **擅长**: 睡眠
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://jcsm.aasm.org/doi/10.5664/jcsm.5866)
- **和 3–4 岁相关的要点**:
  - 3–5 y: 10–13 h of sleep per 24 h including naps. CDC's preschool page uses the same range, so this is confirmed indirectly.
- **待核实要点**:
  - Paruthi et al. 2016 (J Clin Sleep Med 12(6)): 1–2 y need 11–14 h per 24 h including naps. Endorsed by the AAP.
  - Regularly sleeping fewer than the recommended hours is linked to attention, behavior and learning problems and to a higher risk of accidents and injuries.
- **备注**: DOI from prior knowledge, not re-verified.

### American Speech-Language-Hearing Association (ASHA) — communication milestones（美国言语听力协会 ASHA（语言发育里程碑））

- **id**: `asha` · **发布方**: American Speech-Language-Hearing Association · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — Not verified. Assume © ASHA, all rights reserved.
- **获取**: robots 未确认
- **擅长**: 语言、社交
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.asha.org/public/developmental-milestones/communication-milestones-3-to-4-years/)
- **待核实要点**:
  - ASHA updated its milestone pages by age band (2023). The 3–4 y band includes answering simple who/what/where questions, using pronouns and plurals, telling what happened during the day, and being understood by most listeners.
  - Learning two languages does not cause language delay. Families should keep using the home language(s).
  - If family members or strangers understand little of what a 3–4-year-old says, or she isn't combining words into sentences, ask about a speech-language evaluation.
- **备注**: Not checked (search budget exhausted). The URL comes from memory of the 2023 ASHA restructure. The older path was asha.org/public/speech/development/. Re-verify.

### CDC Learn the Signs. Act Early. — Milestones by 3 and 4 Years（美国疾控中心「早发现早行动」3 岁/4 岁发育里程碑）

- **id**: `cdc-milestones` · **发布方**: U.S. Centers for Disease Control and Prevention (National Center on Birth Defects and Developmental Disabilities) · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — US federal government work, mostly public domain. CDC 'Use of Agency Materials' (https://www.cdc.gov/other/agencymaterials.html): most CDC content can be used without permission. Attribution to CDC/HHS is required, with no implied endorsement. Exceptions: items carrying their own copyright statement (contractor or third-party material, e.g. some photos and videos). The CDC logo is a trademark.
- **获取**: API · robots 未确认
- **擅长**: 语言、社交、情绪、游戏、自理
- **核实状态**: ✅ 搜索索引中可见，要点与摘录一致
- **链接**: [1](https://www.cdc.gov/act-early/milestones/3-years.html) · [2](https://www.cdc.gov/act-early/milestones/4-years.html) · [3](https://www.cdc.gov/act-early/digital-online-checklist/4-years.html)
- **和 3–4 岁相关的要点**:
  - CDC/AAP 2022 revision: a milestone is something most children (75% or more) can do by that age. Missing one is a reason to talk to the doctor, not a diagnosis. The checklists do not replace validated developmental screening.
  - By 3 y (social/emotional): calms down within about 10 minutes after the caregiver leaves, e.g. at childcare drop-off.
  - By 3 y (language/cognitive): holds a conversation with at least 2 back-and-forth exchanges. Draws a circle when shown how.
  - By 4 y (social/language): pretends to be something else in play (teacher, superhero, dog). Comforts others who are hurt or sad, e.g. hugs a crying friend. Says sentences of 4 or more words. Says some words from a song, story or nursery rhyme.
  - By 4 y (cognitive/movement): names a few colors of items. Draws a person with 3 or more body parts. Catches a large ball most of the time. Serves self food or pours water with adult supervision. Holds a crayon or pencil between fingers and thumb.
  - Act early: if the child misses milestones, loses skills she had, or the parent has other concerns, talk to the doctor, ask about developmental screening and contact early intervention. Don't wait and see.
- **备注**: The 2024 restructure moved the pages to /act-early/milestones/{3-years,4-years}.html. The old URLs (/ncbddd/actearly/milestones/milestones-3yr.html, -4yr.html) are still in the index and presumably redirect. Live resolution could not be tested because direct fetch was blocked. The free CDC Milestone Tracker app (iOS/Android) and the digital checklists are linked from these pages. Third-party mirrors of 2025-era CDC pages exist at restoredcdc.org; cite cdc.gov. Content Services API v2 (JSON/XML, REST, no key mentioned) is listed on open.cdc.gov, but CDC notes that not all content is in its syndication catalog, and the syndication rules forbid altering syndicated content (adaptation must be your own rewrite). The full 3-y list also includes asks who/what/where/why questions, says first name when asked, puts on some clothes by herself, uses a fork, and avoids touching hot objects when warned (prior knowledge of the 2022 list; seen only in part this session).

### CDC Positive Parenting Tips (Toddlers 2–3, Preschoolers 3–5) & Essentials for Parenting Toddlers and Preschoolers（美国疾控中心 积极养育小贴士（学步儿/学龄前）与《幼儿养育要点》）

- **id**: `cdc-parenting` · **发布方**: U.S. Centers for Disease Control and Prevention · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — US federal government work, mostly public domain (same CDC 'Use of Agency Materials' policy: attribute CDC/HHS, no implied endorsement, third-party items excepted). Videos in Essentials for Parenting may include third-party footage, so check per item.
- **获取**: API · robots 未确认
- **擅长**: 管教、睡眠、安全、家庭、游戏、情绪
- **核实状态**: ✅ 搜索索引中可见，要点与摘录一致
- **链接**: [1](https://www.cdc.gov/child-development/positive-parenting-tips/preschooler-3-5-years.html) · [2](https://www.cdc.gov/child-development/positive-parenting-tips/toddlers-2-3-years.html) · [3](https://www.cdc.gov/parenting-toddlers/)
- **和 3–4 岁相关的要点**:
  - Preschoolers (3–5 y): 10–13 h of sleep per 24 h including naps. Keep TVs out of the bedroom. Limit screens to 1 h/day or less of quality programming and make a family media use plan. Offer age-appropriate play equipment (balls, plastic bats) and let the child choose what to play.
  - Toddlers (2–3 y): 11–14 h of sleep per 24 h including naps. Give attention and praise when the child follows instructions or behaves well, and limit attention for defiant behavior like tantrums. Teach acceptable ways to show she is upset.
  - Toddler safety: have her sit while eating and chew thoroughly to prevent choking. Check toys for loose parts. She rides in the back seat, buckled into a car seat with a harness.
  - Essentials for Parenting Toddlers and Preschoolers (ages 2–4, research-based): modules on communicating with your child, creating structure and rules, giving directions, discipline and consequences, and time-out. Includes free printable chore charts and daily schedules.
  - Time-out (CDC guide): give the warning only once, calmly and as a statement, not a question. If the behavior continues, use time-out. Afterwards the child still has to do what was asked. One version suggests about 1 minute per year of age.
  - Older CDC preschool handout (reproduced by state agencies): offer a limited number of simple choices (what to wear, which snack). When you say no, follow up with what she should do instead. Be consistent with discipline.
- **备注**: Time-out page: https://www.cdc.gov/parenting-toddlers/time-out/index.html. The old Essentials URLs (cdc.gov/parents/essentials/...) are superseded by /parenting-toddlers/. The '≤1 h/day' screen advice on the preschool page mirrors AAP 2016, which the AAP replaced in Jan 2026 (see 'aap'). WHO 2019 and the UK 2026 guidance still use ≤1 h for this age.

### Dialogic Reading (Whitehurst; PEER/CROWD) — What Works Clearinghouse（对话式阅读（Whitehurst，PEER/CROWD））

- **id**: `dialogic-reading` · **发布方**: Grover J. Whitehurst (Stony Brook Reading and Language Project); evidence review by the What Works Clearinghouse, U.S. Institute of Education Sciences · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — WWC intervention reports are US federal works (public domain). Third-party explainers such as Reading Rockets are copyrighted. Not re-verified this session.
- **获取**: robots 未确认
- **擅长**: 语言、游戏
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://ies.ed.gov/ncee/wwc/)
- **待核实要点**:
  - PEER sequence: Prompt the child to say something about the book, Evaluate her response, Expand it by rephrasing and adding information, Repeat the prompt so she uses the expansion.
  - CROWD prompt types: Completion (fill in the blank), Recall, Open-ended, Wh- questions, Distancing (link the book to her own life).
  - The WWC early-childhood intervention report (2007) found positive effects of dialogic reading on preschoolers' oral language.
  - Whitehurst et al. 1988 (Developmental Psychology): about a month of parent dialogic reading with 2-year-olds produced substantially larger expressive-language gains than typical reading.
- **备注**: Not checked (search budget exhausted). Only the WWC root is given because the intervention-report URL wasn't verified. IES/WWC operations were disrupted in 2025, so confirm the report is still online or use an archived copy.

### Center on the Developing Child at Harvard University（哈佛大学儿童发展中心（serve & return、执行功能、压力与韧性））

- **id**: `harvard-cdc` · **发布方**: Center on the Developing Child, Harvard University · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — © President and Fellows of Harvard College (assumed). A Terms and Conditions page exists at https://developingchild.harvard.edu/terms-and-conditions, but its text could not be read. A republisher reports that the Center asks reuse to be nonprofit and educational, with materials reproduced in full and unedited (secondhand, unverified).
- **获取**: robots 未确认
- **擅长**: 语言、情绪、游戏、家庭
- **核实状态**: ✅ 搜索索引中可见，要点与摘录一致
- **链接**: [1](https://developingchild.harvard.edu/resource-guides/guide-serve-and-return/) · [2](https://developingchild.harvard.edu/resources/handouts-tools/activities-guide-enhancing-and-practicing-executive-function-skills/) · [3](https://developingchild.harvard.edu/resource-guides/guide-toxic-stress/)
- **和 3–4 岁相关的要点**:
  - Serve and return, 5 steps: (1) notice the serve and share the child's focus, (2) return the serve by supporting and encouraging, (3) give it a name (label what she sees, does or feels), (4) take turns and wait, since waiting is crucial, (5) practice endings and beginnings.
  - Responsive back-and-forth exchanges with a caring adult shape brain architecture. When adult responses are inconsistent or absent, development can be disrupted.
  - Executive-function activities guide (2014): ages 3–5 are when EF capacities spike. Suitable activities include imaginary play, movement games with stop/start rules such as freeze dance, songs with repeated or cumulative verses, and counting-backward songs.
  - Three kinds of stress response: positive (brief, normal, e.g. a first day with a new caregiver), tolerable (more severe, but time-limited and buffered by supportive adults), and toxic (prolonged activation without supportive relationships, which can impair brain architecture). Responsive adult relationships buffer stress and build resilience.
- **备注**: The site seems to have been reorganized. The same resources appear under several path patterns (/resource-guides/..., /resources/handouts-tools/..., /key-concept/serve-and-return/, /science/key-concepts/toxic-stress/). Confirm the canonical path when the site can be reached. The 5-step wording was confirmed via faithful reproductions (Better Care Network and others) that cite the Center's PDF.

### Hirsh-Pasek & Golinkoff — playful / guided-play learning research（Hirsh-Pasek & Golinkoff 游戏化学习研究（《爱因斯坦从不用闪卡》《成为聪明人》））

- **id**: `hirsh-pasek` · **发布方**: Kathy Hirsh-Pasek (Temple University; Brookings) & Roberta Michnick Golinkoff (University of Delaware) · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — Peer-reviewed papers and books are copyrighted. Cite and paraphrase.
- **获取**: robots 未确认
- **擅长**: 游戏、语言
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://playfullearninglandscapes.com/)
- **待核实要点**:
  - Guided play (child-led, with adult scaffolding toward a learning goal) beat direct instruction for some early math and shape outcomes in a meta-analysis (Skene et al. 2022, Child Development).
  - The quality of conversational exchanges (shared attention, turn-taking, routines) at age 2 predicted later language beyond word quantity (Hirsh-Pasek et al. 2015, Psychological Science).
  - 'Becoming Brilliant' (2016) 6 Cs: collaboration, communication, content, critical thinking, creative innovation, confidence.
  - 'Einstein Never Used Flash Cards' (2003): play, not drilling with flashcards, builds early learning.
- **备注**: Not checked (search budget exhausted). Covers both research and books. For a book-only citation, reuse = book.

### NAEYC — For Families（全美幼教协会 NAEYC 家长资源）

- **id**: `naeyc` · **发布方**: National Association for the Education of Young Children · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — Not verified. © NAEYC, with reprint permission required.
- **获取**: robots 未确认
- **擅长**: 游戏、社交、语言、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.naeyc.org/our-work/families)
- **待核实要点**:
  - NAEYC's Developmentally Appropriate Practice position statement (4th ed., 2020): play is a central context for learning at 3–5. Adults scaffold within the child's zone of proximal development.
  - Family articles cover choosing a preschool, easing transitions to a new class, and supporting early math and literacy through everyday play.
- **备注**: Not checked (search budget exhausted).

### NHS (UK) — Best Start in Life / Start for Life, toddler behaviour, potty training, feeding（英国国家医疗服务体系 NHS（Best Start in Life / Start for Life））

- **id**: `nhs` · **发布方**: NHS England (nhs.uk); UK Department for Education and Department of Health and Social Care for the 2026 screen-time guidance · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — NHS website content can be republished by any organization under the Open Government Licence (v3.0), subject to the NHS website terms (https://www.nhs.uk/our-policies/terms-and-conditions/). Attribute as 'Information from the NHS website', with a date stamp if the copy isn't refreshed. Some content is excluded (https://www.nhs.uk/our-policies/terms-and-conditions/content-not-licensed-for-re-use/): certain named sub-sites, some images and videos, and third-party material.
- **获取**: API · robots 未确认
- **擅长**: 情绪、自理、饮食、语言、安全、睡眠
- **核实状态**: 🟡 部分核实
- **链接**: [1](https://www.nhs.uk/best-start-in-life/) · [2](https://www.nhs.uk/conditions/baby/babys-development/behaviour/temper-tantrums/) · [3](https://www.nhs.uk/baby/weaning-and-feeding/vitamins-for-children/)
- **和 3–4 岁相关的要点**:
  - Tantrums usually start around 18 months and are very common in toddlers. By age 4 they are far less common. They come from frustration at wanting to express themselves and finding it hard. Check whether she is tired or hungry. You may feel angry too, but show it in other ways. See the health visitor or GP if seriously concerned.
  - The UK government recommends daily vitamin A, C and D supplements for children aged 6 months to 5 years. Children aged 1–4 need a daily supplement containing 10 µg of vitamin D.
  - UK DfE/DHSC under-5 screen-time guidance (27 Mar 2026, on Best Start in Life): ages 2–5, try to keep it to 1 h/day or less. Under 2s, avoid screens except shared activities that encourage bonding and conversation. No screens at mealtimes or in the hour before bed. Prefer slow-paced, age-appropriate content and watch together. Avoid fast-paced short-form video and AI toys for young children.
  - Evidence behind the UK guidance (EYSTAG, Mar 2026): toddlers with about 5 h of daily screen time learned significantly fewer words. About 98% of 2-year-olds watch screens daily.
  - Start for Life now lives inside the Best Start in Life hub, which covers learning to talk, potty training (a five-step guide), healthy eating and activity for toddlers.
- **备注**: The tantrum page is past its review date (last reviewed 2022-11-07, next review due 2025-11-07), so re-check it. The NHS Website Content API is a free syndication programme. v1 is retired, so use v2, which needs developer-portal registration, acceptance of the terms, and a legal basis for use. The catalogue URL was seen only via mirrors of digital.nhs.uk. 2026 screen-time figures come from news reports (Irish Times 27 Mar 2026, Institute of Health Visiting) quoting the official tips. The official gov.uk/Best Start in Life page wasn't retrieved. The EYSTAG report PDF is at assets.publishing.service.gov.uk/media/69c53daf4a06660f085442a7/EYSTAG_report.pdf. UK CMO 2019 activity guidelines (pre-schoolers ≥180 min/day incl. ≥60 min MVPA) were not re-verified because the search budget ran out. Vitamin doses are evidence for basis only; SPEC forbids drug doses in tip bodies.

### WHO Guidelines on physical activity, sedentary behaviour and sleep for children under 5 years of age (2019)（世界卫生组织《5 岁以下儿童身体活动、久坐行为和睡眠指南》（2019））

- **id**: `who-under5` · **发布方**: World Health Organization · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — CC BY-NC-SA 3.0 IGO, per the WHO IRIS record (handle 10665/311664). Non-commercial reuse and adaptation are allowed with citation. Adaptations must carry the same or an equivalent license. No WHO logo. Numeric recommendations (facts) can be restated freely. If the app is monetized, don't reproduce WHO text verbatim and treat the source as cite-only.
- **获取**: robots 未确认
- **擅长**: 游戏、睡眠、自理、家庭
- **核实状态**: ✅ 搜索索引中可见，要点与摘录一致
- **链接**: [1](https://www.who.int/publications/i/item/9789241550536) · [2](https://iris.who.int/items/677e3562-1deb-48ec-bb1e-4bc8aad3ea09) · [3](https://www.who.int/news/item/24-04-2019-to-grow-up-healthy-children-need-to-sit-less-and-play-more)
- **和 3–4 岁相关的要点**:
  - Ages 3–4: at least 180 min/day in a variety of physical activities at any intensity, of which at least 60 min is moderate-to-vigorous, spread through the day. More is better.
  - Ages 3–4: not restrained for more than 1 h at a time (prams/strollers) and no long stretches of sitting. Sedentary screen time no more than 1 h/day, less is better.
  - When sedentary, reading and storytelling with a caregiver is encouraged.
  - Ages 3–4: 10–13 h of good-quality sleep, which may include a nap, with regular sleep and wake-up times.
  - Framing: the recommendations cover the whole 24-hour day (activity, restraint/screen time, sleep together), not separate targets.
- **备注**: The 3–4 y wording was confirmed against the WHO 24 Apr 2019 news release (PAHO reprint identical). ISBN 978-92-4-155053-6. The PDF can be downloaded from IRIS. For the child's current age (35 mo), WHO's '1–2 years' group applies until she turns 3 (prior knowledge, not re-verified: ≥180 min any intensity, screen ≤1 h for 2-year-olds, 11–14 h sleep incl. naps).

### WHO/UNICEF Nurturing Care Framework, WHO ECD & parenting guidelines, UNICEF Parenting（世卫组织/联合国儿基会「养育照护」框架与 UNICEF 育儿指导）

- **id**: `who-unicef-nurturing-care` · **发布方**: World Health Organization, UNICEF, World Bank Group (with PMNCH and ECDAN) · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — WHO guideline PDFs: CC BY-NC-SA 3.0 IGO. Confirmed for the 2020 ECD guideline (IRIS 10665/331306) and the 2022 parenting guideline (ISBN 978-92-4-006550-5). The license of the Framework document itself was not separately confirmed. UNICEF Parenting web pages: terms not found; treat as © UNICEF.
- **获取**: robots 未确认
- **擅长**: 家庭、情绪、游戏、饮食、安全
- **核实状态**: ✅ 搜索索引中可见，要点与摘录一致
- **链接**: [1](https://nurturing-care.org/about/what-is-the-nurturing-care-framework/) · [2](https://www.who.int/teams/social-determinants-of-health/violence-prevention/parenting-guidelines) · [3](https://www.unicef.org/asia-pacific/place-for-parents/positive-parenting-ages-0-5)
- **和 3–4 岁相关的要点**:
  - Nurturing Care Framework (launched 23 May 2018 at the 71st World Health Assembly): five indivisible components. They are good health, adequate nutrition, security and safety, responsive caregiving, and opportunities for early learning. Focus: pregnancy to age 3.
  - WHO 2020 ECD guideline: recommends responsive care and early-learning activities for caregivers of children in the first 3 years, built into nutrition interventions, plus psychosocial support for maternal mental health. Responsive caregiving means noticing, understanding and responding to the child's signals promptly and appropriately.
  - WHO 2022 guideline on parenting interventions (ages 0–17): recommends evidence-based parenting programs to reduce maltreatment and harsh parenting, strengthen the parent–child relationship, and reduce parental mental-health problems and child emotional/behavioral difficulties.
  - UNICEF positive parenting 0–5: treat crying and tantrums as communication. First check hunger, tiredness, discomfort or temperature. Repetitive behavior is how toddlers learn. Set clear expectations and repeat them, and don't push activities when the child is sleepy, hungry or tired.
- **备注**: Other useful URLs: https://nurturing-care.org/nurturing-care-components ; WHO 2022 parenting guideline https://www.who.int/publications-detail-redirect/9789240065505 ; WHO 2020 ECD guideline https://apps.who.int/iris/handle/10665/331306 ; UNICEF India 'Handling temper tantrums' https://www.unicef.org/india/parentingtips/handling-temper-tantrums. UNICEF China runs a Chinese-language positive-parenting site (https://www.unicef.cn/en/parenting-site/2023parenting), useful for zh terminology. Sources disagree on whether the 2020 ECD guideline has 3 or 4 recommendations, so don't cite a count.

### Child Mind Institute（美国儿童心智研究所 Child Mind Institute）

- **id**: `child-mind` · **发布方**: Child Mind Institute (US nonprofit) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. Assume © Child Mind Institute, all rights reserved.
- **获取**: robots 未确认
- **擅长**: 情绪、管教、社交、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://childmind.org/) · [2](https://childmind.org/article/how-to-handle-tantrums-and-meltdowns/)
- **待核实要点**:
  - Tantrums: stay calm, don't reward the tantrum with what was demanded, use planned ignoring of attention-seeking behavior (while keeping her safe), and give specific labeled praise for calm behavior.
  - Separation anxiety at preschool drop-off: keep goodbyes short and predictable, don't sneak out, and acknowledge the feeling while expressing confidence.
  - Its behavior guidance draws on parent management training and PCIT principles: catch good behavior, give clear commands, and follow through consistently.
- **备注**: Not checked (search budget exhausted). The URLs come from prior knowledge, and the article URL in particular should be re-verified.

### Gottman Emotion Coaching（Gottman 情绪辅导（Emotion Coaching））

- **id**: `gottman` · **发布方**: John Gottman (University of Washington research); The Gottman Institute · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. Book and Institute materials are copyrighted.
- **获取**: robots 未确认
- **擅长**: 情绪、家庭、管教
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.gottman.com/)
- **待核实要点**:
  - Five steps of emotion coaching ('Raising an Emotionally Intelligent Child', 1997): (1) notice the child's emotion, (2) see it as a chance for connection and teaching, (3) listen with empathy and validate, (4) help her name the feeling, (5) set limits while helping her solve the problem.
  - Meta-emotion research (Gottman, Katz & Hooven 1996, J Family Psychology): children of emotion-coaching parents showed better emotion regulation and peer relations.
  - Four parent styles toward emotions: dismissing, disapproving, laissez-faire (accepting with no limits), and emotion coaching (accepting feelings while limiting behavior).
- **备注**: Not checked (search budget exhausted). Only the root domain is given because the exact parenting-section URL is uncertain.

### The Hanen Centre（加拿大 Hanen 中心（幼儿语言发展））

- **id**: `hanen` · **发布方**: The Hanen Centre (Canadian nonprofit) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. Assume © Hanen Centre, all rights reserved. Program names such as 'It Takes Two to Talk' are trademarks.
- **获取**: robots 未确认
- **擅长**: 语言、游戏、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.hanen.org/)
- **待核实要点**:
  - OWL (Observe, Wait, Listen): follow the child's lead. Pause long enough for her to start or answer before you speak.
  - Respond to her words and expand them by adding a word or idea ('Car!' → 'A red car, going fast!'), and comment more than you question.
  - Shared book reading: let the child choose, follow her interest in the pictures, and turn the reading into a conversation.
- **备注**: Not checked (search budget exhausted). The articles section (hanen.org Helpful-Info/Articles) has the most useful parent pages. The exact path wasn't verified.

### Nemours KidsHealth — for parents（Nemours KidsHealth 儿童健康网）

- **id**: `kidshealth` · **发布方**: Nemours Children's Health (US nonprofit pediatric health system) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. © Nemours, all rights reserved. KidsHealth offers content licensing (prior knowledge).
- **获取**: robots 未确认
- **擅长**: 安全、自理、饮食、睡眠、语言
- **核实状态**: 🟡 部分核实
- **链接**: [1](https://kidshealth.org/en/parents/development-4yrs.html) · [2](https://kidshealth.org/en/parents/)
- **和 3–4 岁相关的要点**:
  - 'Your Child's Development: 4 Years': many 4-year-olds can unbutton bigger buttons.
- **待核实要点**:
  - Medically reviewed parent articles on preschool development, toilet training, picky eating, sleep problems, and common illnesses and injuries.
- **备注**: The development-4yrs URL was confirmed in the search index. The parents root and a 3-year development page (likely development-3yrs.html) were not checked.

### Parent-Child Interaction Therapy (PCIT) — PRIDE skills（亲子互动疗法 PCIT（PRIDE 技巧））

- **id**: `pcit` · **发布方**: PCIT International; developed by Sheila Eyberg (University of Florida) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. The PCIT protocol and coding manuals are copyrighted, but the PRIDE skills themselves are widely described in peer-reviewed literature. Rewrite in your own words and cite.
- **获取**: robots 未确认
- **擅长**: 管教、游戏、情绪、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.pcit.org/) · [2](https://www.cebc4cw.org/program/parent-child-interaction-therapy/)
- **待核实要点**:
  - PRIDE skills during child-directed play ('special time'): labeled Praise, Reflect her words, Imitate her play, Describe what she's doing, Enjoy (show enthusiasm).
  - During special time, avoid commands, questions and criticism, and let the child lead.
  - Home practice: about 5 minutes of special time daily.
  - Parent-directed phase: give direct, specific, positively stated commands one at a time, praise compliance, and follow a consistent time-out sequence for non-compliance.
  - Designed for ages ~2–7. The California Evidence-Based Clearinghouse (CEBC) rates it 'Well-Supported by Research Evidence'.
- **备注**: Not checked (search budget exhausted). The UC Davis PCIT training site (pcit.ucdavis.edu) is another candidate URL. Re-verify the CEBC rating and URLs.

### Raising Children Network (Australia) — preschoolers（澳大利亚 Raising Children Network 育儿网（学龄前））

- **id**: `raising-children-au` · **发布方**: Raising Children Network (Australia) Limited, funded by the Australian Government · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Copyright Raising Children Network. A Terms of Use page exists (URL seen in the search index), but its text was not retrieved. Assume all rights reserved.
- **获取**: robots 未确认
- **擅长**: 情绪、管教、游戏、社交、睡眠、饮食、自理、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://raisingchildren.net.au/preschoolers) · [2](https://raisingchildren.net.au/toddlers/behaviour/crying-tantrums/tantrums) · [3](https://raisingchildren.net.au/legal/terms-of-use)
- **待核实要点**:
  - Australian 24-Hour Movement Guidelines for the Early Years (2017), preschoolers 3–5: at least 180 min/day of physical activity including at least 60 min of energetic play. Not restrained for more than 1 h at a time. Sedentary screen time 1 h/day or less. 10–13 h of good-quality sleep.
  - Tantrums are common at 1–3 y. Stay calm, keep her safe, wait it out without giving in, then reconnect. Reduce triggers such as tiredness, hunger and abrupt transitions.
  - Preschool behavior: describe the behavior you want, praise it specifically, and use consistent, calm consequences.
- **备注**: Only the terms-of-use URL turned up in the search index. The other two URLs and all key points come from prior knowledge, because the search budget ran out before this id. Re-verify.

### Ellyn Satter Institute — Division of Responsibility in Feeding（Ellyn Satter 喂养责任分工）

- **id**: `satter` · **发布方**: Ellyn Satter Institute · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. © Ellyn Satter Institute. The Institute asks for exact attribution of the Division of Responsibility (prior knowledge).
- **获取**: robots 未确认
- **擅长**: 饮食、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.ellynsatterinstitute.org/)
- **待核实要点**:
  - Division of Responsibility: the parent decides what, when and where food is served. The child decides whether to eat and how much.
  - Toddlers and preschoolers do best with structured meals and sit-down snacks at predictable times, with nothing in between except water.
  - Avoid pressure ('one more bite'), bribes, praise for eating, and restriction. Include at least one food she usually eats at each meal.
- **备注**: Not checked (search budget exhausted). The Division of Responsibility page is under /how-to-feed/ on the Institute site. Exact path not verified.

### Triple P — Positive Parenting Program（Triple P 积极养育项目）

- **id**: `triple-p` · **发布方**: Triple P International / University of Queensland (Matthew Sanders) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. A proprietary licensed program, so assume all rights reserved.
- **获取**: robots 未确认
- **擅长**: 管教、家庭、情绪
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.triplep.net/) · [2](https://www.triplep-parenting.com/)
- **待核实要点**:
  - Five core principles: a safe, engaging environment; a positive learning environment; assertive discipline; realistic expectations; parents taking care of themselves.
  - Sanders et al. 2014 (Clinical Psychology Review) meta-analysis of about 100 studies: small-to-moderate improvements in child social, emotional and behavioral outcomes and in parenting practices.
  - Practical techniques include quality time in brief frequent moments, descriptive praise, planned ignoring of minor problem behavior, clear calm instructions, and logical consequences.
- **备注**: Not checked (search budget exhausted). Triple P is available in China through local providers. Re-verify.

### Vroom (brain-building tips)（Vroom 大脑建构日常小活动）

- **id**: `vroom` · **发布方**: Bezos Family Foundation · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — All Vroom content licenses are free but apply to noncommercial use only. A Basic License covers unaltered use under the Content and Attribution Guidelines. Adapting, translating or localizing needs a free A.L.T. (Adaptation, Localization, Translation) license, and adaptations and trademark use are reviewed by the Bezos Family Foundation (about 7–8+ weeks). Vroom content, logos or trademarks may not be used commercially.
- **获取**: robots 未确认
- **擅长**: 游戏、语言、家庭、情绪
- **核实状态**: ✅ 搜索索引中可见，要点与摘录一致
- **链接**: [1](https://www.vroom.org/) · [2](https://toolkit.vroom.org/module-1/brain-building-basics) · [3](https://www.vroom.org/for-professionals/adapting-vroom-content)
- **和 3–4 岁相关的要点**:
  - Five Brain Building Basics for caregivers: Look (notice what holds her attention), Chat (talk, sing or make sounds back and forth), Follow (tune into her words, sounds and ideas and respond), Stretch (extend the moment with open what/when/where/how/why questions), Take Turns (back-and-forth exchanges).
  - Over 1,000 free brain-building activity tips built around everyday routines (via the app and partner channels).
  - Same format as this app (short daily activity tips), so it is a useful benchmark for tone and length. Its tips must not be translated or adapted for a commercial app without a license.
- **备注**: Still active under Bezos Family Foundation funding: a Feb 2026 Child Care Aware of America release describes a 3-year BFF grant using Vroom. The Washington State DOH says its Vroom work moved to Washington Communities for Children (undated). Content must be at or below primary-school reading level, a useful style cue.

### ZERO TO THREE — resources for families（ZERO TO THREE（零到三岁）家长资源）

- **id**: `zero-to-three` · **发布方**: ZERO TO THREE (US nonprofit) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — All rights reserved. The Reprints & Permissions page (https://www.zerotothree.org/reprints-permissions) forbids reproducing, displaying or distributing materials in any format without express written consent. Individuals may make one personal non-commercial copy. Permission is per product, via a form or the Copyright Clearance Center for the Journal and Press, with up to 5 weeks to respond.
- **获取**: robots 未确认
- **擅长**: 情绪、管教、游戏、语言、家庭
- **核实状态**: 🟡 部分核实
- **链接**: [1](https://www.zerotothree.org/resources/for-families/) · [2](https://www.zerotothree.org/resource/tantrums/) · [3](https://www.zerotothree.org/resource/tantrums-defiance-aggression-oh-my)
- **和 3–4 岁相关的要点**:
  - Content covers pregnancy to age 3, so it applies mainly to 30–36 months. Use other sources for 3–4.5 y.
  - 'Tantrums, Defiance, Aggression – Oh My!' (Magic of Everyday Moments series): set age-appropriate limits while helping toddlers learn to cope with their emotions.
  - 'Responding to Challenging Behaviors' hub: guidance on biting and tantrums. The stated approach is to help parents build their own strategies rather than follow scripts.
- **待核实要点**:
  - Tantrums at this age usually come from limited language and self-regulation, not manipulation. Stay calm and present, name the feeling, and hold the limit.
  - Pretend play at 2–3 y builds language, self-regulation and empathy. Join in and follow the child's lead.
- **备注**: URLs and the permissions policy were confirmed in the search index. Only search snippets of the articles were available, so the evidence points here are thin. Spanish versions are under /espanol/.

### Thirty Million Words: Building a Child's Brain (Dana Suskind, 2015)（《父母的语言》Dana Suskind）

- **id**: `book-30m-words` · **发布方**: Dutton (Penguin Random House); TMW Center for Early Learning + Public Health, University of Chicago · **层级**: C 书籍/商业科普
- **使用方式**: 📕 书籍，仅参考 — Copyrighted book. The TMW Center offers free parent resources, but their terms weren't verified.
- **获取**: robots 未确认
- **擅长**: 语言、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://tmwcenter.uchicago.edu/)
- **待核实要点**:
  - The 3 Ts: Tune In (notice what she is focused on), Talk More (narrate and describe), Take Turns (back-and-forth conversation).
  - Conversational turns matter more than words heard: Romeo et al. 2018 (Psychological Science) linked turn-taking to Broca's-area activation, independent of SES.
  - Caveat: the original Hart & Risley '30-million-word gap' has been contested (e.g. Sperry et al. 2019). Frame tips around back-and-forth talk, not word counts.
- **备注**: URL from prior knowledge, not re-verified.

### The Gardener and the Carpenter (Alison Gopnik, 2016)（《园丁与木匠》Alison Gopnik）

- **id**: `book-gardener-carpenter` · **发布方**: Farrar, Straus and Giroux · **层级**: C 书籍/商业科普
- **使用方式**: 📕 书籍，仅参考 — Copyrighted book.
- **获取**: robots 未确认
- **擅长**: 游戏、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **待核实要点**:
  - Parenting as gardening, not carpentry: provide a safe, rich, stable environment rather than shaping the child to a blueprint.
  - Young children learn by exploring, imitating and pretend play. Pretending supports counterfactual thinking and theory of mind.
  - Direct instruction can narrow exploration: children shown one function of a toy explored it less (Bonawitz et al. 2011).
- **备注**: No URL verified (search budget exhausted).

### Good Inside: A Guide to Becoming the Parent You Want to Be (Becky Kennedy, 2022)（《Good Inside》Becky Kennedy）

- **id**: `book-good-inside` · **发布方**: Harper Wave (HarperCollins) · **层级**: C 书籍/商业科普
- **使用方式**: 📕 书籍，仅参考 — Copyrighted book. Good Inside is a commercial membership platform.
- **获取**: robots 未确认
- **擅长**: 情绪、管教、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.goodinside.com/)
- **待核实要点**:
  - Two things are true: you can hold a firm limit and also validate her disappointment.
  - Sturdy leadership: set boundaries as what you will do ('I won't let you hit') rather than trying to control the child's feelings.
  - Repair after ruptures: go back, take responsibility for your part, and reconnect. Repair matters more than never yelling.
- **备注**: URL from prior knowledge, not re-verified.

### How to Talk So Little Kids Will Listen: A Survival Guide to Life with Children Ages 2–7 (Joanna Faber & Julie King, 2017)（《如何说孩子才会听（2–7 岁版）》Faber & King）

- **id**: `book-how-to-talk` · **发布方**: Scribner (Simon & Schuster) · **层级**: C 书籍/商业科普
- **使用方式**: 📕 书籍，仅参考 — Copyrighted book.
- **获取**: robots 未确认
- **擅长**: 管教、情绪、语言、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **待核实要点**:
  - Acknowledge feelings before fixing: name the feeling, and grant in fantasy what you can't give in reality ('You wish we could stay at the park all day!').
  - Get cooperation with playfulness, choices, describing the problem instead of commanding, and a one-word reminder.
  - Alternatives to punishment: express your own feelings strongly without attacking her character, show how to make amends, take action to protect, and problem-solve together.
  - Praise descriptively (say what you see) instead of using global labels like 'good girl'.
- **备注**: No URL given because no publisher or author page was verified (search budget exhausted).

### Hunt, Gather, Parent (Michaeleen Doucleff, 2021)（《Hunt, Gather, Parent》Michaeleen Doucleff）

- **id**: `book-hunt-gather-parent` · **发布方**: Avid Reader Press (Simon & Schuster) · **层级**: C 书籍/商业科普
- **使用方式**: 📕 书籍，仅参考 — Copyrighted book.
- **获取**: robots 未确认
- **擅长**: 家庭、自理、管教、情绪
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **待核实要点**:
  - TEAM framework: Togetherness, Encouragement, Autonomy, Minimal interference.
  - Invite young children into real household work. Even imperfect help builds a habit of helping without being asked (Maya 'acomedido').
  - Respond to anger with calm, low-key reactions rather than yelling. Use stories and gentle drama (Inuit practice) to teach behavior.
- **备注**: Journalistic, anecdotal evidence (tier C). Pair with research sources in basis. No URL verified.

### The Montessori Toddler (Simone Davies, 2019)（《蒙特梭利幼儿教育》Simone Davies）

- **id**: `book-montessori-toddler` · **发布方**: Workman Publishing · **层级**: C 书籍/商业科普
- **使用方式**: 📕 书籍，仅参考 — Copyrighted book.
- **获取**: robots 未确认
- **擅长**: 自理、游戏、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.themontessorinotebook.com/)
- **待核实要点**:
  - Prepare the environment: child-sized, reachable tools such as a step stool, a low hook for her coat and a small pitcher, so she can do things herself.
  - Include her in practical life (pouring, wiping spills, food prep, dressing). 'Help me to do it by myself.'
  - Keep a few toys on an open shelf and rotate them, instead of large toy bins.
  - Observe and follow her interests. Allow time and don't rush in to help.
- **备注**: Author-site URL from prior knowledge, not re-verified.

### Positive Discipline for Preschoolers (Jane Nelsen, Cheryl Erwin, Roslyn Ann Duffy)（《正面管教（学龄前）》Nelsen）

- **id**: `book-positive-discipline` · **发布方**: Harmony / Three Rivers Press (Penguin Random House) · **层级**: C 书籍/商业科普
- **使用方式**: 📕 书籍，仅参考 — Copyrighted book.
- **获取**: robots 未确认
- **擅长**: 管教、家庭、自理
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.positivediscipline.com/)
- **待核实要点**:
  - Be kind and firm at the same time: connect before you correct.
  - Focus on solutions. Use logical consequences that are related, respectful and reasonable, rather than punishment.
  - Give limited choices and use routine charts the child helps make, so the chart is 'the boss'.
  - Positive time-out: a calm-down space the child helps design, used to regain calm, not as punishment.
- **备注**: Author-site URL from prior knowledge, not re-verified. Edition and year not verified.

### The Whole-Brain Child (2011) & No-Drama Discipline (2014) — Daniel J. Siegel & Tina Payne Bryson（《全脑教养法》《不打不骂》Siegel & Bryson）

- **id**: `book-whole-brain` · **发布方**: Delacorte Press / Bantam (Random House) · **层级**: C 书籍/商业科普
- **使用方式**: 📕 书籍，仅参考 — Copyrighted books.
- **获取**: robots 未确认
- **擅长**: 情绪、管教、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://drdansiegel.com/)
- **待核实要点**:
  - Connect and redirect: soothe and connect emotionally first, then guide behavior once she is calm.
  - Name it to tame it: help her tell the story of an upsetting event in words to make sense of it.
  - Upstairs/downstairs brain: during a meltdown the reasoning 'upstairs' brain is offline, so lecturing doesn't work.
  - Discipline means teaching: ask why she did it, what lesson you want to teach, and how best to teach it.
- **备注**: Author-site URL from prior knowledge, not re-verified.

## 候选来源（尚未用于 tip）

### 国家疾病预防控制局健康科普

- **id**: `ndcpa-kepu` · **发布方**: 国家疾病预防控制局 · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — 政府网站发布的科普文章不属于「具有立法、行政、司法性质的文件」，一般仍受著作权保护，只能引用并用自己的话转述。
- **获取**: robots 未确认
- **擅长**: 安全
- **链接**: [1](https://www.ndcpa.gov.cn/jbkzzx/c100040/common/content/content_1810957390935076864.html)
- **为什么值得加**: 国家级疾控科普，覆盖近视、伤害、传染病等主题，可作为 cn-myopia 和 cn-child-injury 的通俗版依据，比商业平台权威。
- **备注**: 【只核实了存在】检索结果中出现了该站文章「远离『小眼镜』系列之九：正确选择和使用电子产品」的标题和 URL。其他栏目（如防溺水、传染病）和 RSS 都未检查。

### 0～6岁儿童孤独症筛查干预服务规范（试行）

- **id**: `nhc-asd-screening` · **发布方**: 国家卫生健康委员会办公厅 · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — 国家卫健委办公厅通知的附件，属行政文件，依《著作权法》第五条可改写并注明出处。
- **获取**: robots 未确认
- **擅长**: 语言、社交
- **链接**: [1](https://wjw.xinjiang.gov.cn/hfpc/fybj/202209/a55d423cb0564d989fba1b3b3bc969c5.shtml)
- **为什么值得加**: 3 岁前后是孤独症早筛的关键节点。可以为 language/social 类「留意……，问儿科医生」式提醒提供 A 级依据，并和 nhc-psych-norm 的预警征象衔接。
- **备注**: 【只核实了存在】通知标题见新疆维吾尔自治区卫健委的转载页（2022 年 9 月）。文号和条款内容未核实，暂时没有可用要点。

### 0～6岁儿童眼保健及视力检查服务规范（试行）

- **id**: `nhc-eye-0-6` · **发布方**: 国家卫生健康委员会办公厅 · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — 国卫办妇幼发〔2021〕11号（2021-06-17），属行政文件，依《著作权法》第五条可改写并注明出处。
- **获取**: robots 未确认
- **擅长**: 安全
- **链接**: [1](http://www.nhc.gov.cn/fys/s7906/202106/15c5e7d23b3843daa3d87d2d7cebc3ce.shtml) · [2](http://big5.www.gov.cn/gate/big5/www.gov.cn/zhengce/zhengceku/2021-06/24/content_5620637.htm)
- **为什么值得加**: 可以直接写成提醒：36 月龄那次视力检查别漏掉，4 岁起每年做屈光筛查。补上 cn-myopia 在学龄前阶段缺的可执行细节。
- **和 3–4 岁相关的要点**:
  - 0～6 岁共安排 13 次眼保健和视力检查服务，其中 1～3 岁幼儿期 4 次，分别在 18、24、30、36 月龄。
  - 4、5、6 岁时，基层机构每年告知家长带孩子到县级妇幼保健机构或其他有条件的县级医疗机构做眼位检查、屈光筛查等专项检查。
  - 目的：早期发现儿童常见眼病、视力不良和远视储备量不足，并及时转诊干预。
- **备注**: 【已核实：nhc.gov.cn 原文页和 gov.cn 政策库（繁体网关）页面都出现在检索结果中，文号和日期见法律数据库】检索摘要里关于各年龄视力正常值的表述有混淆，未采用。

### 儿童青少年肥胖食养指南（2024年版）（含同期《体重管理指导原则（2024年版）》学龄前部分）

- **id**: `nhc-obesity-diet-2024` · **发布方**: 国家卫生健康委员会（据既有知识；本次未通过检索核实发文信息） · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — 如果确为国家卫健委办公厅印发的行政文件，适用《著作权法》第五条，可改写并注明出处。发文信息核实前，先按 can_adapt 登记。
- **获取**: robots 未确认
- **擅长**: 饮食、游戏、睡眠
- **为什么值得加**: 这是国家卫健委 2024 年的新文件，把学龄前的奶量、活动、睡眠、屏幕、添加糖、含糖饮料一并写进了行政文件。可以替代 cns-preschool-diet（学会作品，只能引用）和 cn-preschool-exercise（期刊共识），作为 A 级、可改写的依据。
- **和 3–4 岁相关的要点**:
  - 学龄前儿童（2～5 岁）每天摄入 350～500 mL 或相当量的奶及奶制品。
  - 学龄前儿童每天身体活动总时长应达 3 小时，其中户外活动至少 2 小时；每次久坐限制在 1 小时以内。
  - 建议 5 岁以下儿童每天睡眠 10～13 小时；学龄前儿童每天视屏时间不超过 1 小时，越少越好。
  - 零食首选奶及奶制品、新鲜蔬果、原味坚果，少吃高油、高盐、高糖的过度加工食品；不喝含糖饮料，足量饮用清洁的白水，少量多次。
  - 《体重管理指导原则（2024 年版）》学龄前部分：2 岁以下避免添加糖，2 岁及以上限制添加糖；1～5 岁每天至少 180 分钟多样化身体活动，以户外为主；2～5 岁每天使用电子产品少于 1 小时。
- **备注**: 【部分核实】已读取全文副本（第三方 GitHub 仓库里由 PDF 转成的 Markdown，标题为「儿童青少年肥胖食养指南(2024年版)」，同一文件还收录了「体重管理指导原则 (2024年版)」），上述条目均出自该文本。发文机关、文号和官方 URL 因检索额度用尽未核实，所以 urls 暂空。注意它的定位是肥胖防控食养，不是一般人群指南：可以引用通用数值（奶量、活动、睡眠、屏幕、含糖饮料），不要引用针对肥胖儿童的食养方和中医辨证内容。

### 0岁～5岁儿童睡眠卫生指南（WS/T 579—2017）

- **id**: `nhc-sleep-ws579` · **发布方**: 国家卫生和计划生育委员会（卫生行业标准） · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — 推荐性卫生行业标准，由国家卫生主管部门以公告形式发布，一般视为具有行政性质的规范文件（第五条）。但推荐性标准是否受著作权保护在实践中有争议，保守做法是改写并注明出处，不整段复制。
- **获取**: robots 未确认
- **擅长**: 睡眠
- **链接**: [1](https://www.nhc.gov.cn/ewebeditor/uploadfile/2017/10/20171026154305316.pdf)
- **为什么值得加**: 这是国内唯一的国家级学龄前睡眠卫生标准。sleep 分类目前只有 AASM、WHO 两个国际来源，补上一个国内 A 级依据，还能给出 3 岁前后切换的睡眠时长口径。
- **和 3–4 岁相关的要点**:
  - 推荐 24 小时总睡眠：1～2 岁 11～14 小时；3～5 岁 10～13 小时。
  - 睡眠环境：卧室空气流通、温度适宜；可以开小夜灯，入睡后关灯；卧室不放电视、手机、游戏机（据二手摘要，原文措辞待核）。
  - 和 nhc-care-guide（幼儿期 10～14 小时）口径略有不同；国家卫健委《儿童青少年肥胖食养指南（2024 年版）》采用「5 岁以下 10～13 小时」。
- **备注**: 【已核实：nhc.gov.cn 上的 PDF 链接和标题（检索结果），3～5 岁 10～13 h 出现在原文检索片段里；1～2 岁 11～14 h 来自转载摘要】发布和实施日期未核实（附件路径为 2017/10）。链接是 PDF，harvest.py 的 HTML 抽取器解析不了。

### 中国儿童保健杂志

- **id**: `cjchc` · **发布方**: 《中国儿童保健杂志》编辑部（主办单位未核实） · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — 期刊论文受著作权保护，只能引用（文中转载的政府文件正文本身不受保护）。
- **获取**: robots 未确认
- **擅长**: 语言、情绪、游戏、睡眠、饮食、安全、社交
- **链接**: [1](https://cjchc.xjtu.edu.cn/CN/10.11852/zgetbjzz2022-1446) · [2](https://cjchc.xjtu.edu.cn/CN/10.11852/zgetbjzz2020-0982) · [3](https://cjchc.xjtu.edu.cn/CN/10.11852/zgetbjzz2018-26-01-32) · [4](https://cjchc.xjtu.edu.cn/CN/10.11852/zgetbjzz2023-0543)
- **为什么值得加**: 这是国内儿童保健领域的核心期刊，国家指南的正式期刊版、专家共识和本土研究多在这里首发，适合给审核人追溯 basis 的学术出处。
- **和 3–4 岁相关的要点**:
  - 刊载国家卫健委《3 岁以下婴幼儿健康养育照护指南（试行）》（2023;31(1):1-9，DOI 10.11852/zgetbjzz2022-1446）。
  - 刊载《学龄前儿童（3～6 岁）运动指南》（2020;28(6):714-720，DOI 10.11852/zgetbjzz2020-0982）。
  - 刊载《中国儿童心理行为发育问题预警征编制及释义》（DOI 10.11852/zgetbjzz2018-26-01-32）和《婴幼儿养育照护专科建设专家共识》（DOI 10.11852/zgetbjzz2023-0543）。
- **备注**: 【已核实：上述 DOI 页面都出现在检索结果中】主办单位（据记忆为中华预防医学会和西安交通大学）、是否开放获取、是否有 RSS 都未核实。

### 婴幼儿养育照护专家共识

- **id**: `nurturing-care-consensus` · **发布方**: 执笔：邵洁、童梅玲、张悦、汪鸿、易彬、许培斌（发布学会未核实，推测为中华预防医学会儿童保健分会） · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — 专家共识，受著作权保护，只能引用。
- **获取**: robots 未确认
- **擅长**: 家庭、游戏、情绪、语言
- **链接**: [1](http://www.jcme.org.cn/update/2021/06/%E5%A9%B4%E5%B9%BC%E5%84%BF%E5%85%BB%E8%82%B2%E7%85%A7%E6%8A%A4%E4%B8%93%E5%AE%B6%E5%85%B1%E8%AF%86.pdf)
- **为什么值得加**: 它是国家照护指南的学术底稿之一，是回应性照护、亲子交流与玩耍等 family/play 主题的本土专家共识。
- **备注**: 【只核实了存在】检索结果显示了这份 PDF 和执笔人名单；发布学会、年份、期刊出处和内容要点都未核实。如果核实为学会发布，tier 可升为 A。

### American Academy of Pediatric Dentistry (AAPD)（美国儿童牙科学会 AAPD）

- **id**: `aapd` · **发布方**: American Academy of Pediatric Dentistry · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — Not verified.
- **获取**: robots 未确认
- **擅长**: 安全、自理、饮食
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.aapd.org/)
- **为什么值得加**: Authoritative for oral-health tips (the SPEC safety category includes 口腔). The child turns 3 within the app window, when toothpaste amount changes.
- **待核实要点**:
  - Use a smear (rice-grain amount) of fluoride toothpaste under age 3 and a pea-sized amount at ages 3–6. Parents brush or supervise twice daily.
  - First dental visit by the first birthday, then regular check-ups. Limit sugary drinks and frequent snacking.
- **备注**: Not checked (search budget exhausted).

### Canadian Paediatric Society — Caring for Kids（加拿大儿科学会 家长网站 Caring for Kids）

- **id**: `cps-caring-for-kids` · **发布方**: Canadian Paediatric Society · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — Not verified.
- **获取**: robots 未确认
- **擅长**: 安全、睡眠、自理、饮食、语言
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://caringforkids.cps.ca/)
- **为什么值得加**: A second national pediatric society for cross-checking AAP claims, especially now that AAP has dropped numeric screen limits.
- **待核实要点**:
  - CPS 2017 screen-time position: for ages 2–5, limit routine sedentary screen time to under 1 h/day, and avoid screens for 1 h before bed.
  - Parent pages on toilet learning, sleep, and safety.
- **备注**: Not checked (search budget exhausted).

### Healthy Drinks, Healthy Kids — 2019 beverage recommendations for ages 0–5 (Healthy Eating Research)（0–5 岁儿童饮品共识建议（Healthy Eating Research 2019，AAP/AAPD/AND/AHA））

- **id**: `healthy-drinks-healthy-kids` · **发布方**: Healthy Eating Research (Robert Wood Johnson Foundation), with the American Academy of Pediatrics, American Academy of Pediatric Dentistry, Academy of Nutrition and Dietetics, and American Heart Association · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — Not verified.
- **获取**: robots 未确认
- **擅长**: 饮食、安全
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://healthydrinkshealthykids.org/)
- **为什么值得加**: A multi-society consensus with specific numbers for one of the most common food questions (juice, milk, yogurt drinks).
- **待核实要点**:
  - Ages 2–5: water and plain milk are the main drinks. Limit 100% juice to about 4 oz/day at 2–3 y and 4–6 oz/day at 4–5 y. Avoid sugar-sweetened drinks, flavored milks, plant milks other than fortified soy (unless medically indicated), toddler formulas, and caffeinated or low-calorie sweetened drinks.
- **备注**: Not checked (search budget exhausted). Re-verify the juice and milk ounce figures before use.

### USDA MyPlate — Preschoolers（美国农业部 MyPlate 学龄前儿童饮食）

- **id**: `myplate-preschoolers` · **发布方**: U.S. Department of Agriculture · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — US federal government work, public domain (general principle). Page-specific terms not verified.
- **获取**: robots 未确认
- **擅长**: 饮食、自理
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.myplate.gov/life-stages/preschoolers)
- **为什么值得加**: Public-domain feeding tips (kitchen helper tasks by age, picky-eater strategies) that can be adapted.
- **待核实要点**:
  - Preschooler pages cover food-group amounts by age, picky eating, letting kids help in the kitchen, and modeling healthy eating.
- **备注**: Not checked (search budget exhausted). US dietary guidance was revised for 2025–2030, so check that the MyPlate pages reflect it.

### NHTSA — Car Seats and Booster Seats（美国国家公路交通安全管理局 儿童安全座椅指南）

- **id**: `nhtsa-car-seats` · **发布方**: U.S. National Highway Traffic Safety Administration · **层级**: A 政府/国际组织/学术
- **使用方式**: ✅ 可改写引用 — US federal government work, public domain (general principle). Page-specific terms not verified.
- **获取**: robots 未确认
- **擅长**: 安全
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.nhtsa.gov/vehicle-safety/car-seats-and-booster-seats)
- **为什么值得加**: Public-domain, precise car-seat staging. Relevant to Chinese families, where car-seat use is still inconsistent.
- **待核实要点**:
  - Keep the child in a forward-facing seat with harness and tether until she reaches the seat maker's top height or weight limit, then a booster. All children under 13 ride in the back seat.
- **备注**: Not checked (search budget exhausted).

### Parenting for Lifelong Health (PLH)（终身健康育儿项目 PLH（WHO/UNICEF/牛津大学））

- **id**: `parenting-for-lifelong-health` · **发布方**: University of Oxford, University of Cape Town, WHO, UNICEF and partners · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — Believed to be open-access under a Creative Commons NonCommercial license (prior knowledge). Not verified.
- **获取**: robots 未确认
- **擅长**: 管教、家庭、情绪、游戏
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.parentingforlifelonghealth.org/)
- **为什么值得加**: Evidence-based, WHO/UNICEF-backed, and possibly open-licensed, with existing Chinese translations. It may be the most reusable high-tier parenting content.
- **待核实要点**:
  - Free, non-commercial parenting programmes tested in randomized trials in low- and middle-income countries. Includes PLH for Young Children (ages 2–9).
  - Its COVID-19 parenting tip sheets (with WHO/UNICEF) cover one-on-one time, keeping it positive, structure, and handling misbehavior, and were translated into 100+ languages, including Chinese.
- **备注**: Not checked (search budget exhausted). Verify the URL and exact license.

### National Center for Pyramid Model Innovations (NCPMI) — Backpack Connection Series（美国金字塔模型创新中心 NCPMI（家长「书包联络」系列））

- **id**: `pyramid-model` · **发布方**: NCPMI, University of South Florida; funded by U.S. Dept. of Education, Office of Special Education Programs · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — Not verified. Federally funded, and older CSEFEL/TACSEI materials said reproduction was encouraged and permission to copy was not required (prior knowledge). Confirm for current NCPMI materials.
- **获取**: robots 未确认
- **擅长**: 情绪、管教、社交、自理
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://challengingbehavior.org/)
- **为什么值得加**: Free, short, practical one-skill family handouts on exactly this app's emotion/discipline/social categories, possibly with a permissive license.
- **待核实要点**:
  - Backpack Connection handouts give families one strategy at a time for ages 0–5, e.g. teaching feeling words, calming down with 'turtle technique', using visual schedules, and giving positive directions.
  - The Pyramid Model is a tiered framework for social-emotional competence in early childhood: nurturing relationships, then supportive environments, then targeted skill teaching.
- **备注**: Not checked (search budget exhausted).

### Yale Parenting Center — Alan Kazdin (Kazdin Method / Everyday Parenting)（耶鲁育儿中心（Kazdin 方法））

- **id**: `yale-parenting-center` · **发布方**: Yale University (Alan E. Kazdin) · **层级**: A 政府/国际组织/学术
- **使用方式**: 🔗 仅参考+链接 — Not verified. Coursera course and books are copyrighted.
- **获取**: robots 未确认
- **擅长**: 管教、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.coursera.org/learn/everyday-parenting)
- **为什么值得加**: Clear behavior-science scripts (positive opposites, 'practice games') that fit 160-character tips well.
- **待核实要点**:
  - ABC approach: Antecedents (clear prompts, choices), Behavior (shape the positive opposite of the problem behavior, use practice and pretend 'games'), Consequences (effusive, specific praise right away; mild, brief punishment only).
  - Punishment alone suppresses behavior temporarily. Building the desired behavior through repetition and praise produces lasting change.
- **备注**: Not checked (search budget exhausted). The Yale Parenting Center's own page URL wasn't verified, so only the Coursera course URL (prior knowledge) is given.

### The Incredible Years (Carolyn Webster-Stratton)（Incredible Years 不可思议的岁月 家长培训项目）

- **id**: `incredible-years` · **发布方**: The Incredible Years, Inc. · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. Proprietary licensed program.
- **获取**: robots 未确认
- **擅长**: 管教、游戏、情绪、社交
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://incredibleyears.com/)
- **为什么值得加**: Pairs with PCIT/Triple P as a third major evidence-based program, and is especially strong on child-directed play and coaching scripts.
- **待核实要点**:
  - Parenting 'pyramid': a large base of child-directed play, praise and coaching (academic, persistence, social, emotion coaching), with limit-setting, ignoring and time-out used sparingly at the top.
  - Preschool BASIC program (ages 3–6) is rated well-supported by multiple RCTs for reducing conduct problems.
- **备注**: Not checked (search budget exhausted).

### Reach Out and Read（Reach Out and Read（儿科门诊早期阅读项目））

- **id**: `reach-out-and-read` · **发布方**: Reach Out and Read, Inc. (US nonprofit) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. Assume all rights reserved.
- **获取**: robots 未确认
- **擅长**: 语言、游戏、家庭
- **核实状态**: 🟡 部分核实
- **链接**: [1](https://reachoutandread.org/)
- **为什么值得加**: Clinic-based, AAP-aligned literacy promotion. Good for read-aloud tips and for tracking AAP media guidance.
- **和 3–4 岁相关的要点**:
  - Active in 2026: it published a Feb 2026 podcast transcript on the new AAP digital-media guidance (Dr. Tiffany Munzer) and hosts summaries of AAP media research (e.g. Moreno et al. 2024).
- **待核实要点**:
  - Pediatric clinicians give a new book and model shared reading at well-child visits from 6 months to 5 years. Studies link participation to more frequent home reading and better receptive/expressive vocabulary.
- **备注**: Domain confirmed in the search index (reachoutandread.org/moreno_2024 and a 2026 PDF under /wp-content/uploads/2026/02/). Program evidence not re-verified.

### Ready4K / ParentPowered — text-message parenting tips（Ready4K 短信育儿提示（ParentPowered））

- **id**: `ready4k` · **发布方**: ParentPowered (formerly Ready4K); research by Susanna Loeb, Benjamin York et al. · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. Proprietary program content.
- **获取**: robots 未确认
- **擅长**: 语言、游戏、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://www.parentpowered.com/)
- **为什么值得加**: The closest evidence that short, regular push tips to parents of 3–5-year-olds change behavior. Use it as a design reference and cite it as the evidence base for the app format.
- **待核实要点**:
  - York, Loeb & Doss 2019 (Journal of Human Resources): a year of weekly text messages to parents of preschoolers increased home literacy activities and improved some early literacy outcomes.
  - Its message design (a fact, then a concrete tip, then a growth message) maps directly onto this app's 'why + what to do + script' structure.
- **备注**: Not checked (search budget exhausted). The rename from Ready4K to ParentPowered and its URL need checking.

### Sesame Workshop — Sesame Street in Communities parent resources（芝麻街工作室 家长资源（Sesame Street in Communities））

- **id**: `sesame-workshop` · **发布方**: Sesame Workshop (US nonprofit) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. Characters are trademarked and content is copyrighted.
- **获取**: robots 未确认
- **擅长**: 情绪、社交、自理、家庭、安全
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://sesameworkshop.org/) · [2](https://sesamestreetincommunities.org/)
- **为什么值得加**: Strong on transitions (starting school, a new sibling) and emotion vocabulary for 3–5-year-olds, with research-backed content development.
- **待核实要点**:
  - Free bilingual parent resources on routines, big feelings, starting school, new siblings, and coping with stress or grief, developed with child-development advisers.
- **备注**: Not checked (search budget exhausted). The Sesame Street in Communities domain may have been folded into sesameworkshop.org.

### Talking is Teaching: Talk, Read, Sing (Too Small to Fail)（「说话就是教育」Talking is Teaching（Too Small to Fail））

- **id**: `talking-is-teaching` · **发布方**: Too Small to Fail (Clinton Foundation) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified.
- **获取**: robots 未确认
- **擅长**: 语言、家庭
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://talkingisteaching.org/)
- **为什么值得加**: Ready-made 'everyday moment' language prompts that fit Chinese family routines when rewritten.
- **待核实要点**:
  - Public-awareness campaign with simple talk/read/sing prompts tied to everyday places such as the grocery store, laundromat and bath time.
- **备注**: Not checked (search budget exhausted). Campaign activity level in 2026 unknown.

### The Basics (Boston Basics) — five evidence-based parenting practices（The Basics 五个养育基本功（源自哈佛成就差距研究所））

- **id**: `the-basics` · **发布方**: The Basics, Inc. (originated at Harvard's Achievement Gap Initiative, Ronald Ferguson) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. Materials are offered free to communities (prior knowledge).
- **获取**: robots 未确认
- **擅长**: 语言、游戏、家庭、情绪
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://thebasics.org/)
- **为什么值得加**: Simple, memorable framework with free short videos. Its 'Count, Group, and Compare' category fills an early-math gap in the current list.
- **待核实要点**:
  - Five Basics for ages 0–3: Maximize Love, Manage Stress; Talk, Sing, and Point; Count, Group, and Compare; Explore Through Movement and Play; Read and Discuss Stories.
- **备注**: Not checked (search budget exhausted).

### Tools of the Mind（Tools of the Mind 心智工具（维果茨基式游戏课程））

- **id**: `tools-of-the-mind` · **发布方**: Tools of the Mind (Elena Bodrova & Deborah Leong) · **层级**: B 循证项目/权威机构
- **使用方式**: 🔗 仅参考+链接 — Not verified. Proprietary curriculum.
- **获取**: robots 未确认
- **擅长**: 游戏、情绪
- **核实状态**: ⚪ 未核实（凭已有知识，待复核）
- **链接**: [1](https://toolsofthemind.org/)
- **为什么值得加**: Turns EF research into home-adaptable play tactics (play plans, buddy reading, 'freeze' games) for 3–5-year-olds.
- **待核实要点**:
  - A Vygotskian curriculum built on mature make-believe play with 'play plans' (the child says or draws what she will play before starting), which builds self-regulation.
  - Diamond et al. 2007 (Science) reported executive-function benefits. Later large RCTs show mixed results, so frame claims modestly.
- **备注**: Not checked (search budget exhausted).
