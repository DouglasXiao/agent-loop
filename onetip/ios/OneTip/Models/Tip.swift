import Foundation

/// One parenting tip, decoded from `tips.bundle.json` (schema: onetip/data/SPEC.md).
struct Tip: Codable, Identifiable, Hashable {
    let id: String
    let category: TipCategory
    let title: String
    let body: String
    let ageMinMonths: Int
    let ageMaxMonths: Int
    let tags: [String]
    let sourceIds: [String]
    let basis: String

    enum CodingKeys: String, CodingKey {
        case id, category, title, body, tags, basis
        case ageMinMonths = "age_min_months"
        case ageMaxMonths = "age_max_months"
        case sourceIds = "source_ids"
    }

    func fits(ageInMonths age: Int, slack: Int = 0) -> Bool {
        ageMinMonths - slack <= age && age <= ageMaxMonths + slack
    }
}

enum TipCategory: String, Codable, CaseIterable {
    case language, emotion, discipline, play, selfcare, sleep, food, safety, social, family

    var displayName: String {
        switch self {
        case .language: "语言与沟通"
        case .emotion: "情绪与情商"
        case .discipline: "规则与正面管教"
        case .play: "游戏与学习"
        case .selfcare: "生活自理"
        case .sleep: "睡眠"
        case .food: "饮食营养"
        case .safety: "健康安全"
        case .social: "社交与入园"
        case .family: "亲子与家庭"
        }
    }

    var emoji: String {
        switch self {
        case .language: "💬"
        case .emotion: "🌦️"
        case .discipline: "🧭"
        case .play: "🧩"
        case .selfcare: "👟"
        case .sleep: "🌙"
        case .food: "🥕"
        case .safety: "🛡️"
        case .social: "🤝"
        case .family: "🏡"
        }
    }
}

/// A citation target shown under "依据" (from `sources.bundle.json`).
struct TipSource: Codable, Identifiable, Hashable {
    let id: String
    let name: String
    let url: URL?
}
