import Foundation

/// Pure, deterministic tip selection. No I/O so it can be unit tested and
/// used to pre-plan future days for local notifications.
///
/// Rules, in order:
/// 1. Never pick a tip the parent marked 不适合 (disliked).
/// 2. Prefer tips whose age window contains the child's age; widen by 3, then
///    6 months only if nothing fits.
/// 3. Prefer tips never shown. Once everything is shown, recycle the least
///    recently shown third.
/// 4. Rotate categories: pick from the category shown longest ago, so a week
///    covers emotions, play, sleep, ... instead of five discipline tips in a row.
/// 5. Within that category, pick pseudo-randomly from `seed`, so the same day
///    always yields the same tip.
struct TipPicker {
    let tips: [Tip]

    func next(ageInMonths age: Int, history: [String], disliked: Set<String>, seed: UInt64) -> Tip? {
        let allowed = tips.filter { !disliked.contains($0.id) }
        guard !allowed.isEmpty else { return nil }

        var ageFit: [Tip] = []
        for slack in [0, 3, 6] {
            ageFit = allowed.filter { $0.fits(ageInMonths: age, slack: slack) }
            if !ageFit.isEmpty { break }
        }
        if ageFit.isEmpty { ageFit = allowed }

        var lastShown: [String: Int] = [:]
        for (index, id) in history.enumerated() { lastShown[id] = index }

        var pool = ageFit.filter { lastShown[$0.id] == nil }
        if pool.isEmpty {
            let byAge = ageFit.sorted { lastShown[$0.id, default: -1] < lastShown[$1.id, default: -1] }
            pool = Array(byAge.prefix(max(1, byAge.count / 3)))
        }

        let categoryOf = Dictionary(tips.map { ($0.id, $0.category) }, uniquingKeysWith: { first, _ in first })
        var categoryLastShown: [TipCategory: Int] = [:]
        for (index, id) in history.enumerated() {
            if let category = categoryOf[id] { categoryLastShown[category] = index }
        }

        var rng = SplitMix64(seed: seed)
        let categories = Set(pool.map(\.category)).sorted { $0.rawValue < $1.rawValue }
        let oldest = categories.map { categoryLastShown[$0, default: -1] }.min() ?? -1
        let stalest = categories.filter { categoryLastShown[$0, default: -1] == oldest }
        guard let category = stalest.randomElement(using: &rng) else { return nil }

        let candidates = pool.filter { $0.category == category }.sorted { $0.id < $1.id }
        return candidates.randomElement(using: &rng)
    }

    /// Plans one tip per day for `days` consecutive days, as if each were read
    /// on its day. Used to put the real tip title into scheduled notifications.
    func plan(
        from start: Date,
        days: Int,
        birthday: Date,
        history: [String],
        disliked: Set<String>,
        calendar: Calendar = .current
    ) -> [(day: Date, tip: Tip)] {
        var simulated = history
        var result: [(day: Date, tip: Tip)] = []
        for offset in 0..<days {
            guard let day = calendar.date(byAdding: .day, value: offset, to: calendar.startOfDay(for: start)) else { continue }
            let age = ChildAge.months(birthday: birthday, on: day, calendar: calendar)
            guard let tip = next(ageInMonths: age, history: simulated, disliked: disliked, seed: DayKey.seed(for: day, calendar: calendar)) else { break }
            simulated.append(tip.id)
            result.append((day, tip))
        }
        return result
    }
}

enum ChildAge {
    static func months(birthday: Date, on date: Date, calendar: Calendar = .current) -> Int {
        calendar.dateComponents([.month], from: birthday, to: date).month ?? 0
    }

    /// "2岁11个月零7天"
    static func description(birthday: Date, on date: Date, calendar: Calendar = .current) -> String {
        let parts = calendar.dateComponents([.year, .month, .day], from: birthday, to: date)
        let years = parts.year ?? 0, months = parts.month ?? 0, days = parts.day ?? 0
        var text = years > 0 ? "\(years)岁" : ""
        if months > 0 { text += "\(months)个月" }
        if days > 0 { text += (text.isEmpty ? "" : "零") + "\(days)天" }
        return text.isEmpty ? "今天出生" : text
    }
}

enum DayKey {
    /// "2026-10-18" in the user's calendar; used as the persistence key for a day.
    static func string(for date: Date, calendar: Calendar = .current) -> String {
        let c = calendar.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", c.year ?? 0, c.month ?? 0, c.day ?? 0)
    }

    static func seed(for date: Date, calendar: Calendar = .current) -> UInt64 {
        let c = calendar.dateComponents([.year, .month, .day], from: date)
        return UInt64((c.year ?? 0) * 10_000 + (c.month ?? 0) * 100 + (c.day ?? 0))
    }
}

/// Small seedable RNG (SplitMix64) so selection is reproducible across launches.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
