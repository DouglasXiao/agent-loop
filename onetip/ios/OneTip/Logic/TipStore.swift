import Foundation
import Observation

/// Everything the app remembers between launches. Small enough for UserDefaults.
struct PersistedState: Codable {
    var birthday: Date?
    var notifyHour = 8
    var notifyMinute = 0
    var notificationsOn = true
    /// Tip ids in the order the parent read them (most recent last).
    var history: [String] = []
    /// DayKey -> tip id. Today and past days are fixed; future days are a plan
    /// that gets recomputed on every launch and mirrored into notifications.
    var dailyTips: [String: String] = [:]
    var favorites: [String] = []
    var liked: Set<String> = []
    var disliked: Set<String> = []
    /// Prefills the next invite's 地点.
    var lastPlaydatePlace = ""
    var weeklyPlaydate: WeeklyPlaydate?
}

extension PersistedState {
    /// Field by field, so state saved by an older build (missing newer keys)
    /// keeps its history and favorites instead of failing to decode wholesale.
    /// Add every new field here with `decodeIfPresent`.
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        birthday = try c.decodeIfPresent(Date.self, forKey: .birthday)
        notifyHour = try c.decodeIfPresent(Int.self, forKey: .notifyHour) ?? notifyHour
        notifyMinute = try c.decodeIfPresent(Int.self, forKey: .notifyMinute) ?? notifyMinute
        notificationsOn = try c.decodeIfPresent(Bool.self, forKey: .notificationsOn) ?? notificationsOn
        history = try c.decodeIfPresent([String].self, forKey: .history) ?? history
        dailyTips = try c.decodeIfPresent([String: String].self, forKey: .dailyTips) ?? dailyTips
        favorites = try c.decodeIfPresent([String].self, forKey: .favorites) ?? favorites
        liked = try c.decodeIfPresent(Set<String>.self, forKey: .liked) ?? liked
        disliked = try c.decodeIfPresent(Set<String>.self, forKey: .disliked) ?? disliked
        lastPlaydatePlace = try c.decodeIfPresent(String.self, forKey: .lastPlaydatePlace) ?? lastPlaydatePlace
        weeklyPlaydate = try c.decodeIfPresent(WeeklyPlaydate.self, forKey: .weeklyPlaydate)
    }
}

@MainActor
@Observable
final class TipStore {
    static let planDays = 60 // leaves room under NotificationScheduler.maxPending for the playdate reminder

    private(set) var tips: [Tip] = []
    private(set) var sources: [String: TipSource] = [:]
    private(set) var state: PersistedState
    /// Set when the user taps a notification; HomeView reveals this tip.
    var pendingTipId: String?
    /// Set when the user taps the playdate reminder; HomeView opens the invite.
    var pendingPlaydateInvite = false

    private let defaults: UserDefaults
    private let stateKey = "onetip.state.v1"
    private var picker: TipPicker { TipPicker(tips: tips) }

    init(defaults: UserDefaults = .standard, bundle: Bundle = .main) {
        self.defaults = defaults
        if let data = defaults.data(forKey: stateKey),
           let saved = try? JSONDecoder().decode(PersistedState.self, from: data) {
            state = saved
        } else {
            state = PersistedState()
        }
        tips = Self.decode([Tip].self, resource: "tips.bundle", bundle: bundle) ?? []
        let list = Self.decode([TipSource].self, resource: "sources.bundle", bundle: bundle) ?? []
        sources = Dictionary(list.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
    }

    var isOnboarded: Bool { state.birthday != nil }

    func tip(id: String) -> Tip? { tips.first { $0.id == id } }

    func ageText(on date: Date = .now) -> String {
        guard let birthday = state.birthday else { return "" }
        return ChildAge.description(birthday: birthday, on: date)
    }

    // MARK: Reading

    /// Today's tip: the one already planned/notified for today, else a fresh pick.
    func todayTip(now: Date = .now) -> Tip? {
        let key = DayKey.string(for: now)
        if let id = state.dailyTips[key], let tip = self.tip(id: id), !state.disliked.contains(id) {
            return tip
        }
        guard let tip = pick(on: now, seed: DayKey.seed(for: now)) else { return nil }
        state.dailyTips[key] = tip.id
        save()
        return tip
    }

    /// "再来一条": another tip right now, never today's and never a repeat while unseen ones remain.
    func extraTip(now: Date = .now) -> Tip? {
        let seed = DayKey.seed(for: now) &* 31 &+ UInt64(state.history.count)
        return pick(on: now, seed: seed)
    }

    func markRead(_ tip: Tip) {
        guard state.history.last != tip.id else { return }
        state.history.append(tip.id)
        save()
    }

    private func pick(on date: Date, seed: UInt64) -> Tip? {
        guard let birthday = state.birthday else { return nil }
        // Treat today's tip and tips already promised to future notifications as
        // seen, so "再来一条" never returns them.
        let todayKey = DayKey.string(for: date)
        let reserved = state.dailyTips.filter { $0.key >= todayKey }.sorted { $0.key < $1.key }.map(\.value)
        let history = state.history + reserved.filter { !state.history.contains($0) }
        return picker.next(
            ageInMonths: ChildAge.months(birthday: birthday, on: date),
            history: history,
            disliked: state.disliked,
            seed: seed
        )
    }

    // MARK: Feedback

    func isFavorite(_ tip: Tip) -> Bool { state.favorites.contains(tip.id) }

    func toggleFavorite(_ tip: Tip) {
        if let index = state.favorites.firstIndex(of: tip.id) {
            state.favorites.remove(at: index)
        } else {
            state.favorites.insert(tip.id, at: 0)
        }
        save()
    }

    func like(_ tip: Tip) {
        state.liked.insert(tip.id)
        state.disliked.remove(tip.id)
        save()
    }

    func dislike(_ tip: Tip) {
        state.disliked.insert(tip.id)
        state.liked.remove(tip.id)
        save()
        Task { await rescheduleNotifications() }
    }

    var favoriteTips: [Tip] { state.favorites.compactMap(tip(id:)) }

    // MARK: Settings

    func completeOnboarding(birthday: Date, hour: Int, minute: Int) {
        state.birthday = birthday
        state.notifyHour = hour
        state.notifyMinute = minute
        save()
    }

    func update(_ change: (inout PersistedState) -> Void) {
        change(&state)
        save()
        Task { await rescheduleNotifications() }
    }

    func resetHistory() {
        state.history = []
        state.dailyTips = state.dailyTips.filter { $0.key <= DayKey.string(for: .now) }
        save()
        Task { await rescheduleNotifications() }
    }

    // MARK: Playdates

    /// What a new invite starts with: the weekly playdate if one is set,
    /// otherwise next Saturday 10:00 at the last place used.
    func inviteDefaults(now: Date = .now) -> (date: Date, place: String) {
        if let weekly = state.weeklyPlaydate {
            return (Playdate.nextOccurrence(of: weekly, after: now), weekly.place.isEmpty ? state.lastPlaydatePlace : weekly.place)
        }
        return (Playdate.defaultDate(after: now), state.lastPlaydatePlace)
    }

    /// "2岁10个月" on the playdate's day.
    func inviteAge(on date: Date) -> String {
        guard let birthday = state.birthday else { return "" }
        return ChildAge.short(birthday: birthday, on: date)
    }

    func rememberPlace(_ place: String) {
        let place = place.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !place.isEmpty, place != state.lastPlaydatePlace else { return }
        state.lastPlaydatePlace = place
        save()
    }

    func setWeeklyPlaydate(_ playdate: WeeklyPlaydate?) {
        guard playdate != state.weeklyPlaydate else { return }
        update { $0.weeklyPlaydate = playdate }
    }

    // MARK: Notifications

    /// Re-plans tomorrow onward and replaces all pending notifications. Called on
    /// launch, after settings change and after a 不适合, so the notification text
    /// always matches the tip the app will show that day.
    func rescheduleNotifications(now: Date = .now) async {
        guard let birthday = state.birthday else { return }
        let calendar = Calendar.current
        let todayKey = DayKey.string(for: now)
        _ = todayTip(now: now) // fix today's tip before planning around it

        state.dailyTips = state.dailyTips.filter { $0.key <= todayKey }
        var history = state.history
        if let todayId = state.dailyTips[todayKey], !history.contains(todayId) { history.append(todayId) }

        guard let tomorrow = calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: now)) else { return }
        let plan = picker.plan(from: tomorrow, days: Self.planDays - 1, birthday: birthday, history: history, disliked: state.disliked)
        for item in plan { state.dailyTips[DayKey.string(for: item.day)] = item.tip.id }
        save()

        var schedule = plan
        if let today = tip(id: state.dailyTips[todayKey] ?? "") {
            schedule.insert((calendar.startOfDay(for: now), today), at: 0) // scheduler skips it if the time has passed
        }
        await NotificationScheduler.replaceAll(
            with: state.notificationsOn ? schedule : [],
            hour: state.notifyHour,
            minute: state.notifyMinute,
            playdate: state.weeklyPlaydate
        )
    }

    // MARK: Persistence

    private func save() {
        if let data = try? JSONEncoder().encode(state) { defaults.set(data, forKey: stateKey) }
    }

    private static func decode<T: Decodable>(_ type: T.Type, resource: String, bundle: Bundle) -> T? {
        guard let url = bundle.url(forResource: resource, withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }
}
