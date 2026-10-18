import XCTest
import UserNotifications
@testable import OneTip

final class PlaydateTests: XCTestCase {
    private let calendar: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "Asia/Shanghai")!
        return calendar
    }()

    private func date(_ year: Int, _ month: Int, _ day: Int, _ hour: Int = 0) -> Date {
        calendar.date(from: DateComponents(year: year, month: month, day: day, hour: hour))!
    }

    private func plan(days: Int, from start: Date) -> [(day: Date, tip: Tip)] {
        (0..<days).map { offset in
            let tip = Tip(id: "t\(offset)", category: .play, title: "t\(offset)", body: "", ageMinMonths: 0, ageMaxMonths: 99, tags: [], sourceIds: [], basis: "")
            return (calendar.date(byAdding: .day, value: offset, to: start)!, tip)
        }
    }

    func testInviteAge() {
        let birthday = date(2023, 11, 29)
        XCTAssertEqual(ChildAge.short(birthday: birthday, on: date(2026, 10, 24), calendar: calendar), "2岁10个月")
        XCTAssertEqual(ChildAge.short(birthday: birthday, on: date(2026, 11, 29), calendar: calendar), "3岁")
        XCTAssertEqual(ChildAge.short(birthday: birthday, on: date(2024, 9, 1), calendar: calendar), "9个月")
    }

    func testDefaultIsNextSaturdayTenAM() {
        // 2026-10-18 is a Sunday.
        XCTAssertEqual(Playdate.defaultDate(after: date(2026, 10, 18, 15), calendar: calendar), date(2026, 10, 24, 10))
        // On a Saturday: still today before 10:00, next week after.
        XCTAssertEqual(Playdate.defaultDate(after: date(2026, 10, 24, 9), calendar: calendar), date(2026, 10, 24, 10))
        XCTAssertEqual(Playdate.defaultDate(after: date(2026, 10, 24, 11), calendar: calendar), date(2026, 10, 31, 10))
    }

    func testInviteText() {
        let text = Playdate.inviteText(date: date(2026, 10, 24, 10), place: " 中心花园 ", note: "", age: "2岁10个月", calendar: calendar)
        XCTAssertEqual(text, "【约玩】10月24日 周六 10:00\n📍 中心花园\n👧 2岁10个月女孩，想找同龄小伙伴一起玩\n有兴趣的家长回我一声～")
    }

    func testReminderFiresTheDayBeforeEveryWeek() {
        XCTAssertEqual(Playdate.reminderWeekday(for: WeeklyPlaydate(weekday: 1)), 7) // Sunday → Saturday

        let playdate = WeeklyPlaydate(weekday: 7, hour: 10, minute: 30, place: "中心花园")
        let requests = NotificationScheduler.requests(plan: [], hour: 8, minute: 0, playdate: playdate, now: date(2026, 10, 18), calendar: calendar)
        let trigger = requests.first?.trigger as? UNCalendarNotificationTrigger
        XCTAssertEqual(requests.count, 1)
        XCTAssertEqual(trigger?.repeats, true)
        XCTAssertEqual(trigger?.dateComponents.weekday, 6)
        XCTAssertEqual(trigger?.dateComponents.hour, 10)
        XCTAssertEqual(trigger?.dateComponents.minute, 30)
    }

    func testNotificationsStayUnderSystemLimit() {
        let now = date(2026, 10, 18, 7)
        let requests = NotificationScheduler.requests(plan: plan(days: 70, from: now), hour: 8, minute: 0, playdate: WeeklyPlaydate(), now: now, calendar: calendar)
        XCTAssertEqual(requests.count, NotificationScheduler.maxPending)
        XCTAssertTrue(requests.contains { $0.identifier == NotificationScheduler.playdateId })
        // The furthest days are dropped; the nearest keep their real titles.
        XCTAssertEqual(requests.last?.content.body, "t62——点开看看怎么做")
    }

    @MainActor
    func testRealPlanLeavesRoomForPlaydate() {
        XCTAssertLessThanOrEqual(TipStore.planDays + 1, NotificationScheduler.maxPending)
    }

    func testOldSavedStateStillDecodes() throws {
        let old = #"{"notifyHour":7,"notifyMinute":30,"notificationsOn":true,"history":["a","b"],"dailyTips":{},"favorites":["a"],"liked":[],"disliked":["c"]}"#
        let state = try JSONDecoder().decode(PersistedState.self, from: Data(old.utf8))
        XCTAssertEqual(state.history, ["a", "b"])
        XCTAssertEqual(state.favorites, ["a"])
        XCTAssertEqual(state.notifyHour, 7)
        XCTAssertEqual(state.disliked, ["c"])
        XCTAssertEqual(state.lastPlaydatePlace, "")
        XCTAssertNil(state.weeklyPlaydate)
    }

    func testPlaydateStateRoundTrips() throws {
        var state = PersistedState()
        state.lastPlaydatePlace = "中心花园"
        state.weeklyPlaydate = WeeklyPlaydate(weekday: 3, hour: 16, minute: 0, place: "图书馆")
        let decoded = try JSONDecoder().decode(PersistedState.self, from: JSONEncoder().encode(state))
        XCTAssertEqual(decoded.lastPlaydatePlace, "中心花园")
        XCTAssertEqual(decoded.weeklyPlaydate, state.weeklyPlaydate)
    }
}
