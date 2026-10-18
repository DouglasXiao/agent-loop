import XCTest
@testable import OneTip

final class TipPickerTests: XCTestCase {
    private func tip(_ id: String, _ category: TipCategory, _ ages: ClosedRange<Int> = 30...48) -> Tip {
        Tip(id: id, category: category, title: id, body: "", ageMinMonths: ages.lowerBound, ageMaxMonths: ages.upperBound, tags: [], sourceIds: [], basis: "")
    }

    private lazy var tips = [
        tip("emotion-001", .emotion), tip("emotion-002", .emotion),
        tip("play-001", .play), tip("play-002", .play),
        tip("sleep-001", .sleep),
        tip("social-001", .social, 40...54),
    ]

    func testSameSeedSameTip() {
        let picker = TipPicker(tips: tips)
        let a = picker.next(ageInMonths: 35, history: [], disliked: [], seed: 20261018)
        let b = picker.next(ageInMonths: 35, history: [], disliked: [], seed: 20261018)
        XCTAssertEqual(a, b)
    }

    func testSkipsDislikedAndOutOfAge() {
        let picker = TipPicker(tips: tips)
        for seed in 0..<50 as Range<UInt64> {
            let pick = picker.next(ageInMonths: 35, history: [], disliked: ["sleep-001"], seed: seed)
            XCTAssertNotEqual(pick?.id, "sleep-001")
            XCTAssertNotEqual(pick?.id, "social-001") // starts at 40 months
        }
    }

    func testRotatesToStalestCategory() {
        let picker = TipPicker(tips: tips)
        let pick = picker.next(ageInMonths: 35, history: ["emotion-001", "play-001"], disliked: [], seed: 7)
        XCTAssertEqual(pick?.category, .sleep) // the only age-fitting category not shown yet
    }

    func testPlanHasNoRepeatsUntilExhausted() {
        let picker = TipPicker(tips: tips)
        let birthday = Calendar.current.date(byAdding: .month, value: -35, to: .now)!
        let plan = picker.plan(from: .now, days: 5, birthday: birthday, history: [], disliked: [])
        XCTAssertEqual(plan.count, 5)
        XCTAssertEqual(Set(plan.map(\.tip.id)).count, 5)
    }

    func testAgeDescription() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "Asia/Shanghai")!
        let birthday = calendar.date(from: DateComponents(year: 2023, month: 11, day: 11))!
        let today = calendar.date(from: DateComponents(year: 2026, month: 10, day: 18))!
        XCTAssertEqual(ChildAge.description(birthday: birthday, on: today, calendar: calendar), "2岁11个月零7天")
        XCTAssertEqual(ChildAge.months(birthday: birthday, on: today, calendar: calendar), 35)
    }
}
