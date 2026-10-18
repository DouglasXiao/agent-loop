import Foundation

/// A playdate that repeats every week. The parent gets a reminder the day
/// before, at the same time, to post the invite in their group chat.
struct WeeklyPlaydate: Codable, Equatable {
    /// Calendar weekday: 1 = Sunday … 7 = Saturday.
    var weekday = 7
    var hour = 10
    var minute = 0
    var place = ""
}

/// Pure helpers for 约玩: default times, reminder timing and the invite text.
/// Nothing here touches the network; the parent shares the card themselves.
enum Playdate {
    static func weekdayName(_ weekday: Int) -> String {
        ["周日", "周一", "周二", "周三", "周四", "周五", "周六"][(weekday + 6) % 7]
    }

    /// The next Saturday 10:00, a typical weekend-morning meetup.
    static func defaultDate(after now: Date, calendar: Calendar = .current) -> Date {
        nextOccurrence(of: WeeklyPlaydate(), after: now, calendar: calendar)
    }

    static func nextOccurrence(of playdate: WeeklyPlaydate, after now: Date, calendar: Calendar = .current) -> Date {
        let match = DateComponents(hour: playdate.hour, minute: playdate.minute, weekday: playdate.weekday)
        return calendar.nextDate(after: now, matching: match, matchingPolicy: .nextTime) ?? now
    }

    /// The reminder fires one day earlier, at the same time of day.
    static func reminderWeekday(for playdate: WeeklyPlaydate) -> Int {
        playdate.weekday == 1 ? 7 : playdate.weekday - 1
    }

    /// "每周六 10:00"
    static func summary(_ playdate: WeeklyPlaydate) -> String {
        "每\(weekdayName(playdate.weekday)) \(timeText(hour: playdate.hour, minute: playdate.minute))"
    }

    /// "10月24日 周六"
    static func dayText(_ date: Date, calendar: Calendar = .current) -> String {
        let c = calendar.dateComponents([.month, .day, .weekday], from: date)
        return "\(c.month ?? 0)月\(c.day ?? 0)日 \(weekdayName(c.weekday ?? 1))"
    }

    /// "10:00"
    static func timeText(_ date: Date, calendar: Calendar = .current) -> String {
        let c = calendar.dateComponents([.hour, .minute], from: date)
        return timeText(hour: c.hour ?? 0, minute: c.minute ?? 0)
    }

    static func timeText(hour: Int, minute: Int) -> String {
        String(format: "%d:%02d", hour, minute)
    }

    /// The text shared alongside the card, for chats that drop the image caption.
    static func inviteText(date: Date, place: String, note: String, age: String, calendar: Calendar = .current) -> String {
        let place = place.trimmingCharacters(in: .whitespacesAndNewlines)
        let note = note.trimmingCharacters(in: .whitespacesAndNewlines)
        var lines = ["【约玩】\(dayText(date, calendar: calendar)) \(timeText(date, calendar: calendar))"]
        if !place.isEmpty { lines.append("📍 \(place)") }
        lines.append("👧 \(age)女孩，想找同龄小伙伴一起玩")
        if !note.isEmpty { lines.append(note) }
        lines.append("有兴趣的家长回我一声～")
        return lines.joined(separator: "\n")
    }
}

extension ChildAge {
    /// "2岁10个月": whole months only, for invites (no name, no birthday).
    static func short(birthday: Date, on date: Date, calendar: Calendar = .current) -> String {
        let parts = calendar.dateComponents([.year, .month], from: birthday, to: date)
        let years = parts.year ?? 0, months = parts.month ?? 0
        if years == 0 { return "\(months)个月" }
        return months == 0 ? "\(years)岁" : "\(years)岁\(months)个月"
    }
}
