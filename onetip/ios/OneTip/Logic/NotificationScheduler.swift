import Foundation
import UserNotifications

/// Local notifications only — no server, no account. iOS cannot compute the
/// body at delivery time, so we pre-schedule one notification per planned day
/// carrying that day's real title, plus one repeating weekly playdate reminder.
enum NotificationScheduler {
    static let idPrefix = "onetip."
    static let playdateId = "onetip.playdate.weekly"
    /// iOS keeps at most 64 pending local notifications per app and silently
    /// drops the rest, so we cap ourselves and drop the furthest tip days.
    static let maxPending = 64

    static func requestAuthorization() async -> Bool {
        (try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])) ?? false
    }

    static func replaceAll(with plan: [(day: Date, tip: Tip)], hour: Int, minute: Int, playdate: WeeklyPlaydate?, calendar: Calendar = .current) async {
        let center = UNUserNotificationCenter.current()
        let pending = await center.pendingNotificationRequests().map(\.identifier).filter { $0.hasPrefix(idPrefix) }
        center.removePendingNotificationRequests(withIdentifiers: pending)

        for request in requests(plan: plan, hour: hour, minute: minute, playdate: playdate, now: Date(), calendar: calendar) {
            try? await center.add(request)
        }
    }

    /// Every request to schedule, at most `maxPending`: the playdate reminder
    /// first (one repeating request), then one per planned day in order.
    static func requests(plan: [(day: Date, tip: Tip)], hour: Int, minute: Int, playdate: WeeklyPlaydate?, now: Date, calendar: Calendar = .current) -> [UNNotificationRequest] {
        var result: [UNNotificationRequest] = []
        if let playdate { result.append(playdateRequest(playdate)) }

        for item in plan where result.count < maxPending {
            var parts = calendar.dateComponents([.year, .month, .day], from: item.day)
            parts.hour = hour
            parts.minute = minute
            guard let fireDate = calendar.date(from: parts), fireDate > now else { continue }

            let content = UNMutableNotificationContent()
            content.title = "今日一招 · \(item.tip.category.displayName)"
            content.body = "\(item.tip.title)——点开看看怎么做"
            content.sound = .default
            content.userInfo = ["tipId": item.tip.id]

            let trigger = UNCalendarNotificationTrigger(dateMatching: parts, repeats: false)
            result.append(UNNotificationRequest(identifier: idPrefix + "daily." + DayKey.string(for: item.day, calendar: calendar), content: content, trigger: trigger))
        }
        return result
    }

    private static func playdateRequest(_ playdate: WeeklyPlaydate) -> UNNotificationRequest {
        let content = UNMutableNotificationContent()
        content.title = "明天约玩 · \(Playdate.weekdayName(playdate.weekday)) \(Playdate.timeText(hour: playdate.hour, minute: playdate.minute))"
        let place = playdate.place.trimmingCharacters(in: .whitespacesAndNewlines)
        content.body = (place.isEmpty ? "" : "\(place) · ") + "记得在群里招呼一声，点开就能发邀请卡"
        content.sound = .default
        content.userInfo = ["kind": "playdate"]

        let when = DateComponents(hour: playdate.hour, minute: playdate.minute, weekday: Playdate.reminderWeekday(for: playdate))
        return UNNotificationRequest(identifier: playdateId, content: content, trigger: UNCalendarNotificationTrigger(dateMatching: when, repeats: true))
    }
}
