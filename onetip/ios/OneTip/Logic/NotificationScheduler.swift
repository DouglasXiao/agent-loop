import Foundation
import UserNotifications

/// Local notifications only — no server, no account. iOS cannot compute the
/// body at delivery time, so we pre-schedule one notification per planned day
/// (up to 60; the system limit is 64 pending) carrying that day's real title.
enum NotificationScheduler {
    static let idPrefix = "onetip.daily."

    static func requestAuthorization() async -> Bool {
        (try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])) ?? false
    }

    static func replaceAll(with plan: [(day: Date, tip: Tip)], hour: Int, minute: Int, calendar: Calendar = .current) async {
        let center = UNUserNotificationCenter.current()
        let pending = await center.pendingNotificationRequests().map(\.identifier).filter { $0.hasPrefix(idPrefix) }
        center.removePendingNotificationRequests(withIdentifiers: pending)

        let now = Date()
        for item in plan {
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
            let request = UNNotificationRequest(identifier: idPrefix + DayKey.string(for: item.day), content: content, trigger: trigger)
            try? await center.add(request)
        }
    }
}
