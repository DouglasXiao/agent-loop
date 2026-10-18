import SwiftUI

struct OnboardingView: View {
    @Environment(TipStore.self) private var store
    @State private var birthday = Calendar.current.date(from: DateComponents(year: 2023, month: 11, day: 1)) ?? .now
    @State private var notifyTime = Calendar.current.date(bySettingHour: 8, minute: 0, second: 0, of: .now) ?? .now

    var body: some View {
        VStack(alignment: .leading, spacing: 28) {
            Spacer()
            Text("每日一招")
                .font(.largeTitle.weight(.bold))
            Text("每天一条和她相处的小方法，100 多字，来自儿科、心理和学前教育的可靠来源。")
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                Text("她的生日").font(.headline)
                DatePicker("生日", selection: $birthday, in: ...Date.now, displayedComponents: .date)
                    .labelsHidden()
                Text("用来挑选适合她当前月龄的内容").font(.footnote).foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("每天几点提醒你").font(.headline)
                DatePicker("提醒时间", selection: $notifyTime, displayedComponents: .hourAndMinute)
                    .labelsHidden()
            }

            Spacer()

            BigButton(title: "开始", subtitle: "允许通知，每天准时送到") {
                let time = Calendar.current.dateComponents([.hour, .minute], from: notifyTime)
                store.completeOnboarding(birthday: birthday, hour: time.hour ?? 8, minute: time.minute ?? 0)
                Task {
                    _ = await NotificationScheduler.requestAuthorization()
                    await store.rescheduleNotifications()
                }
            }
        }
        .padding(24)
    }
}
