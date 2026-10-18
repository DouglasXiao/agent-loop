import SwiftUI
import UIKit

/// 约个玩伴: pick when and where, get a card image plus text to post in a group
/// chat. Nothing leaves the phone except what the parent chooses to share.
struct PlaydateInviteView: View {
    @Environment(TipStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    @State private var date: Date
    @State private var place: String
    @State private var note = ""
    @State private var image: Image?
    @State private var copied = false

    init(date: Date, place: String) {
        _date = State(initialValue: date)
        _place = State(initialValue: place)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("这次约玩") {
                    DatePicker("时间", selection: $date, in: Date.now..., displayedComponents: [.date, .hourAndMinute])
                    TextField("地点，比如小区中心花园", text: $place)
                    TextField("备注（可选），比如带了泡泡机", text: $note, axis: .vertical)
                        .lineLimit(1...3)
                }

                if let image {
                    Section {
                        image
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                            .frame(maxWidth: 280)
                            .frame(maxWidth: .infinity)
                            .accessibilityLabel("邀请卡预览")
                    }
                    .listRowBackground(Color.clear)

                    Section {
                        ShareLink(item: image, subject: Text("约玩"), message: Text(inviteText), preview: SharePreview("约玩邀请卡", image: image)) {
                            Label("分享邀请卡", systemImage: "square.and.arrow.up")
                        }
                        Button(action: copyText) {
                            Label(copied ? "已复制" : "复制文字", systemImage: copied ? "checkmark" : "doc.on.doc")
                        }
                    } footer: {
                        Text("发微信群时，先发图片，再粘贴文字。卡片上只写月龄，不写名字。")
                    }
                }

                Section {
                    NavigationLink { WeeklyPlaydateView(suggestion: weeklySuggestion) } label: {
                        LabeledContent("每周固定约玩", value: store.state.weeklyPlaydate.map(Playdate.summary) ?? "未设置")
                    }
                } footer: {
                    Text("孩子交朋友靠反复见面。每周同一时间、同一地点，比每次临时约更容易成。")
                }
            }
            .navigationTitle("约个玩伴")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("完成") { dismiss() } }
            }
            .onChange(of: card, initial: true) {
                render()
                copied = false
            }
            .onDisappear { store.rememberPlace(place) }
        }
    }

    private var card: InviteCardView {
        InviteCardView(date: date, place: place, note: note, age: store.inviteAge(on: date))
    }

    /// "Make this one weekly": the invite's day, time and place.
    private var weeklySuggestion: WeeklyPlaydate {
        let c = Calendar.current.dateComponents([.weekday, .hour, .minute], from: date)
        return WeeklyPlaydate(weekday: c.weekday ?? 7, hour: c.hour ?? 10, minute: c.minute ?? 0, place: place)
    }

    private var inviteText: String {
        Playdate.inviteText(date: date, place: place, note: note, age: store.inviteAge(on: date))
    }

    private func render() {
        // Always light, so the image looks the same in every chat.
        let renderer = ImageRenderer(content: card.environment(\.colorScheme, .light))
        renderer.scale = 3
        image = renderer.uiImage.map(Image.init(uiImage:))
    }

    private func copyText() {
        UIPasteboard.general.string = inviteText
        store.rememberPlace(place)
        copied = true
    }
}

/// The shareable card. Fixed size and fixed font sizes so the rendered image
/// is the same 1080×1350 picture whatever the phone's text size.
struct InviteCardView: View, Equatable {
    let date: Date
    let place: String
    let note: String
    let age: String

    private static let ink = Color(red: 0.16, green: 0.14, blue: 0.13)
    private static let accent = Color(red: 0.93, green: 0.36, blue: 0.42)
    private static let paper = Color(red: 1.0, green: 0.97, blue: 0.95)

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("🤝 找同龄小伙伴")
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 28)
                .padding(.vertical, 22)
                .background(.onetipWarm)

            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(Playdate.dayText(date))
                        .font(.system(size: 34, weight: .bold, design: .rounded))
                    Text(Playdate.timeText(date))
                        .font(.system(size: 28, weight: .semibold, design: .rounded))
                        .foregroundStyle(Self.accent)
                }
                line("📍", trimmed(place).isEmpty ? "地点待定" : trimmed(place))
                line("👧", "\(age)女孩")
                if !trimmed(note).isEmpty {
                    Text(trimmed(note))
                        .font(.system(size: 17))
                        .foregroundStyle(Self.ink.opacity(0.7))
                        .lineLimit(3)
                }
                Spacer(minLength: 0)
                Text("想约同龄的小朋友一起玩\n有兴趣的家长，回我一声就好～")
                    .font(.system(size: 16))
                    .lineSpacing(4)
                    .foregroundStyle(Self.ink.opacity(0.6))
            }
            .padding(28)
        }
        .foregroundStyle(Self.ink)
        .frame(width: 360, height: 450)
        .background(Self.paper)
    }

    private func line(_ icon: String, _ text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(icon)
            Text(text).lineLimit(2).minimumScaleFactor(0.8)
        }
        .font(.system(size: 20, weight: .semibold))
    }

    private func trimmed(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// The one recurring playdate. Day and time save as they change; the place
/// saves on return or when leaving the screen, not on every keystroke (each
/// save reschedules all notifications).
struct WeeklyPlaydateView: View {
    @Environment(TipStore.self) private var store
    /// Prefill when no weekly playdate is set yet (e.g. from the invite being edited).
    var suggestion: WeeklyPlaydate?
    @State private var enabled = false
    @State private var playdate = WeeklyPlaydate()

    var body: some View {
        Form {
            Section {
                Toggle("每周固定约玩", isOn: $enabled)
                if enabled {
                    Picker("星期", selection: $playdate.weekday) {
                        ForEach([2, 3, 4, 5, 6, 7, 1], id: \.self) { Text(Playdate.weekdayName($0)).tag($0) }
                    }
                    DatePicker("时间", selection: time, displayedComponents: .hourAndMinute)
                    TextField("地点", text: $playdate.place)
                        .onSubmit(save)
                }
            } footer: {
                if enabled {
                    let remind = Playdate.weekdayName(Playdate.reminderWeekday(for: playdate))
                    Text("每个\(remind)的 \(Playdate.timeText(hour: playdate.hour, minute: playdate.minute)) 提醒你明天约玩，记得在群里招呼一声；点提醒就能直接发邀请卡。")
                } else {
                    Text("定一个每周固定的时间和地点，前一天提醒你在群里招呼一声。")
                }
            }
        }
        .navigationTitle("每周约玩")
        .onAppear(perform: load)
        .onChange(of: enabled) { _, on in
            save()
            if on { Task { _ = await NotificationScheduler.requestAuthorization() } }
        }
        .onChange(of: playdate.weekday) { save() }
        .onDisappear(perform: save)
    }

    private var time: Binding<Date> {
        Binding(
            get: { Calendar.current.date(bySettingHour: playdate.hour, minute: playdate.minute, second: 0, of: .now) ?? .now },
            set: { value in
                let parts = Calendar.current.dateComponents([.hour, .minute], from: value)
                playdate.hour = parts.hour ?? 10
                playdate.minute = parts.minute ?? 0
                save()
            }
        )
    }

    private func load() {
        if let weekly = store.state.weeklyPlaydate {
            playdate = weekly
            enabled = true
        } else if let suggestion {
            playdate = suggestion
        } else {
            playdate.place = store.state.lastPlaydatePlace
        }
    }

    private func save() {
        var value = playdate
        value.place = value.place.trimmingCharacters(in: .whitespacesAndNewlines)
        store.setWeeklyPlaydate(enabled ? value : nil)
    }
}
