import SwiftUI

struct SettingsView: View {
    @Environment(TipStore.self) private var store
    @State private var confirmReset = false

    var body: some View {
        Form {
            Section("孩子") {
                DatePicker("生日", selection: birthday, in: ...Date.now, displayedComponents: .date)
                LabeledContent("现在", value: store.ageText())
            }
            Section("每日提醒") {
                Toggle("推送今日一招", isOn: binding(\.notificationsOn))
                DatePicker("时间", selection: notifyTime, displayedComponents: .hourAndMinute)
                    .disabled(!store.state.notificationsOn)
            }
            Section("约玩") {
                NavigationLink { WeeklyPlaydateView() } label: {
                    LabeledContent("每周固定约玩", value: store.state.weeklyPlaydate.map(Playdate.summary) ?? "未设置")
                }
            }
            Section("内容") {
                LabeledContent("已读", value: "\(Set(store.state.history).count) / \(store.tips.count)")
                LabeledContent("标记不适合", value: "\(store.state.disliked.count)")
                Button("重新开始（清空已读）", role: .destructive) { confirmReset = true }
            }
            Section("关于") {
                NavigationLink("内容来源") { SourcesView() }
                Text("内容是一般性的养育建议，不能代替医生或专业人士对你家孩子的具体判断。")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("设置")
        .confirmationDialog("清空已读记录？收藏不受影响。", isPresented: $confirmReset, titleVisibility: .visible) {
            Button("清空", role: .destructive) { store.resetHistory() }
        }
    }

    private var birthday: Binding<Date> {
        Binding(get: { store.state.birthday ?? .now }, set: { value in store.update { $0.birthday = value } })
    }

    private var notifyTime: Binding<Date> {
        Binding(
            get: { Calendar.current.date(bySettingHour: store.state.notifyHour, minute: store.state.notifyMinute, second: 0, of: .now) ?? .now },
            set: { value in
                let parts = Calendar.current.dateComponents([.hour, .minute], from: value)
                store.update { $0.notifyHour = parts.hour ?? 8; $0.notifyMinute = parts.minute ?? 0 }
            }
        )
    }

    private func binding(_ keyPath: WritableKeyPath<PersistedState, Bool>) -> Binding<Bool> {
        Binding(get: { store.state[keyPath: keyPath] }, set: { value in store.update { $0[keyPath: keyPath] = value } })
    }
}

struct SourcesView: View {
    @Environment(TipStore.self) private var store

    var body: some View {
        List(store.sources.values.sorted { $0.name < $1.name }) { source in
            if let url = source.url {
                Link(source.name, destination: url)
            } else {
                Text(source.name)
            }
        }
        .navigationTitle("内容来源")
    }
}

struct FavoritesView: View {
    @Environment(TipStore.self) private var store

    var body: some View {
        Group {
            if store.favoriteTips.isEmpty {
                ContentUnavailableView("还没有收藏", systemImage: "heart", description: Text("读到好用的招，点 ♡ 收起来"))
            } else {
                List(store.favoriteTips) { tip in
                    NavigationLink {
                        TipCardView(tip: tip).padding()
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(tip.title).font(.headline)
                            Text("\(tip.category.emoji) \(tip.category.displayName)").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .navigationTitle("收藏")
    }
}
