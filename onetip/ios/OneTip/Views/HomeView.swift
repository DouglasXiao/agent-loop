import SwiftUI

/// The whole app in one screen: a card area on top and one big button at the
/// bottom (thumb reach). First press reveals today's tip; later presses give
/// "再来一条".
struct HomeView: View {
    @Environment(TipStore.self) private var store
    @State private var current: Tip?
    @State private var readToday = false
    @State private var pressCount = 0
    @State private var showInvite = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                header
                ZStack {
                    if let tip = current {
                        TipCardView(tip: tip, onDislike: { showNext() })
                            .id(tip.id)
                            .transition(.asymmetric(insertion: .scale(scale: 0.9).combined(with: .opacity), removal: .opacity))
                    } else {
                        placeholder
                    }
                }
                .frame(maxHeight: .infinity)
                BigButton(title: readToday ? "再来一条" : "今日一招", subtitle: readToday ? "换个角度再学一招" : "按一下，看今天怎么陪她") {
                    showNext()
                }
                .sensoryFeedback(.impact(weight: .medium), trigger: pressCount)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 24)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    NavigationLink { FavoritesView() } label: { Image(systemName: "heart") }
                        .accessibilityLabel("收藏")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink { SettingsView() } label: { Image(systemName: "gearshape") }
                        .accessibilityLabel("设置")
                }
            }
            .onChange(of: store.pendingTipId) { _, id in openFromNotification(id) }
            .onChange(of: store.pendingPlaydateInvite) { openInviteFromNotification() }
            .onAppear {
                openFromNotification(store.pendingTipId)
                openInviteFromNotification()
            }
            .sheet(isPresented: $showInvite) {
                let defaults = store.inviteDefaults()
                PlaydateInviteView(date: defaults.date, place: defaults.place)
                    .environment(store)
            }
        }
    }

    private var header: some View {
        VStack(spacing: 4) {
            Text("她今天")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text(store.ageText())
                .font(.title3.weight(.semibold))
            Button { showInvite = true } label: {
                Label(store.state.weeklyPlaydate.map { "约个玩伴 · \(Playdate.summary($0))" } ?? "约个玩伴", systemImage: "figure.2.and.child.holdinghands")
                    .font(.footnote.weight(.medium))
            }
            .buttonStyle(.bordered)
            .buttonBorderShape(.capsule)
            .controlSize(.small)
            .tint(.pink)
            .padding(.top, 6)
        }
        .padding(.top, 8)
    }

    private var placeholder: some View {
        VStack(spacing: 12) {
            Text("🌱").font(.system(size: 56))
            Text("每天一招，100 多字，一分钟读完")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    private func showNext() {
        pressCount += 1
        let tip = readToday ? store.extraTip() : store.todayTip()
        guard let tip else { return }
        store.markRead(tip)
        withAnimation(.spring(duration: 0.45, bounce: 0.25)) {
            current = tip
            readToday = true
        }
    }

    private func openFromNotification(_ id: String?) {
        guard let id, let tip = store.tip(id: id) else { return }
        store.pendingTipId = nil
        store.markRead(tip)
        withAnimation { current = tip; readToday = true }
    }

    private func openInviteFromNotification() {
        guard store.pendingPlaydateInvite else { return }
        store.pendingPlaydateInvite = false
        showInvite = true
    }
}

extension ShapeStyle where Self == LinearGradient {
    /// The app's coral-to-pink accent (big button, invite card header).
    static var onetipWarm: LinearGradient {
        LinearGradient(colors: [Color(red: 1.0, green: 0.55, blue: 0.45), Color(red: 0.96, green: 0.36, blue: 0.45)], startPoint: .topLeading, endPoint: .bottomTrailing)
    }
}

struct BigButton: View {
    let title: String
    let subtitle: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 6) {
                Text(title).font(.system(size: 30, weight: .bold, design: .rounded))
                Text(subtitle).font(.footnote).opacity(0.85)
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .frame(height: 132)
            .background(
                RoundedRectangle(cornerRadius: 36, style: .continuous)
                    .fill(.onetipWarm)
            )
            .shadow(color: Color(red: 0.96, green: 0.36, blue: 0.45).opacity(0.35), radius: 16, y: 8)
        }
        .buttonStyle(PressableStyle())
    }
}

private struct PressableStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .animation(.spring(duration: 0.2), value: configuration.isPressed)
    }
}
