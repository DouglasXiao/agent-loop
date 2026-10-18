import SwiftUI
import UserNotifications

@main
struct OneTipApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var store = TipStore()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            Group {
                if store.isOnboarded {
                    HomeView()
                } else {
                    OnboardingView()
                }
            }
            .environment(store)
            .onAppear { appDelegate.store = store }
            .onChange(of: scenePhase) { _, phase in
                if phase == .active { Task { await store.rescheduleNotifications() } }
            }
        }
    }
}

/// Routes a tapped notification to the tip it announced, or the playdate
/// reminder to the invite card. A cold launch from a notification can arrive
/// before the store is attached, so hold it until then.
final class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {
    weak var store: TipStore? { didSet { deliverPending() } }
    private var pendingTipId: String?
    private var pendingPlaydate = false

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        UNUserNotificationCenter.current().delegate = self
        return true
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse) async {
        let info = response.notification.request.content.userInfo
        let tipId = info["tipId"] as? String
        let isPlaydate = info["kind"] as? String == "playdate"
        await MainActor.run {
            pendingTipId = tipId
            pendingPlaydate = isPlaydate
            deliverPending()
        }
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }

    private func deliverPending() {
        guard let store else { return }
        if let id = pendingTipId {
            store.pendingTipId = id
            pendingTipId = nil
        }
        if pendingPlaydate {
            store.pendingPlaydateInvite = true
            pendingPlaydate = false
        }
    }
}
