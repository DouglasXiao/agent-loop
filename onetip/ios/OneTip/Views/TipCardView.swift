import SwiftUI

struct TipCardView: View {
    @Environment(TipStore.self) private var store
    let tip: Tip
    var onDislike: (() -> Void)?
    @State private var showBasis = false

    init(tip: Tip, onDislike: (() -> Void)? = nil) {
        self.tip = tip
        self.onDislike = onDislike
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("\(tip.category.emoji) \(tip.category.displayName)")
                    .font(.caption.weight(.medium))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(.orange.opacity(0.15)))

                Text(tip.title)
                    .font(.title2.weight(.bold))

                Text(tip.body)
                    .font(.body)
                    .lineSpacing(7)
                    .textSelection(.enabled)

                DisclosureGroup("依据", isExpanded: $showBasis) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(tip.basis).font(.footnote)
                        ForEach(tip.sourceIds, id: \.self) { id in
                            if let source = store.sources[id], let url = source.url {
                                Link(source.name, destination: url).font(.footnote)
                            } else {
                                Text(store.sources[id]?.name ?? id).font(.footnote)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 6)
                }
                .font(.footnote)
                .foregroundStyle(.secondary)

                actions
            }
            .padding(22)
            .background(RoundedRectangle(cornerRadius: 28, style: .continuous).fill(Color(.secondarySystemBackground)))
        }
        .scrollBounceBehavior(.basedOnSize)
    }

    private var actions: some View {
        HStack(spacing: 18) {
            Button { store.toggleFavorite(tip) } label: {
                Label("收藏", systemImage: store.isFavorite(tip) ? "heart.fill" : "heart")
            }
            Button { store.like(tip) } label: {
                Label("有用", systemImage: store.state.liked.contains(tip.id) ? "hand.thumbsup.fill" : "hand.thumbsup")
            }
            Button {
                store.dislike(tip)
                onDislike?()
            } label: {
                Label("不适合", systemImage: "hand.thumbsdown")
            }
            Spacer()
            ShareLink(item: "【\(tip.title)】\(tip.body)") {
                Image(systemName: "square.and.arrow.up")
            }
            .accessibilityLabel("分享")
        }
        .labelStyle(.iconOnly)
        .font(.title3)
        .tint(.pink)
        .padding(.top, 4)
    }
}
