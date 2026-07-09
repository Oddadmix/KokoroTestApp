import SwiftUI

/// Full-screen live camera with a voice loop: ask a question out loud and the
/// on-device VLM answers about what the camera sees, spoken back in Nabra's
/// voice. Shown while `viewModel.isCameraMode` is true.
struct CameraScreen: View {
  @ObservedObject var viewModel: TestAppModel

  var body: some View {
    ZStack {
      Color.black.ignoresSafeArea()

      if viewModel.camera.permissionDenied {
        permissionDenied
      } else {
        CameraPreview(session: viewModel.camera.session)
          .ignoresSafeArea()
      }

      // Dim the top/bottom so the overlay text stays readable over any scene.
      LinearGradient(colors: [.black.opacity(0.55), .clear, .clear, .black.opacity(0.65)],
                     startPoint: .top, endPoint: .bottom)
        .ignoresSafeArea()
        .allowsHitTesting(false)

      VStack(spacing: 0) {
        topBar
        Spacer()
        if let error = viewModel.visionError {
          errorBanner(error)
        }
        if viewModel.visionLoading {
          loadingCard
        } else {
          answerCard
          statusPill
        }
      }
      .padding(.horizontal, 16)
      .padding(.bottom, 24)
    }
    .preferredColorScheme(.dark)
  }

  // MARK: - Top bar

  private var topBar: some View {
    HStack {
      Label("Camera", systemImage: "camera.viewfinder")
        .font(.subheadline.weight(.semibold))
        .foregroundStyle(.white.opacity(0.9))
      Spacer()
      Button {
        viewModel.toggleCamera()
      } label: {
        Image(systemName: "xmark")
          .font(.system(size: 15, weight: .bold))
          .foregroundStyle(.white)
          .frame(width: 34, height: 34)
          .background(.ultraThinMaterial, in: Circle())
      }
    }
    .padding(.top, 8)
  }

  // MARK: - Loading (first open downloads the VLM)

  private var loadingCard: some View {
    VStack(spacing: 12) {
      ProgressView(value: viewModel.visionProgress)
        .progressViewStyle(.linear)
        .tint(.white)
      Text(viewModel.visionProgress > 0.001
           ? "جارٍ تحميل نموذج الرؤية… \(Int(viewModel.visionProgress * 100))٪"
           : "جارٍ تحضير نموذج الرؤية…")
        .font(.subheadline.weight(.medium))
        .foregroundStyle(.white)
    }
    .padding(16)
    .frame(maxWidth: .infinity)
    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
  }

  // MARK: - Latest answer

  @ViewBuilder private var answerCard: some View {
    if let answer = viewModel.conversation.last(where: { $0.role == .app })?.text,
       viewModel.conversationState != .listening {
      Text(answer)
        .font(.body)
        .foregroundStyle(.white)
        .multilineTextAlignment(.trailing)
        .frame(maxWidth: .infinity, alignment: .trailing)
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .padding(.bottom, 10)
        .transition(.opacity)
    }
  }

  // MARK: - Status pill (listening transcript / thinking / speaking)

  private var statusPill: some View {
    HStack(spacing: 8) {
      Image(systemName: statusIcon)
        .symbolEffect(.variableColor.iterative)
        .foregroundStyle(statusTint)
      Text(statusText)
        .font(.callout.weight(.medium))
        .foregroundStyle(.white)
        .lineLimit(2)
        .multilineTextAlignment(.center)
    }
    .padding(.horizontal, 16)
    .padding(.vertical, 12)
    .frame(maxWidth: .infinity)
    .background(.ultraThinMaterial, in: Capsule())
  }

  private var statusIcon: String {
    switch viewModel.conversationState {
    case .listening: return "waveform"
    case .thinking:  return "eye"
    case .speaking:  return "speaker.wave.2.fill"
    case .idle:      return "mic.slash"
    }
  }

  private var statusTint: Color {
    switch viewModel.conversationState {
    case .listening: return viewModel.speechRecognizer.speechDetected ? .green : .white
    case .thinking:  return .orange
    case .speaking:  return .blue
    case .idle:      return .white
    }
  }

  private var statusText: String {
    switch viewModel.conversationState {
    case .listening:
      return viewModel.speechRecognizer.transcript.isEmpty
        ? "اسأل عمّا تراه الكاميرا…" : viewModel.speechRecognizer.transcript
    case .thinking:  return viewModel.toolStatus ?? "يفكّر…"
    case .speaking:
      return viewModel.stringToFollowTheAudio.isEmpty
        ? "يتحدّث…" : viewModel.stringToFollowTheAudio
    case .idle:      return "الكاميرا متوقفة"
    }
  }

  // MARK: - Error banner (raw VLM error, for diagnosis)

  private func errorBanner(_ text: String) -> some View {
    Text(text)
      .font(.caption.monospaced())
      .foregroundStyle(.white)
      .multilineTextAlignment(.leading)
      .frame(maxWidth: .infinity, alignment: .leading)
      .padding(12)
      .background(.red.opacity(0.85), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
      .padding(.bottom, 10)
      .textSelection(.enabled)
  }

  // MARK: - Permission denied

  private var permissionDenied: some View {
    VStack(spacing: 12) {
      Image(systemName: "camera.metering.unknown")
        .font(.system(size: 48))
        .foregroundStyle(.white.opacity(0.8))
      Text("لا يمكن الوصول إلى الكاميرا")
        .font(.headline)
        .foregroundStyle(.white)
      Text("فعّل إذن الكاميرا من الإعدادات لاستخدام هذه الميزة.")
        .font(.subheadline)
        .foregroundStyle(.white.opacity(0.7))
        .multilineTextAlignment(.center)
    }
    .padding(32)
  }
}
