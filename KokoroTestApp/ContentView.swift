import SwiftUI

/// Chat-style interface for the voice conversation demo: message bubbles,
/// a composer bar with mic / send buttons, and a live listening indicator.
struct ContentView: View {
  /// The view model that manages the TTS engine, speech recognition, and chat state
  @ObservedObject var viewModel: TestAppModel

  /// The text typed into the composer
  @State private var inputText: String = ""

  @FocusState private var inputFocused: Bool

  var body: some View {
    NavigationStack {
      Group {
        if viewModel.isReady {
          VStack(spacing: 0) {
            messages
            composer
          }
        } else {
          loadingView
        }
      }
      .navigationTitle("Nabra")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        if viewModel.isReady {
          ToolbarItem(placement: .topBarTrailing) {
            Menu {
              Picker("Voice", selection: $viewModel.selectedVoice) {
                ForEach(viewModel.voiceNames, id: \.self) { voice in
                  Text(voice).tag(voice)
                }
              }
            } label: {
              Label(viewModel.selectedVoice, systemImage: "person.wave.2")
            }
          }
        }
      }
    }
  }

  // MARK: - Loading

  private var loadingView: some View {
    VStack(spacing: 18) {
      Spacer()
      Image(systemName: "waveform.circle.fill")
        .font(.system(size: 64))
        .foregroundStyle(Color.accentColor)
      Text("Nabra")
        .font(.title2.weight(.bold))
      ProgressView(value: viewModel.loadingProgress)
        .progressViewStyle(.linear)
        .frame(maxWidth: 260)
        .animation(.easeInOut(duration: 0.3), value: viewModel.loadingProgress)
      Text(viewModel.loadingStage)
        .font(.footnote)
        .foregroundStyle(.secondary)
      Spacer()
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(Color(.systemGroupedBackground))
  }

  // MARK: - Messages

  private var messages: some View {
    ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(spacing: 10) {
          if viewModel.conversation.isEmpty && !viewModel.speechRecognizer.isListening {
            emptyState
          }

          ForEach(viewModel.conversation) { message in
            MessageBubble(message: message)
              .id(message.id)
          }

          if viewModel.speechRecognizer.isListening {
            listeningBubble
              .id("listening")
          }

          if let error = viewModel.speechRecognizer.errorMessage {
            Text(error)
              .font(.footnote)
              .foregroundStyle(.red)
              .frame(maxWidth: .infinity, alignment: .center)
              .padding(.top, 4)
          }
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
      }
      .background(Color(.systemGroupedBackground))
      .onChange(of: viewModel.conversation.count) { _, _ in
        scrollToBottom(proxy)
      }
      .onChange(of: viewModel.speechRecognizer.transcript) { _, _ in
        scrollToBottom(proxy)
      }
      .onChange(of: viewModel.speechRecognizer.isListening) { _, _ in
        scrollToBottom(proxy)
      }
    }
  }

  private func scrollToBottom(_ proxy: ScrollViewProxy) {
    withAnimation(.easeOut(duration: 0.2)) {
      if viewModel.speechRecognizer.isListening {
        proxy.scrollTo("listening", anchor: .bottom)
      } else if let last = viewModel.conversation.last {
        proxy.scrollTo(last.id, anchor: .bottom)
      }
    }
  }

  private var emptyState: some View {
    VStack(spacing: 10) {
      Image(systemName: "waveform.circle.fill")
        .font(.system(size: 52))
        .foregroundStyle(Color.accentColor)
      Text("Talk to Nabra")
        .font(.title3.weight(.semibold))
      Text("Tap the mic and speak, or type a message.\nThe app replies with the \(viewModel.selectedVoice) voice.")
        .font(.subheadline)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
    }
    .frame(maxWidth: .infinity)
    .padding(.top, 80)
  }

  /// Bubble on the user's side showing the live transcript while recording
  private var listeningBubble: some View {
    HStack {
      Spacer(minLength: 48)
      HStack(spacing: 8) {
        Image(systemName: "waveform")
          .symbolEffect(.variableColor.iterative)
          .foregroundStyle(Color.accentColor)
        Text(viewModel.speechRecognizer.transcript.isEmpty
             ? "Listening…"
             : viewModel.speechRecognizer.transcript)
          .foregroundStyle(.primary)
      }
      .padding(.horizontal, 14)
      .padding(.vertical, 10)
      .background(
        RoundedRectangle(cornerRadius: 18, style: .continuous)
          .fill(Color.accentColor.opacity(0.15))
      )
    }
  }

  // MARK: - Composer

  private var composer: some View {
    HStack(spacing: 10) {
      TextField("Type a message…", text: $inputText, axis: .vertical)
        .lineLimit(1...4)
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(
          RoundedRectangle(cornerRadius: 20, style: .continuous)
            .fill(Color(.systemGray6))
        )
        .focused($inputFocused)
        .onSubmit(send)

      if inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        // Mic button (red stop while recording)
        Button {
          inputFocused = false
          viewModel.toggleListening()
        } label: {
          Image(systemName: viewModel.speechRecognizer.isListening ? "stop.fill" : "mic.fill")
            .font(.system(size: 17, weight: .semibold))
            .foregroundStyle(.white)
            .frame(width: 38, height: 38)
            .background(
              Circle().fill(viewModel.speechRecognizer.isListening ? Color.red : Color.accentColor)
            )
        }
      } else {
        // Send button
        Button(action: send) {
          Image(systemName: "arrow.up")
            .font(.system(size: 17, weight: .semibold))
            .foregroundStyle(.white)
            .frame(width: 38, height: 38)
            .background(Circle().fill(Color.accentColor))
        }
      }
    }
    .padding(.horizontal)
    .padding(.vertical, 8)
    .background(.bar)
  }

  private func send() {
    viewModel.sendText(inputText)
    inputText = ""
  }
}

/// A single chat bubble: user turns on the right, app turns on the left.
struct MessageBubble: View {
  let message: ChatMessage

  var body: some View {
    HStack {
      if message.role == .user {
        Spacer(minLength: 48)
      }

      Text(message.text)
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .foregroundStyle(message.role == .user ? .white : .primary)
        .background(
          RoundedRectangle(cornerRadius: 18, style: .continuous)
            .fill(message.role == .user ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(Color(.systemGray5)))
        )

      if message.role == .app {
        Spacer(minLength: 48)
      }
    }
  }
}

#Preview {
  ContentView(viewModel: TestAppModel())
}
