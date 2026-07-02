import AVFoundation
import Combine
import Speech

/// Captures microphone audio and transcribes it with SFSpeechRecognizer.
/// Defaults to Arabic (ar-SA); recognition may run on Apple's servers for
/// locales without on-device support, so a network connection is expected.
final class SpeechRecognizer: ObservableObject {
  /// Live transcript, updated with partial results while listening.
  @Published var transcript: String = ""

  /// True while the microphone tap is active.
  @Published var isListening = false

  /// Human-readable error for the UI, if something went wrong.
  @Published var errorMessage: String?

  /// Called once with the final transcript after listening stops.
  var onFinal: ((String) -> Void)?

  private let recognizer: SFSpeechRecognizer?
  private let audioEngine = AVAudioEngine()
  private var request: SFSpeechAudioBufferRecognitionRequest?
  private var task: SFSpeechRecognitionTask?
  private var finished = false

  init(locale: Locale = Locale(identifier: "ar-SA")) {
    recognizer = SFSpeechRecognizer(locale: locale)
  }

  /// Asks for speech-recognition and microphone permission (first launch).
  func requestAuthorization() {
    SFSpeechRecognizer.requestAuthorization { _ in }
    AVAudioApplication.requestRecordPermission { _ in }
  }

  func startListening() {
    guard !isListening else { return }
    guard let recognizer, recognizer.isAvailable else {
      errorMessage = "Speech recognition is unavailable for this locale (check network)."
      return
    }

    transcript = ""
    errorMessage = nil
    finished = false

    let request = SFSpeechAudioBufferRecognitionRequest()
    request.shouldReportPartialResults = true
    // Arabic has no on-device model on most devices; allow server recognition.
    request.requiresOnDeviceRecognition = false
    self.request = request

    let inputNode = audioEngine.inputNode
    let format = inputNode.outputFormat(forBus: 0)
    inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
      request.append(buffer)
    }

    do {
      audioEngine.prepare()
      try audioEngine.start()
    } catch {
      errorMessage = "Microphone unavailable: \(error.localizedDescription)"
      inputNode.removeTap(onBus: 0)
      return
    }
    isListening = true

    task = recognizer.recognitionTask(with: request) { [weak self] result, error in
      DispatchQueue.main.async {
        guard let self else { return }
        if let result {
          self.transcript = result.bestTranscription.formattedString
          if result.isFinal {
            self.finish()
            return
          }
        }
        // The task errors out when audio ends without a final result (e.g.
        // silence) — deliver whatever partial transcript we have.
        if error != nil {
          self.finish()
        }
      }
    }
  }

  /// Stops capturing; `onFinal` fires once the recognizer settles.
  func stopListening() {
    guard isListening else { return }
    request?.endAudio()
    stopAudio()
  }

  private func finish() {
    guard !finished else { return }
    finished = true
    stopAudio()
    isListening = false
    task = nil
    request = nil
    let text = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
    if !text.isEmpty {
      onFinal?(text)
    }
  }

  private func stopAudio() {
    if audioEngine.isRunning {
      audioEngine.stop()
    }
    audioEngine.inputNode.removeTap(onBus: 0)
  }
}
