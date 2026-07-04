import Accelerate
import AVFoundation
import Combine
import Speech

/// Captures microphone audio and transcribes it with SFSpeechRecognizer.
/// Defaults to Arabic (ar-SA); recognition may run on Apple's servers for
/// locales without on-device support, so a network connection is expected.
///
/// Includes an energy-based Voice Activity Detector (VAD): once the user has
/// spoken, a trailing pause auto-finalizes the turn — no stop button needed.
final class SpeechRecognizer: ObservableObject {
  /// Live transcript, updated with partial results while listening.
  @Published var transcript: String = ""

  /// True while the microphone tap is active.
  @Published var isListening = false

  /// True once the VAD has detected the user actually speaking this turn.
  @Published var speechDetected = false

  /// Human-readable error for the UI, if something went wrong.
  @Published var errorMessage: String?

  /// Called once with the final transcript after a turn with speech ends.
  var onFinal: ((String) -> Void)?

  /// Called when a turn ends with no speech (VAD no-speech timeout / silence).
  var onNoSpeech: (() -> Void)?

  // MARK: - VAD tuning

  /// Trailing silence (seconds) that ends a turn once speech has started.
  var endSilenceSeconds: Double = 1.2
  /// Minimum voiced time (seconds) before the auto-stop arms — ignores coughs.
  var minSpeechSeconds: Double = 0.25
  /// If the user never speaks, give up after this many seconds.
  var noSpeechTimeoutSeconds: Double = 12
  /// A frame counts as speech when it's this many dB above the noise floor.
  var speechMarginDb: Float = 12

  private let recognizer: SFSpeechRecognizer?
  private let audioEngine = AVAudioEngine()
  private var request: SFSpeechAudioBufferRecognitionRequest?
  private var task: SFSpeechRecognitionTask?
  private var finished = false

  // VAD state (touched only on the audio thread, reset on start)
  private var noiseFloorDb: Float = -50
  private var speechAccum: Double = 0
  private var silenceAccum: Double = 0
  private var elapsedAccum: Double = 0
  private var armed = false
  private var vadStopRequested = false

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
      onNoSpeech?()
      return
    }

    transcript = ""
    errorMessage = nil
    speechDetected = false
    finished = false

    // Reset VAD
    noiseFloorDb = -50
    speechAccum = 0
    silenceAccum = 0
    elapsedAccum = 0
    armed = false
    vadStopRequested = false

    let request = SFSpeechAudioBufferRecognitionRequest()
    request.shouldReportPartialResults = true
    // Arabic has no on-device model on most devices; allow server recognition.
    request.requiresOnDeviceRecognition = false
    self.request = request

    let inputNode = audioEngine.inputNode
    let format = inputNode.outputFormat(forBus: 0)
    inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
      request.append(buffer)
      self?.processVAD(buffer)
    }

    do {
      audioEngine.prepare()
      try audioEngine.start()
    } catch {
      errorMessage = "Microphone unavailable: \(error.localizedDescription)"
      inputNode.removeTap(onBus: 0)
      onNoSpeech?()
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
        if let error {
          if self.transcript.isEmpty && !self.vadStopRequested {
            // kAFAssistantErrorDomain 203 "Corrupt"/"Retry": Apple's speech
            // servers rejected the session. Common in the iOS Simulator, which
            // can't run server-based recognition — use a real device.
            self.errorMessage = "Recognition failed: \(error.localizedDescription). "
              + "Server-based Arabic recognition usually fails in the Simulator — try a real device."
          }
          self.finish()
        }
      }
    }
  }

  /// Energy-based VAD. Runs on the audio thread; measures each buffer's level,
  /// tracks an adaptive noise floor, and requests a stop after enough trailing
  /// silence (once armed) or after the no-speech timeout.
  private func processVAD(_ buffer: AVAudioPCMBuffer) {
    guard !vadStopRequested, let channel = buffer.floatChannelData else { return }
    let frames = vDSP_Length(buffer.frameLength)
    guard frames > 0 else { return }

    var rms: Float = 0
    vDSP_rmsqv(channel[0], 1, &rms, frames)
    let db = 20 * log10(max(rms, 1e-7))
    let dur = Double(buffer.frameLength) / buffer.format.sampleRate
    elapsedAccum += dur

    let isSpeech = db > noiseFloorDb + speechMarginDb
    if isSpeech {
      speechAccum += dur
      silenceAccum = 0
      if !armed && speechAccum >= minSpeechSeconds {
        armed = true
        DispatchQueue.main.async { [weak self] in self?.speechDetected = true }
      }
    } else {
      silenceAccum += dur
      // Adapt the noise floor from silent frames, kept in a sane range.
      noiseFloorDb += 0.02 * (db - noiseFloorDb)
      noiseFloorDb = min(max(noiseFloorDb, -60), -25)
    }

    let shouldStop = armed ? (silenceAccum >= endSilenceSeconds)
                           : (elapsedAccum >= noSpeechTimeoutSeconds)
    if shouldStop {
      vadStopRequested = true
      DispatchQueue.main.async { [weak self] in self?.stopListening() }
    }
  }

  /// Stops capturing; the appropriate callback fires once the recognizer settles.
  func stopListening() {
    guard isListening else { return }
    request?.endAudio()
    stopAudio()
    // If the user never spoke, don't wait on a server round-trip — end now.
    if !armed {
      finish()
    }
  }

  /// Stops immediately without firing any callbacks (e.g. user typed instead).
  func cancel() {
    guard isListening else { return }
    finished = true
    vadStopRequested = true
    task?.cancel()
    stopAudio()
    isListening = false
    speechDetected = false
    task = nil
    request = nil
  }

  private func finish() {
    guard !finished else { return }
    finished = true
    stopAudio()
    isListening = false
    speechDetected = false
    task = nil
    request = nil
    let text = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
    if text.isEmpty {
      onNoSpeech?()
    } else {
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
