import AVFoundation
import MLX
import SwiftUI
import KokoroSwift
import Combine
import MLXUtilsLibrary

/// One turn in the conversation, rendered as a chat bubble.
struct ChatMessage: Identifiable, Equatable {
  enum Role {
    case user, app
  }

  let id = UUID()
  let role: Role
  let text: String
}

/// The view model that manages text-to-speech functionality using the Kokoro TTS engine.
/// - Loading and managing the Kokoro TTS model
/// - Managing available voice options
/// - Audio playback using AVAudioEngine
/// - Converting text to speech audio
final class TestAppModel: ObservableObject {
  /// The Kokoro text-to-speech engine instance (set by loadModels)
  private(set) var kokoroTTSEngine: KokoroTTS!

  /// The audio engine used for playback
  let audioEngine: AVAudioEngine

  /// The audio player node attached to the audio engine
  let playerNode: AVAudioPlayerNode

  /// Dictionary of available voices, mapped by voice name to MLX array data
  private(set) var voices: [String: MLXArray] = [:]

  /// On-device Arabic diacritizer (CATT) — set by loadModels
  private var diacritizer: ArabicDiacritizer?

  /// On-device Nawah-50M Arabic chat LLM — set by loadModels
  private var llm: NawahLLM?

  /// Array of voice names available for selection in the UI
  @Published var voiceNames: [String] = []

  /// The currently selected voice name
  @Published var selectedVoice: String = ""

  @Published var stringToFollowTheAudio: String = ""

  /// True once every model is loaded and warmed up
  @Published var isReady = false

  /// Human-readable description of the current loading stage
  @Published var loadingStage = "Preparing…"

  /// Overall loading progress in 0...1 (stages weighted by model size)
  @Published var loadingProgress: Double = 0

  /// Speech-to-text for the conversation demo (Arabic by default)
  let speechRecognizer = SpeechRecognizer()

  /// Running transcript of the conversation (user + app turns)
  @Published var conversation: [ChatMessage] = []

  /// Forwards nested ObservableObject changes (speechRecognizer) to SwiftUI
  private var cancellables = Set<AnyCancellable>()

  var timer: Timer?

  /// Performs only lightweight setup; the models load asynchronously in
  /// `loadModels()` so the UI can show progress instead of freezing.
  init() {
    // Initialize audio engine and player node
    audioEngine = AVAudioEngine()
    playerNode = AVAudioPlayerNode()
    audioEngine.attach(playerNode)

    // Configure audio session for iOS (.playAndRecord so the mic works for
    // speech recognition; .defaultToSpeaker keeps TTS playback on the speaker)
    #if os(iOS)
      do {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothHFP])
        try audioSession.setActive(true)
      } catch {
        logPrint("Failed to set up AVAudioSession: \(error.localizedDescription)")
      }
    #endif

    // Conversation loop: when a spoken turn is transcribed, reply and speak it
    speechRecognizer.onFinal = { [weak self] text in
      self?.sendText(text)
    }
    speechRecognizer.requestAuthorization()
    speechRecognizer.objectWillChange
      .sink { [weak self] _ in self?.objectWillChange.send() }
      .store(in: &cancellables)

    Task { await loadModels() }
  }

  /// Loads all models off the main thread with staged progress (weights
  /// proportional to file sizes), then warms each one up so the first
  /// conversation turn doesn't stall on Metal kernel compilation.
  private func loadModels() async {
    // Stage 1: Kokoro TTS (312 MB — the bulk of the wait)
    setStage("Loading speech synthesizer…", progress: 0.02)
    let ttsURL = Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors")!
    let tts = await Task.detached(priority: .userInitiated) { KokoroTTS(modelPath: ttsURL) }.value
    kokoroTTSEngine = tts

    // Stage 2: voice embeddings
    setStage("Loading voices…", progress: 0.55)
    let voicesURL = Bundle.main.url(forResource: "voices", withExtension: "npz")!
    voices = await Task.detached { NpyzReader.read(fileFromPath: voicesURL) ?? [:] }.value
    voiceNames = voices.keys.map { String($0.split(separator: ".")[0]) }.sorted(by: <)
    selectedVoice = voiceNames.contains("ar_msa") ? "ar_msa" : (voiceNames.first ?? "")

    // Stage 3: Nawah chat LLM (104 MB)
    setStage("Loading chat model…", progress: 0.60)
    if let modelURL = Bundle.main.url(forResource: "nawah_50m_fp16", withExtension: "safetensors"),
       let tokenizerURL = Bundle.main.url(forResource: "nawah_tokenizer", withExtension: "json") {
      llm = await Task.detached { try? NawahLLM(modelPath: modelURL, tokenizerPath: tokenizerURL) }.value
    } else {
      logPrint("Nawah LLM resources missing — replies will echo the input")
    }

    // Stage 4: CATT diacritizer (72 MB)
    setStage("Loading diacritizer…", progress: 0.76)
    if let cattURL = Bundle.main.url(forResource: "catt_eo", withExtension: "safetensors") {
      diacritizer = await Task.detached { try? ArabicDiacritizer(modelPath: cattURL) }.value
    } else {
      logPrint("catt_eo.safetensors missing from bundle — Arabic text won't be diacritized")
    }

    // Stage 5: warm-up — run each model once so Metal kernels compile now
    // instead of during the first real turn
    setStage("Warming up…", progress: 0.88)
    let warmupVoice = voices[selectedVoice + ".npy"]
    let warmupLLM = llm
    let warmupDiacritizer = diacritizer
    await Task.detached(priority: .userInitiated) {
      if let warmupVoice {
        _ = try? tts.generateAudio(voice: warmupVoice, language: .ar, text: "مَرْحَبًا")
      }
      _ = warmupLLM?.reply(to: "مرحبا", maxTokens: 1)
      _ = warmupDiacritizer?.diacritize("مرحبا")
    }.value

    setStage("Ready", progress: 1.0)
    isReady = true
  }

  private func setStage(_ stage: String, progress: Double) {
    loadingStage = stage
    loadingProgress = progress
  }

  /// Produces the app's reply for a user turn: the Nawah LLM generates an
  /// Arabic answer, then CATT restores tashkeel so the Nabra voice receives
  /// fully vowelized input. Falls back to echoing when a stage is missing.
  private func respond(to text: String) -> String {
    guard selectedVoice.hasPrefix("ar_") else { return text }

    var reply = text
    if let llm {
      let generated = llm.reply(to: text)
      if !generated.isEmpty {
        reply = generated
      }
    }
    if !ArabicDiacritizer.isDiacritized(reply), let diacritizer {
      reply = diacritizer.diacritize(reply)
    }
    return reply
  }

  /// Adds a user turn (typed or transcribed) to the conversation, generates
  /// the app's reply, and speaks it.
  func sendText(_ text: String) {
    guard isReady else { return }
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }
    conversation.append(ChatMessage(role: .user, text: trimmed))
    let reply = respond(to: trimmed)
    conversation.append(ChatMessage(role: .app, text: reply))
    say(reply)
  }

  /// Starts or stops a spoken conversation turn.
  func toggleListening() {
    guard isReady else { return }
    if speechRecognizer.isListening {
      speechRecognizer.stopListening()
    } else {
      playerNode.stop()
      speechRecognizer.startListening()
    }
  }

  /// Converts the provided text to speech and plays it through the audio engine.
  /// - Parameter text: The text to be converted to speech
  func say(_ text: String) {
    // Generate audio using the selected voice
    // Language is determined by voice name: 'ar_' prefix = Arabic, 'a' prefix = US English, otherwise GB English
    let language: Language = selectedVoice.hasPrefix("ar_") ? .ar : (selectedVoice.first! == "a" ? .enUS : .enGB)
    let (audio, tokenArray) = try! kokoroTTSEngine.generateAudio(voice: voices[selectedVoice + ".npy"]!, language: language, text: text)
    
    if let tokenArray {
      for t in tokenArray {
        print("\(t.text): \(t.start_ts, default: "UNK") - \(t.end_ts, default: "UNK")")
      }
    }
    
    // Calculate audio length and performance metrics
    let sampleRate = Double(KokoroTTS.Constants.samplingRate)
    let audioLength = Double(audio.count) / sampleRate
    // Log performance metrics
    print("Audio Length: " + String(format: "%.4f", audioLength))
    print("Real Time Factor: " + String(format: "%.2f", audioLength / (BenchmarkTimer.getTimeInSec(KokoroTTS.Constants.bm_TTS) ?? 1.0)))

    // Create audio format (mono channel at the model's sample rate)
    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    
    // Create PCM buffer for the audio data
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audio.count)) else {
      print("Couldn't create buffer")
      return
    }

    // Copy audio data into the buffer
    buffer.frameLength = buffer.frameCapacity
    let channels = buffer.floatChannelData!
    let dst: UnsafeMutablePointer<Float> = channels[0]
    
    // Safely copy audio samples to the buffer
    audio.withUnsafeBufferPointer { buf in
        precondition(buf.baseAddress != nil)
        let byteCount = buf.count * MemoryLayout<Float>.stride

        UnsafeMutableRawPointer(dst)
          .copyMemory(from: UnsafeRawPointer(buf.baseAddress!), byteCount: byteCount)
    }

    // Connect the player node to the audio engine's mixer
    audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
    
    // Start the audio engine
    do {
      try audioEngine.start()
    } catch {
      print("Audio engine failed to start: \(error.localizedDescription)")
      return
    }

    // Schedule and play the audio buffer
    playerNode.scheduleBuffer(buffer, at: nil, options: .interrupts, completionHandler: nil)
    playerNode.play()
    
    if let tokenArray {
      stringToFollowTheAudio = ""
      var currentToken = 0
      var audioTime: Double = 0.0
      var added = false
      
      timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] timer in
        guard let self else { return }
        audioTime += 0.1
        
        guard currentToken < tokenArray.count else {
          timer.invalidate()
          return
        }
        
        let token = tokenArray[currentToken]
                
        if !added, let start = token.start_ts, start < audioTime {
          stringToFollowTheAudio += token.text + (token.whitespace.isEmpty ? "" : " ")
          added = true
        }
        
        if let end = token.end_ts, audioTime >= end {
          currentToken += 1
          added = false
        }
      }
    }
  }
}
