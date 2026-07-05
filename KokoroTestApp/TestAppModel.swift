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

/// Where we are in a hands-free conversation turn.
enum ConversationState: Equatable {
  case idle       // not in a turn
  case listening  // mic open, waiting for / capturing speech
  case thinking   // generating the reply (LLM + diacritizer)
  case speaking   // playing the synthesized reply
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

  /// On-device LFM2 tool-using agent (weather/currency/web search) — set by loadModels
  private var agent: LFM2Agent?

  /// Arabic system prompt: answer known facts directly, use tools only when
  /// needed, and always reply in Arabic.
  private static let agentSystemPrompt =
    "أنت مساعد صوتي ذكي. أجب على أسئلة المعرفة العامة مباشرةً من معرفتك. "
    + "استخدم الأدوات فقط عند الضرورة: get_weather للطقس، convert_currency لتحويل "
    + "العملات، web_search للأحداث الجارية أو المعلومات التي لا تعرفها. "
    + "أجب دائماً باللغة العربية الفصحى فقط بإيجاز، حتى لو كانت نتائج الأدوات بالإنجليزية."

  /// Spoken when the agent can't produce an answer (e.g. tools returned nothing).
  private static let agentFallback = "عذراً، لم أتمكن من إيجاد إجابة."

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

  /// True while a hands-free (auto listen → reply → listen) session is active.
  @Published var isHandsFree = false

  /// Current stage of the active turn, drives the UI status indicator.
  @Published var conversationState: ConversationState = .idle

  /// Human-readable note about a tool the agent is currently running.
  @Published var toolStatus: String?

  /// Forwards nested ObservableObject changes (speechRecognizer) to SwiftUI
  private var cancellables = Set<AnyCancellable>()

  /// When the current listen turn started, and how many restarts failed fast —
  /// used to stop hands-free if recognition can't run (offline / Simulator).
  private var lastListenStart: Date?
  private var rapidListenFailures = 0

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

    // Conversation loop: a finished spoken turn → reply → speak → listen again
    speechRecognizer.onFinal = { [weak self] text in
      self?.sendText(text)
    }
    // No speech captured (VAD timeout). In hands-free, keep the mic alive —
    // but bail out if the recognizer keeps failing instantly (e.g. no network,
    // or the Simulator, where server-based Arabic recognition can't run).
    speechRecognizer.onNoSpeech = { [weak self] in
      guard let self else { return }
      guard self.isHandsFree else { self.conversationState = .idle; return }

      let elapsed = self.lastListenStart.map { Date().timeIntervalSince($0) } ?? 0
      if elapsed < 2 {
        self.rapidListenFailures += 1
        if self.rapidListenFailures >= 3 {
          self.isHandsFree = false
          self.conversationState = .idle
          return
        }
      } else {
        self.rapidListenFailures = 0  // a real (long) silence, not a failure
      }
      // Delayed restart breaks any synchronous failure recursion and throttles.
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
        guard let self, self.isHandsFree else { return }
        self.beginListeningTurn()
      }
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

    // Stage 3: LFM2 agent LLM (438 MB) + tools
    setStage("Loading chat model…", progress: 0.60)
    if let modelURL = Bundle.main.url(forResource: "lfm2_230m_fp16", withExtension: "safetensors"),
       let tokenizerURL = Bundle.main.url(forResource: "lfm2_tokenizer", withExtension: "json") {
      let model = await Task.detached(priority: .userInitiated) {
        try? LFM2Model(modelPath: modelURL, tokenizerPath: tokenizerURL)
      }.value
      if let model {
        let agent = LFM2Agent(
          model: model,
          tools: [WeatherTool(), CurrencyTool(), WebSearchTool()],
          system: Self.agentSystemPrompt,
          fallback: Self.agentFallback)
        agent.onToolUse = { [weak self] name, _ in
          DispatchQueue.main.async { self?.toolStatus = Self.toolLabel(name) }
        }
        self.agent = agent
      } else {
        logPrint("LFM2 model failed to load")
      }
    } else {
      logPrint("LFM2 resources missing — replies will echo the input")
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
   		 let warmupDiacritizer = diacritizer
    let warmupAgent = agent
    await Task.detached(priority: .userInitiated) {
      if let warmupVoice {
        _ = try? tts.generateAudio(voice: warmupVoice, language: .ar, text: "مَرْحَبًا")
      }
      _ = warmupDiacritizer?.diacritize("مرحبا")
      // A greeting won't trigger a tool call, so this only compiles kernels.
      _ = await warmupAgent?.respond(to: "مرحبا")
    }.value

    setStage("Ready", progress: 1.0)
    isReady = true
  }

  private func setStage(_ stage: String, progress: Double) {
    loadingStage = stage
    loadingProgress = progress
  }

  /// Produces the app's reply for a user turn: the LFM2 agent generates an
  /// answer (calling weather/currency/web-search tools when useful), then CATT
  /// restores tashkeel so the Nabra voice receives fully vowelized Arabic.
  /// Runs off the main thread — `voiceName` is passed in to avoid touching
  /// published state from a background context.
  private func respond(to text: String, voiceName: String) async -> String {
    var reply = text
    if let agent {
      let generated = await agent.respond(to: text)
      if !generated.isEmpty { reply = generated }
    }
    // Arabic voice path: diacritize for correct pronunciation.
    if voiceName.hasPrefix("ar_"), !ArabicDiacritizer.isDiacritized(reply), let diacritizer {
      reply = diacritizer.diacritize(reply)
    }
    return reply
  }

  /// Adds a user turn (typed or transcribed), runs the agent off the main
  /// thread, then speaks the reply. In hands-free mode the mic reopens after.
  func sendText(_ text: String) {
    guard isReady else { return }
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }

    speechRecognizer.cancel()  // stop any open mic (e.g. typed mid-listen)
    rapidListenFailures = 0    // a real turn arrived; recognition is working
    conversation.append(ChatMessage(role: .user, text: trimmed))
    conversationState = .thinking
    toolStatus = nil

    let voiceName = selectedVoice
    Task.detached(priority: .userInitiated) { [weak self] in
      guard let self else { return }
      let reply = await self.respond(to: trimmed, voiceName: voiceName)
      await MainActor.run {
        self.toolStatus = nil
        self.conversation.append(ChatMessage(role: .app, text: reply))
        self.speak(reply) { [weak self] in self?.handleTurnFinished() }
      }
    }
  }

  /// Short human-readable label for a tool that's running (shown while thinking).
  private static func toolLabel(_ name: String) -> String {
    switch name {
    case "get_weather": return "🌦️ يتحقق من الطقس…"
    case "convert_currency": return "💱 يحوّل العملة…"
    case "web_search": return "🔎 يبحث في الويب…"
    default: return "🔧 \(name)…"
    }
  }

  // MARK: - Hands-free conversation

  /// Toggles the hands-free session. On → the mic opens and each turn flows
  /// automatically (listen → reply → listen). Off → everything stops.
  func toggleHandsFree() {
    guard isReady else { return }
    if isHandsFree {
      isHandsFree = false
      speechRecognizer.cancel()
      playerNode.stop()
      stopKaraokeTimer()
      conversationState = .idle
    } else {
      isHandsFree = true
      beginListeningTurn()
    }
  }

  /// Opens the mic for a new turn (VAD ends it automatically).
  private func beginListeningTurn() {
    guard isReady else { return }
    playerNode.stop()
    stopKaraokeTimer()
    lastListenStart = Date()
    conversationState = .listening
    speechRecognizer.startListening()
  }

  /// Called after the reply finishes playing: reopen the mic if hands-free.
  private func handleTurnFinished() {
    if isHandsFree {
      beginListeningTurn()
    } else {
      conversationState = .idle
    }
  }

  /// Synthesizes `text` off the main thread, plays it, and calls `completion`
  /// once playback has actually finished (so the mic can safely reopen).
  func speak(_ text: String, then completion: (() -> Void)? = nil) {
    guard let engine = kokoroTTSEngine else { completion?(); return }
    // Stay in `.thinking` during synthesis; flip to `.speaking` when audio starts.

    // Language from voice name: 'ar_' = Arabic, 'a…' = US English, else GB.
    let language: Language = selectedVoice.hasPrefix("ar_")
      ? .ar : (selectedVoice.first == "a" ? .enUS : .enGB)
    let voice = voices[selectedVoice + ".npy"]

    Task.detached(priority: .userInitiated) { [weak self] in
      guard let self, let voice else { await MainActor.run { completion?() }; return }
      let result = try? engine.generateAudio(voice: voice, language: language, text: text)
      await MainActor.run {
        guard let (audio, tokenArray) = result, !audio.isEmpty else {
          completion?()
          return
        }
        self.playAudio(audio, tokenArray: tokenArray, then: completion)
      }
    }
  }

  /// Plays already-synthesized samples and starts the karaoke follow-along.
  private func playAudio(_ audio: [Float], tokenArray: [KokoroSwift.MToken]?, then completion: (() -> Void)?) {
    conversationState = .speaking
    let sampleRate = Double(KokoroTTS.Constants.samplingRate)
    let audioLength = Double(audio.count) / sampleRate
    print("Audio Length: " + String(format: "%.4f", audioLength))
    print("Real Time Factor: " + String(format: "%.2f", audioLength / (BenchmarkTimer.getTimeInSec(KokoroTTS.Constants.bm_TTS) ?? 1.0)))

    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audio.count)) else {
      print("Couldn't create buffer")
      completion?()
      return
    }

    buffer.frameLength = buffer.frameCapacity
    let dst = buffer.floatChannelData![0]
    audio.withUnsafeBufferPointer { buf in
      precondition(buf.baseAddress != nil)
      UnsafeMutableRawPointer(dst)
        .copyMemory(from: UnsafeRawPointer(buf.baseAddress!),
                    byteCount: buf.count * MemoryLayout<Float>.stride)
    }

    audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
    do {
      try audioEngine.start()
    } catch {
      print("Audio engine failed to start: \(error.localizedDescription)")
      completion?()
      return
    }

    // Fire `completion` when the audio has actually played out, back on main.
    playerNode.scheduleBuffer(buffer, at: nil, options: .interrupts,
                              completionCallbackType: .dataPlayedBack) { [weak self] _ in
      DispatchQueue.main.async {
        self?.stopKaraokeTimer()
        completion?()
      }
    }
    playerNode.play()

    startKaraokeTimer(tokenArray)
  }

  /// Reveals the spoken text token-by-token in sync with playback.
  private func startKaraokeTimer(_ tokenArray: [KokoroSwift.MToken]?) {
    stopKaraokeTimer()
    guard let tokenArray else { return }
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

  private func stopKaraokeTimer() {
    timer?.invalidate()
    timer = nil
  }
}
