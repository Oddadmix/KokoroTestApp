import AVFoundation
import CoreImage
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

  /// On-device Emhotob-50M Arabic tool-calling agent — set by loadModels.
  /// Tools: password, BMI, tip, live exchange rate. Built from a from-scratch
  /// 50M model; a keyword router exposes one tool per turn (else plain chat).
  private var agent: EmhotobAgent?

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

  /// On-device vision-language model (LFM2-VL) — loaded lazily the first time
  /// camera mode opens.
  let vision = VisionModel()

  /// Live back-camera feed used in camera mode.
  let camera = CameraController()

  /// True while the camera is open and each spoken question is answered about
  /// what the camera sees (instead of the normal chat agent).
  @Published var isCameraMode = false

  /// True while the VLM weights are downloading/loading (first camera open).
  @Published var visionLoading = false

  /// VLM download/load progress in 0…1 (only meaningful while `visionLoading`).
  @Published var visionProgress: Double = 0

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

    // Conversation loop: a finished spoken turn → reply → speak → listen again.
    // In camera mode the question is answered about what the camera sees.
    speechRecognizer.onFinal = { [weak self] text in
      guard let self else { return }
      if self.isCameraMode { self.askAboutImage(text) } else { self.sendText(text) }
    }
    // No speech captured (VAD timeout). In hands-free, keep the mic alive —
    // but bail out if the recognizer keeps failing instantly (e.g. no network,
    // or the Simulator, where server-based Arabic recognition can't run).
    speechRecognizer.onNoSpeech = { [weak self] in
      guard let self else { return }
      guard self.autoListen else { self.conversationState = .idle; return }

      let elapsed = self.lastListenStart.map { Date().timeIntervalSince($0) } ?? 0
      if elapsed < 2 {
        self.rapidListenFailures += 1
        if self.rapidListenFailures >= 3 {
          // Recognition can't run (offline / Simulator): stop the mic loop.
          self.isHandsFree = false
          if self.isCameraMode { self.closeCamera() }
          self.conversationState = .idle
          return
        }
      } else {
        self.rapidListenFailures = 0  // a real (long) silence, not a failure
      }
      // Delayed restart breaks any synchronous failure recursion and throttles.
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
        guard let self, self.autoListen else { return }
        self.beginListeningTurn()
      }
    }
    speechRecognizer.requestAuthorization()
    speechRecognizer.objectWillChange
      .sink { [weak self] _ in self?.objectWillChange.send() }
      .store(in: &cancellables)
    // Surface camera state (isRunning / permissionDenied) to SwiftUI.
    camera.objectWillChange
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

    // Stage 3: Emhotob-50M tool-calling agent (104 MB) + tools
    setStage("Loading chat model…", progress: 0.60)
    if let modelURL = Bundle.main.url(forResource: "emhotob_50m_fp16", withExtension: "safetensors"),
       let tokenizerURL = Bundle.main.url(forResource: "emhotob_tokenizer", withExtension: "json") {
      let model = await Task.detached(priority: .userInitiated) {
        try? NawahLLM(modelPath: modelURL, tokenizerPath: tokenizerURL)
      }.value
      if let model {
        let agent = EmhotobAgent(
          model: model,
          // Ordered specific → general (keyword router picks the first match).
          tools: [PrayerTimesTool(), HijriDateTool(), CalculateZakatTool(),
                  ConvertTemperatureTool(), ConvertLengthTool(), ConvertWeightTool(),
                  CalculateAgeTool(), CalculateDiscountTool(), CalculateVATTool(),
                  SplitBillTool(), CalculateTipTool(), CalculatePercentageTool(),
                  SimpleInterestTool(), DaysUntilTool(), DayOfWeekTool(), CountWordsTool(),
                  CalculateSpeedTool(), RandomNumberTool(), FlipCoinTool(),
                  EmhotobWeatherTool(), GeneratePasswordTool(), CalculateBMITool(),
                  ExchangeRateTool()])
        agent.onToolUse = { [weak self] name, _ in
          DispatchQueue.main.async { self?.toolStatus = Self.toolLabel(name) }
        }
        self.agent = agent
      } else {
        logPrint("Emhotob model failed to load")
      }
    } else {
      logPrint("Emhotob resources missing — replies will echo the input")
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

  /// Produces the app's reply for a user turn: the Emhotob agent generates an
  /// answer (calling a password/BMI/tip/exchange tool when useful), then CATT
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
    case "generate_password": return "🔑 ينشئ كلمة مرور…"
    case "calculate_bmi": return "⚖️ يحسب مؤشر الكتلة…"
    case "calculate_tip": return "🧾 يحسب البقشيش…"
    case "get_exchange_rate": return "💱 يجلب سعر الصرف…"
    case "get_prayer_times": return "🕌 يجلب مواقيت الصلاة…"
    case "convert_to_hijri": return "🌙 يحوّل للتاريخ الهجري…"
    case "calculate_zakat": return "🤲 يحسب الزكاة…"
    case "convert_temperature": return "🌡️ يحوّل درجة الحرارة…"
    case "calculate_age": return "🎂 يحسب العمر…"
    case "calculate_discount": return "🏷️ يحسب الخصم…"
    case "calculate_vat": return "🧮 يحسب الضريبة…"
    case "calculate_percentage": return "٪ يحسب النسبة…"
    case "days_until": return "📅 يحسب الأيام المتبقية…"
    case "random_number": return "🎲 يختار رقمًا…"
    case "split_bill": return "🧾 يقسّم الفاتورة…"
    case "convert_length": return "📏 يحوّل الطول…"
    case "convert_weight": return "⚖️ يحوّل الوزن…"
    case "calculate_simple_interest": return "🏦 يحسب الفائدة…"
    case "count_words": return "🔢 يعدّ الكلمات…"
    case "day_of_week": return "🗓️ يحدّد اليوم…"
    case "calculate_speed": return "🚗 يحسب السرعة…"
    case "flip_coin": return "🪙 يقلب العملة…"
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

  /// Called after the reply finishes playing: reopen the mic if a mic loop
  /// (hands-free voice chat or camera mode) is active.
  private func handleTurnFinished() {
    if autoListen {
      beginListeningTurn()
    } else {
      conversationState = .idle
    }
  }

  /// True while any mode that auto-reopens the mic after each turn is active.
  private var autoListen: Bool { isHandsFree || isCameraMode }

  // MARK: - Camera (vision) mode

  /// Opens or closes camera mode. On open: start the camera, load the VLM if
  /// needed (showing progress), then begin listening. Each spoken question is
  /// answered about what the camera currently sees, spoken back in Nabra's voice.
  func toggleCamera() {
    guard isReady else { return }
    if isCameraMode { closeCamera(); return }

    // Camera mode and voice-only hands-free are mutually exclusive.
    if isHandsFree { isHandsFree = false; speechRecognizer.cancel() }
    isCameraMode = true
    camera.start()

    if vision.isLoaded {
      beginListeningTurn()
    } else {
      visionLoading = true
      visionProgress = 0
      Task { [weak self] in
        guard let self else { return }
        do {
          try await self.vision.load { fraction in
            Task { @MainActor in self.visionProgress = fraction }
          }
        } catch {
          logPrint("VLM failed to load: \(error.localizedDescription)")
        }
        self.visionLoading = false
        if self.isCameraMode { self.beginListeningTurn() }
      }
    }
  }

  /// Tears down camera mode: stop the mic, playback, and the camera feed.
  private func closeCamera() {
    isCameraMode = false
    speechRecognizer.cancel()
    playerNode.stop()
    stopKaraokeTimer()
    camera.stop()
    conversationState = .idle
  }

  /// Snaps the current camera frame and asks the VLM `question` about it, then
  /// speaks the answer. In camera mode the mic reopens afterwards.
  func askAboutImage(_ question: String) {
    guard isReady else { return }
    let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }

    speechRecognizer.cancel()
    rapidListenFailures = 0
    conversation.append(ChatMessage(role: .user, text: trimmed))
    conversationState = .thinking
    toolStatus = "👁️ ينظر إلى ما تراه الكاميرا…"

    let frame = camera.snapshot()
    let voiceName = selectedVoice
    Task { [weak self] in
      guard let self else { return }
      let reply = await self.answerAboutImage(question: trimmed, frame: frame, voiceName: voiceName)
      self.toolStatus = nil
      self.conversation.append(ChatMessage(role: .app, text: reply))
      self.speak(reply) { [weak self] in self?.handleTurnFinished() }
    }
  }

  /// Runs the VLM on `frame` and returns an Arabic, diacritized answer ready
  /// for the Nabra voice. `await`ing the model yields the main actor, so the UI
  /// stays responsive during generation.
  private func answerAboutImage(question: String, frame: CIImage?, voiceName: String) async -> String {
    guard let frame else { return diacritizedForVoice("لَمْ أَلْتَقِطْ صُورَةً بَعْد، حَاوِلْ مَرَّةً أُخْرَى.", voiceName) }
    let prompt = "انظر إلى الصورة وأجب بإيجاز باللغة العربية عن السؤال التالي: \(question)"
    var reply: String
    do {
      reply = try await vision.describe(frame, question: prompt)
    } catch {
      reply = "عذرًا، تعذّر تحليل الصورة."
    }
    reply = reply.trimmingCharacters(in: .whitespacesAndNewlines)
    if reply.isEmpty { reply = "لا أستطيع وصف ذلك." }
    return diacritizedForVoice(reply, voiceName)
  }

  /// Restores tashkeel for an Arabic voice so pronunciation is correct.
  private func diacritizedForVoice(_ text: String, _ voiceName: String) -> String {
    if voiceName.hasPrefix("ar_"), !ArabicDiacritizer.isDiacritized(text), let diacritizer {
      return diacritizer.diacritize(text)
    }
    return text
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
