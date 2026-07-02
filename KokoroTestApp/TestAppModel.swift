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
  /// The Kokoro text-to-speech engine instance
  let kokoroTTSEngine: KokoroTTS!
  
  /// The audio engine used for playback
  let audioEngine: AVAudioEngine!
  
  /// The audio player node attached to the audio engine
  let playerNode: AVAudioPlayerNode!
  
  /// Dictionary of available voices, mapped by voice name to MLX array data
  let voices: [String: MLXArray]
  
  /// Array of voice names available for selection in the UI
  @Published var voiceNames: [String] = []
  
  /// The currently selected voice name
  @Published var selectedVoice: String = ""
  
  @Published var stringToFollowTheAudio: String = ""

  /// Speech-to-text for the conversation demo (Arabic by default)
  let speechRecognizer = SpeechRecognizer()

  /// Running transcript of the conversation (user + app turns)
  @Published var conversation: [ChatMessage] = []

  /// Forwards nested ObservableObject changes (speechRecognizer) to SwiftUI
  private var cancellables = Set<AnyCancellable>()

  var timer: Timer?

  /// Initializes the test app model with TTS engine, audio components, and voice data.
  init() {
    // Load the Kokoro TTS model from the app bundle
    let modelPath = Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors")!    
    kokoroTTSEngine = KokoroTTS(modelPath: modelPath)
    
    // Initialize audio engine and player node
    audioEngine = AVAudioEngine()
    playerNode = AVAudioPlayerNode()
    audioEngine.attach(playerNode)  
    
    // Load voice data from NPZ file
    let voiceFilePath = Bundle.main.url(forResource: "voices", withExtension: "npz")!
    voices = NpyzReader.read(fileFromPath: voiceFilePath) ?? [:]
    
    // Extract voice names and sort them alphabetically
    voiceNames = voices.keys.map { String($0.split(separator: ".")[0]) }.sorted(by: <)
    selectedVoice = voiceNames[0]

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
  }

  /// Produces the app's reply for a transcribed user turn.
  /// Currently echoes the user's words back; swap this for an LLM call to get
  /// real conversations. NOTE: recognized Arabic arrives undiacritized, and the
  /// Nabra voice expects tashkeel'd input — a proper reply generator should
  /// return diacritized text.
  private func respond(to text: String) -> String {
    text
  }

  /// Adds a user turn (typed or transcribed) to the conversation, generates
  /// the app's reply, and speaks it.
  func sendText(_ text: String) {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }
    conversation.append(ChatMessage(role: .user, text: trimmed))
    let reply = respond(to: trimmed)
    conversation.append(ChatMessage(role: .app, text: reply))
    say(reply)
  }

  /// Starts or stops a spoken conversation turn.
  func toggleListening() {
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
