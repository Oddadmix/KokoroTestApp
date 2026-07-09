import SwiftUI
import MLX

/// The main application entry point for the Kokoro TTS test app.
/// This app demonstrates the Kokoro text-to-speech engine with MLX acceleration.
@main
struct KokoroTestApp: App {
  /// The main view model that manages the TTS engine and application state
  let model = TestAppModel()
    
  /// Initializes the application and configures MLX GPU settings.
  init() {
    // MLX GPU limits. The resident model weights alone are ~940 MB (Kokoro 312
    // + Emhotob 104 + CATT 72 + LFM2-VL ~450), so the old 900 MB memory limit
    // was below the working set and forced constant eviction during VLM
    // inference. Raise it to leave headroom for the vision tower's activations.
    GPU.set(cacheLimit: 64 * 1024 * 1024)
    GPU.set(memoryLimit: 1800 * 1024 * 1024)
  }

  var body: some Scene {
    WindowGroup {
      ContentView(viewModel: model)
    }
  }
}
