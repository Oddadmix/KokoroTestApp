import CoreImage
import Foundation
import MLXLMCommon
import MLXVLM

/// On-device vision-language model (LiquidAI LFM2.5-VL-450M, 8-bit) via the
/// MLXVLM library. Answers a question about a camera frame. The weights are
/// fetched from Hugging Face on first load, then cached locally and run on the
/// device's GPU through MLX.
final class VisionModel {
  enum VisionError: Error { case notLoaded }

  private static let modelId = "LiquidAI/LFM2.5-VL-450M-MLX-8bit"
  private var container: ModelContainer?

  var isLoaded: Bool { container != nil }

  /// Loads (downloading on first run) the VLM. `progress` reports 0…1 download.
  ///
  /// The published 8-bit `config.json` declares `projector_use_layernorm: false`,
  /// yet the checkpoint ships `multi_modal_projector.layer_norm` weights — so the
  /// MLX loader builds no layer-norm module and weight loading throws
  /// `incompatibleItems`. We download the repo, correct that flag on disk, then
  /// load from the local directory (which skips the wrong-config download path).
  func load(progress: (@Sendable (Double) -> Void)? = nil) async throws {
    guard container == nil else { return }
    let dir = try await downloadModel(
      hub: defaultHubApi,
      configuration: ModelConfiguration(id: Self.modelId),
      progressHandler: { p in progress?(p.fractionCompleted) })
    Self.fixProjectorLayerNorm(in: dir)
    container = try await VLMModelFactory.shared.loadContainer(
      configuration: ModelConfiguration(directory: dir))
  }

  /// Rewrites `projector_use_layernorm: false → true` in the model's config.json
  /// (a targeted string edit so nothing else in the file changes). Idempotent.
  private static func fixProjectorLayerNorm(in dir: URL) {
    let cfg = dir.appendingPathComponent("config.json")
    guard var text = try? String(contentsOf: cfg, encoding: .utf8) else { return }
    for pattern in ["\"projector_use_layernorm\": false", "\"projector_use_layernorm\":false"]
    where text.contains(pattern) {
      text = text.replacingOccurrences(of: pattern,
                                       with: pattern.replacingOccurrences(of: "false", with: "true"))
      try? text.write(to: cfg, atomically: true, encoding: .utf8)
      return
    }
  }

  /// Answers `question` about `image` and returns the model's reply.
  func describe(_ image: CIImage, question: String, maxTokens: Int = 150) async throws -> String {
    guard let container else { throw VisionError.notLoaded }
    let session = ChatSession(container, generateParameters: GenerateParameters(maxTokens: maxTokens))
    return try await session.respond(to: question, image: .ciImage(image))
  }
}
