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
  func load(progress: (@Sendable (Double) -> Void)? = nil) async throws {
    guard container == nil else { return }
    container = try await VLMModelFactory.shared.loadContainer(
      configuration: ModelConfiguration(id: Self.modelId),
      progressHandler: { p in progress?(p.fractionCompleted) })
  }

  /// Answers `question` about `image` and returns the model's reply.
  func describe(_ image: CIImage, question: String, maxTokens: Int = 150) async throws -> String {
    guard let container else { throw VisionError.notLoaded }
    let session = ChatSession(container, generateParameters: GenerateParameters(maxTokens: maxTokens))
    return try await session.respond(to: question, image: .ciImage(image))
  }
}
