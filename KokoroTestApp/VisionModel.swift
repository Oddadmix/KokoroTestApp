import CoreImage
import Foundation
import Hub
import MLXLMCommon
import MLXVLM

/// On-device vision-language model (LiquidAI LFM2.5-VL-450M, 8-bit) via the
/// MLXVLM library. Answers a question about a camera frame.
///
/// The weights (~450 MB) are fetched from Hugging Face on first load into a
/// durable, backup-excluded directory under Application Support. On later app
/// launches the model is loaded straight from disk with no network round-trip.
final class VisionModel {
  enum VisionError: Error { case notLoaded }

  private static let modelId = "LiquidAI/LFM2.5-VL-450M-MLX-8bit"
  private var container: ModelContainer?

  var isLoaded: Bool { container != nil }

  /// Hub configured to store snapshots in Application Support (not the default
  /// `Caches`, which iOS purges under storage pressure — forcing a re-download).
  private static let hub: HubApi = {
    let fm = FileManager.default
    let base = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask,
                            appropriateFor: nil, create: true))
      ?? fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    var hfBase = base.appendingPathComponent("huggingface", isDirectory: true)
    try? fm.createDirectory(at: hfBase, withIntermediateDirectories: true)
    // Keep the large re-downloadable model out of iCloud/device backups.
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    try? hfBase.setResourceValues(values)
    return HubApi(downloadBase: hfBase)
  }()

  /// Loads the VLM. `progress` reports 0…1 during the first-run download; on a
  /// cached launch it jumps to 1 and the model loads from disk.
  ///
  /// The published 8-bit `config.json` declares `projector_use_layernorm: false`,
  /// yet the checkpoint ships `multi_modal_projector.layer_norm` weights — so the
  /// MLX loader builds no layer-norm module and weight loading throws
  /// `incompatibleItems`. We correct that flag on disk, then load from the local
  /// directory (which also avoids re-fetching the wrong config).
  func load(progress: (@Sendable (Double) -> Void)? = nil) async throws {
    guard container == nil else { return }

    let config = ModelConfiguration(id: Self.modelId)
    let dir = config.modelDirectory(hub: Self.hub)
    let readyMarker = dir.appendingPathComponent(".ready")

    if FileManager.default.fileExists(atPath: readyMarker.path) {
      // Already downloaded + patched on a previous run — load offline from disk.
      progress?(1)
      container = try await VLMModelFactory.shared.loadContainer(
        configuration: ModelConfiguration(directory: dir))
      return
    }

    // First run: download into the durable directory, patch, mark ready, load.
    let downloaded = try await downloadModel(
      hub: Self.hub, configuration: config,
      progressHandler: { p in progress?(p.fractionCompleted) })
    Self.fixProjectorLayerNorm(in: downloaded)
    try? Data().write(to: downloaded.appendingPathComponent(".ready"))
    container = try await VLMModelFactory.shared.loadContainer(
      configuration: ModelConfiguration(directory: downloaded))
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
