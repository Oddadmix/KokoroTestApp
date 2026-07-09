import AVFoundation
import Combine
import CoreImage
import SwiftUI

/// Runs the back camera, keeps the most recent frame, and exposes a live
/// preview. `snapshot()` returns the current frame as a CIImage for the VLM.
///
/// The type is `@MainActor` (the project's default isolation) for its
/// `@Published` UI state, but the capture pipeline — the sample-buffer delegate
/// callback, the frame buffer, and session configuration — is `nonisolated`
/// because AVFoundation drives it from a background dispatch queue.
final class CameraController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
  nonisolated(unsafe) let session = AVCaptureSession()
  nonisolated(unsafe) private let output = AVCaptureVideoDataOutput()
  private let sessionQueue = DispatchQueue(label: "camera.session")
  private let lock = NSLock()
  nonisolated(unsafe) private var latest: CIImage?

  @Published var isRunning = false
  @Published var permissionDenied = false

  /// Requests permission and starts the session (idempotent).
  func start() {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
      sessionQueue.async { self.configureAndStart() }
    case .notDetermined:
      AVCaptureDevice.requestAccess(for: .video) { granted in
        if granted { self.sessionQueue.async { self.configureAndStart() } }
        else { Task { @MainActor in self.permissionDenied = true } }
      }
    default:
      permissionDenied = true
    }
  }

  func stop() {
    sessionQueue.async { if self.session.isRunning { self.session.stopRunning() } }
    isRunning = false
  }

  /// The most recently captured camera frame, if any.
  func snapshot() -> CIImage? {
    lock.lock(); defer { lock.unlock() }
    return latest
  }

  // MARK: - nonisolated capture pipeline (runs on `sessionQueue`)

  nonisolated private func configureAndStart() {
    guard !session.isRunning else { return }
    session.beginConfiguration()
    session.sessionPreset = .high
    // Keep our .playAndRecord audio session intact so TTS + speech recognition
    // keep working while the camera is open (we capture video only).
    session.usesApplicationAudioSession = true
    session.automaticallyConfiguresApplicationAudioSession = false
    if session.inputs.isEmpty,
       let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
       let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) {
      session.addInput(input)
    }
    if session.outputs.isEmpty, session.canAddOutput(output) {
      output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
      output.setSampleBufferDelegate(self, queue: sessionQueue)
      session.addOutput(output)
      // Deliver upright portrait frames so the VLM sees the scene correctly.
      if let conn = output.connection(with: .video), conn.isVideoRotationAngleSupported(90) {
        conn.videoRotationAngle = 90
      }
    }
    session.commitConfiguration()
    session.startRunning()
    Task { @MainActor in self.isRunning = true }
  }

  nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                                 from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    let image = CIImage(cvPixelBuffer: pixelBuffer)
    lock.lock(); latest = image; lock.unlock()
  }
}

/// SwiftUI wrapper around an AVCaptureVideoPreviewLayer.
struct CameraPreview: UIViewRepresentable {
  let session: AVCaptureSession

  func makeUIView(context: Context) -> PreviewView {
    let view = PreviewView()
    view.previewLayer.session = session
    view.previewLayer.videoGravity = .resizeAspectFill
    return view
  }
  func updateUIView(_ uiView: PreviewView, context: Context) {}

  final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
  }
}
