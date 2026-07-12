# Nabra (نبرة) — On-device Arabic Voice Assistant

A SwiftUI app that runs a **fully on-device Arabic voice assistant** on iPhone using
Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework. You talk to it
hands-free, it understands, calls tools when useful, and speaks back in a natural
Arabic voice — with **no server round-trips**. It also has a **camera mode** where an
on-device vision-language model answers questions about what the camera sees.

Everything (speech recognition, the language model, diacritization, and speech
synthesis) runs locally on the device's GPU.

> **NOTE:** This app runs **only on a physical iOS device** — MLX has no Metal GPU on
> the iOS Simulator, so the app cannot run there.

## The pipeline

```
🎙  Speech (SFSpeechRecognizer, ar-SA, energy-based VAD)
      ↓  transcribed question
🧠  Emhotob-50M agent  →  one of 23 tools (weather, prayer times, converters, …)
      ↓  Arabic reply
✍️  CATT diacritizer   →  restores tashkeel for correct pronunciation
      ↓  vowelized Arabic
🔊  Kokoro (Nabra) TTS  →  spoken answer
```

Tap the mic for a hands-free loop (**listen → answer → speak → listen**), or type a
message. Tap the **camera** button to ask about what the camera sees instead.

## Features

- **Hands-free Arabic conversation** — voice-activity detection opens and closes the
  mic automatically; a live transcript, "thinking", and karaoke-style spoken text are
  shown as it goes.
- **23 built-in tools** — the 50M agent calls a tool when the request needs one
  (weather, prayer times, currency, unit converters, financial calculators, dates,
  utilities). Tool results are spoken with their **exact** computed numbers.
- **Camera vision mode** — a live preview opens; ask a question out loud, the frame at
  the moment you finish is sent to an on-device VLM, and its answer is spoken back. The
  captured frame is kept and shown in the chat. TTS and speech recognition keep working
  while the camera is open.
- **Fully diacritized speech** — CATT restores tashkeel so the Arabic voice pronounces
  words correctly.
- **On-device & private** — no network needed for the core loop; a handful of tools
  (weather, prayer times, exchange rate) fetch live data.

## On-device models

The core models are bundled in `Resources/` via **Git LFS** (~500 MB total). The vision
model is downloaded on first use and cached.

| Model | Role | Size | Source |
|---|---|---|---|
| `kokoro-v1_0.safetensors` | Kokoro-82M TTS (Arabic "Nabra" voice) | 312 MB | bundled (LFS) |
| `voices.npz` | Voice-style embeddings | 13 MB | bundled (LFS) |
| `emhotob_50m_fp16.safetensors` | Emhotob-50M tool-calling LLM (LLaMA arch) | 99 MB | bundled (LFS) |
| `emhotob_tokenizer.json` | LLM tokenizer | 2.9 MB | bundled |
| `catt_eo.safetensors` | CATT Arabic diacritizer | 72 MB | bundled (LFS) |
| `LiquidAI/LFM2.5-VL-450M-MLX-8bit` | LFM2-VL vision-language model | ~450 MB | Hugging Face, cached on first camera use |

The LLM is [`oddadmix/Emhotob-50M-GPRO-Arabic-Final`](https://hf.co/oddadmix/Emhotob-50M-GPRO-Arabic-Final);
the VLM is [`LiquidAI/LFM2.5-VL-450M-MLX-8bit`](https://hf.co/LiquidAI/LFM2.5-VL-450M-MLX-8bit).
The VLM is downloaded into Application Support (excluded from backups) and loaded
offline from disk on later launches.

## Tools

A keyword router selects **one tool per turn** from the request; the tool computes an
exact result which is spoken directly.

- **Live data:** `get_weather`, `get_prayer_times`, `get_exchange_rate`
- **Islamic:** `convert_to_hijri`, `calculate_zakat`
- **Money:** `calculate_tip`, `split_bill`, `calculate_discount`, `calculate_vat`, `calculate_percentage`, `calculate_simple_interest`
- **Converters:** `convert_temperature`, `convert_length`, `convert_weight`
- **Date & time:** `calculate_age`, `days_until`, `day_of_week`
- **Utility & fun:** `generate_password`, `random_number`, `flip_coin`, `count_words`, `calculate_speed`, `calculate_bmi`

Example prompts: **ما حالة الطقس في القاهرة؟** · **احسب البقشيش على فاتورة ٢٠٠ جنيه بنسبة ١٥٪** ·
**حوّل ٣٧ مئوية إلى فهرنهايت** · **ما مواقيت الصلاة في الرياض؟**

## Requirements

- iOS 18.0+ **on a physical device** (Apple Silicon; MLX does not run on the Simulator)
- Xcode 16+
- [Git LFS](https://git-lfs.com) (for the bundled model files)

## Build & run

1. **Install Git LFS** and clone (the ~500 MB of models download via LFS):
   ```bash
   brew install git-lfs && git lfs install
   git clone https://github.com/Oddadmix/KokoroTestApp.git
   cd KokoroTestApp
   ```
   Verify the models downloaded (not tiny LFS pointer files):
   ```bash
   ls -lah Resources/*.safetensors   # kokoro ~312MB, emhotob ~99MB, catt ~72MB
   ```

2. **Open in Xcode** and run on a connected device:
   ```bash
   open KokoroTestApp.xcodeproj
   ```
   - The `mlx-swift` package ships a Metal build plugin; when prompted, **trust/enable**
     it. For command-line builds, pass `-skipPackagePluginValidation` to `xcodebuild`.
   - Grant **microphone**, **speech recognition**, and (for camera mode) **camera**
     permissions on first launch.

3. On first launch the app loads the bundled models with a progress bar; opening the
   camera the first time downloads the ~450 MB vision model.

## Architecture

| File | Responsibility |
|---|---|
| `KokoroTestApp.swift` | App entry; MLX GPU limits |
| `ContentView.swift` | Chat UI, composer, message bubbles |
| `TestAppModel.swift` | Orchestration: model loading, hands-free loop, agent + TTS, camera routing |
| `SpeechRecognizer.swift` | Arabic speech-to-text with energy-based VAD |
| `VisionModel.swift` | LFM2-VL loading (durable cache) + image Q&A |
| `CameraController.swift` | Back-camera capture session + latest-frame buffer + preview |
| `CameraScreen.swift` | Full-screen camera + status overlay |

The speech agent, tools, diacritizer, and TTS engine live in the
[kokoro-ios / KokoroSwift](https://github.com/Oddadmix/kokoro-ios) package.

## Dependencies

Swift Package Manager:

- [kokoro-ios (KokoroSwift)](https://github.com/Oddadmix/kokoro-ios) — TTS engine, Emhotob agent + tools, CATT diacritizer, Nawah/LLaMA LLM runtime
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — Apple's MLX array framework
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) — `MLXVLM` + `MLXLMCommon` for the LFM2-VL vision model
- [eSpeakNGSwift](https://github.com/Oddadmix/eSpeakNGSwift) — phonemization for Arabic TTS
- [MLXUtilsLibrary](https://github.com/mlalma/MLXUtilsLibrary) — `.npz` voice-style reader
- [swift-transformers](https://github.com/huggingface/swift-transformers) — tokenizer + Hugging Face download (via mlx-swift-lm)

## License

Licensed under the Apache 2.0 License — see [LICENSE](LICENSE).
