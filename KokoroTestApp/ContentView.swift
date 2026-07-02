import SwiftUI

/// This view provides a simple interface for text-to-speech generation.
struct ContentView: View {
  /// The view model that manages the TTS engine and audio playback
  @ObservedObject var viewModel: TestAppModel
  
  /// The text input from the user that will be converted to speech
  @State private var inputText: String = ""

  var body: some View {
    VStack {
      Spacer()
      
      // Text input field for entering speech content
      TextField("Type something to say...", text: $inputText)
        .padding()
        .background(Color(.systemGray))
        .cornerRadius(8)
        .padding(.horizontal)

      // Voice selection picker
      Picker("Selected Voice: ", selection: $viewModel.selectedVoice) {
        ForEach(viewModel.voiceNames, id: \.self) { voice in
          Text(voice)
            .foregroundStyle(Color.black)
            .tag(voice)
        }
      }
      .accentColor(.black)
      .foregroundColor(.black)
      .pickerStyle(.menu)
      .padding(.horizontal)
      .tint(.accentColor)
      .background(.gray)
      
      // Button to trigger text-to-speech synthesis
      Button {
        if !inputText.isEmpty {
          viewModel.say(inputText)
        } else {
          viewModel.say("Please type something first")
        }
      } label: {
        HStack(alignment: .center) {
          Spacer()
          Text("Say something")
            .foregroundColor(.white)
            .frame(height: 50)
          Spacer()
        }
        .background(.black)
        .padding(.horizontal)
      }

      Text("Spoken string: " + viewModel.stringToFollowTheAudio)
        .padding()
        .foregroundStyle(.black)
        .background(.white)

      // Conversation mode: tap to talk, tap again to stop; the app replies
      Button {
        viewModel.toggleListening()
      } label: {
        HStack(alignment: .center) {
          Spacer()
          Image(systemName: viewModel.speechRecognizer.isListening ? "stop.circle.fill" : "mic.circle.fill")
          Text(viewModel.speechRecognizer.isListening ? "Listening… tap to reply" : "Talk to the app")
            .frame(height: 50)
          Spacer()
        }
        .foregroundColor(.white)
        .background(viewModel.speechRecognizer.isListening ? .red : .blue)
        .padding(.horizontal)
      }

      // Live transcript while listening
      if viewModel.speechRecognizer.isListening, !viewModel.speechRecognizer.transcript.isEmpty {
        Text(viewModel.speechRecognizer.transcript)
          .padding(.horizontal)
          .foregroundStyle(.gray)
      }

      if let error = viewModel.speechRecognizer.errorMessage {
        Text(error)
          .padding(.horizontal)
          .foregroundStyle(.red)
          .font(.footnote)
      }

      // Conversation history (latest turns at the bottom)
      ScrollView {
        VStack(alignment: .leading, spacing: 6) {
          ForEach(Array(viewModel.conversation.enumerated()), id: \.offset) { _, turn in
            Text("\(turn.role): \(turn.text)")
              .foregroundStyle(turn.role == "You" ? .black : .blue)
              .frame(maxWidth: .infinity, alignment: turn.role == "You" ? .trailing : .leading)
          }
        }
        .padding(.horizontal)
      }
      .frame(maxHeight: 180)

      Spacer()
    }
    .background(.white)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}

#Preview {
  ContentView(viewModel: TestAppModel())
}
