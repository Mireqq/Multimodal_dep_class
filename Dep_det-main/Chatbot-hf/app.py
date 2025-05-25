import time
import pickle
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, AutoTokenizer, AutoModel
import numpy as np
import base64
import io
from pydub import AudioSegment
from flask import Flask, request, jsonify
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import traceback
import os
os.environ["HF_HOME"] = "/tmp/cache"
import logging
logging.basicConfig(level=logging.DEBUG)


# Redirect Torch Hub cache
torch.hub.set_dir("/tmp/cache")
# -------------------------------
# Configuration
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 16000 * 4  # 4 seconds of audio
STEP_SIZE = 16000  # 1 second step size
SMOOTHING_FACTOR = 0.1
STATE_FILE = '/tmp/user_state.pkl'

# -------------------------------
# Load Models from Hugging Face
# -------------------------------
# Load Whisper model for transcription
print("Loading Whisper model for transcription...")
whisper_processor = WhisperProcessor.from_pretrained(
    "openai/whisper-base",
    cache_dir="/tmp/cache"
)
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-base",
    cache_dir="/tmp/cache"
).to("cpu")

print("Loading audio model...")

wav2vec_model = Wav2Vec2Model.from_pretrained(
    "Mireq/Wav2Vec2-Dep-Classification",
    cache_dir="/tmp/cache"
).to("cpu")

print("Loading text model...")
tokenizer = AutoTokenizer.from_pretrained(
    "Mireq/DeepSeek-R1-Distill-Llama-8B-Dep-Classification",
    cache_dir="/tmp/cache"
)
text_model = AutoModel.from_pretrained(
    "Mireq/DeepSeek-R1-Distill-Llama-8B-Dep-Classification",
    cache_dir="/tmp/cache"
).to("cpu")

print("Loading fine-tuned fusion model...")
checkpoint = torch.hub.load_state_dict_from_url(
    "https://huggingface.com/Mireq/Wav2VecDeepSeekLlama8BFusionBiLSTM-Dep-Detection/resolve/main/best_fusion_model.pth",
    map_location=DEVICE,
    model_dir="/tmp/cache"
)


# -------------------------------
# Fusion Model with Projection Layer
# -------------------------------
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        # Projection layer for text embeddings
        self.text_proj = nn.Linear(768, 4096)

        self.audio_lstm = nn.LSTM(768, 512, batch_first=True, bidirectional=True, num_layers=2)
        self.text_lstm = nn.LSTM(4096, 512, batch_first=True, bidirectional=True, num_layers=2)

        self.audio_norm = nn.LayerNorm(512 * 2)
        self.text_norm = nn.LayerNorm(512 * 2)
        self.fusion_norm = nn.LayerNorm(512 * 4)

        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=8, dropout=0.4, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512 * 4, 4)

        # Separate linear layers
        self.fc_audio = nn.Linear(512 * 2, 4)
        self.fc_text = nn.Linear(512 * 2, 4)
        self.activation = nn.ReLU()

    def forward(self, audio_emb, text_emb):
        # --- Combined Audio and Text ---
        if audio_emb.numel() > 0 and text_emb.numel() > 0:
            # Reshape before projection: combine batch and seq. length dimensions
            original_shape = text_emb.shape
            text_emb = text_emb.view(-1, text_emb.size(-1))  # (batch_size * seq_len, embedding_size)
        
            if text_emb.size(-1) == 768:
                text_emb = self.text_proj(text_emb)  
        
            # Restore original shape (except for the embedding dimension)
            text_emb = text_emb.view(original_shape[0], original_shape[1], -1)

            if text_emb.dim() == 2:
                text_emb = text_emb.unsqueeze(1)  

            audio_out, _ = self.audio_lstm(audio_emb)
            text_out, _ = self.text_lstm(text_emb)
            audio_out = self.audio_norm(audio_out)
            text_out = self.text_norm(text_out)
            fusion_input = torch.cat((audio_out[:, -1, :].unsqueeze(1), text_out[:, -1, :].unsqueeze(1)), dim=1)
            fusion_output, _ = self.cross_attn(fusion_input, fusion_input, fusion_input)
            fusion_output = fusion_output.view(fusion_output.size(0), -1)
            fusion_output = self.fusion_norm(fusion_output)
            fusion_output = self.activation(fusion_output)
            fusion_output = self.dropout(fusion_output)
            logits = self.fc(fusion_output)

        # --- Audio Only ---
        elif audio_emb.numel() > 0:
            audio_out, _ = self.audio_lstm(audio_emb)
            audio_out = self.audio_norm(audio_out[:, -1, :])
            audio_out = self.activation(audio_out)
            audio_out = self.dropout(audio_out)
            logits = self.fc_audio(audio_out)

        # --- Text Only ---
        elif text_emb.numel() > 0:
            original_shape = text_emb.shape
            text_emb = text_emb.view(-1, text_emb.size(-1))
            if text_emb.size(-1) == 768:
                text_emb = self.text_proj(text_emb)  
            text_emb = text_emb.view(original_shape[0], original_shape[1], -1)

            if text_emb.dim() == 2:
                text_emb = text_emb.unsqueeze(1)

            text_out, _ = self.text_lstm(text_emb)

            text_out = self.text_norm(text_out[:, -1, :])
            text_out = self.activation(text_out)
            text_out = self.dropout(text_out)
            logits = self.fc_text(text_out)

        # --- No Input ---
        else:
            logits = torch.zeros(1, 4, device=DEVICE)

        return logits


fusion_model = FusionModel().to(DEVICE)
fusion_model.load_state_dict(checkpoint, strict=False)
fusion_model.eval()

# -------------------------------
# Sliding Window for Audio
# -------------------------------
def apply_sliding_window(waveform, window_size, step_size):
    num_samples = waveform.shape[-1]
    windows = []
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(waveform[:, :, start:end])

    if not windows:
        padding_size = window_size - num_samples
        padded_waveform = nn.functional.pad(waveform, (0, padding_size))
        windows.append(padded_waveform)
    else:
        max_len = max(w.shape[-1] for w in windows)
        padded_windows = []
        for w in windows:
            if w.shape[-1] < max_len:
                padding_needed = max_len - w.shape[-1]
                padded_w = nn.functional.pad(w, (0, padding_needed))
                padded_windows.append(padded_w)
            else:
                padded_windows.append(w)
        windows = padded_windows
    return torch.stack(windows)


# -------------------------------
# Extract Audio & Text Embeddings
# -------------------------------
def extract_audio_embeddings(audio_input):
    if audio_input is None:
        return torch.tensor([]).to(DEVICE)

    if audio_input.dim() == 1:
        audio_input = audio_input.unsqueeze(0)

    if audio_input.shape[0] > 1:
        audio_input = audio_input.mean(dim=0, keepdim=True)

    audio_input = audio_input.unsqueeze(0).to(DEVICE)

    windows = apply_sliding_window(audio_input, WINDOW_SIZE, STEP_SIZE)
    windows = windows.to(DEVICE)

    embeddings = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16):
        for window in windows:
            if window.dim() == 3:
                window = window.squeeze(0)
            emb = wav2vec_model(window).last_hidden_state
            embeddings.append(emb)

    if not embeddings:
        return torch.tensor([]).to(DEVICE)

    audio_embedding = torch.cat(embeddings, dim=1).mean(dim=1, keepdim=True)
    return audio_embedding.float()

def extract_text_embeddings(text):
    if not text:
        return torch.empty(1, 1, 768).to(DEVICE)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        text_emb = text_model(**inputs).last_hidden_state

    return text_emb.float()


# -------------------------------
# Persistent Emotional State Tracking
# -------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(state, f)

user_state = load_state()

def update_emotional_state(user_id, prediction):
    user_state[user_id] = (SMOOTHING_FACTOR * prediction) + ((1 - SMOOTHING_FACTOR) * user_state.get(user_id, np.zeros(4)))
    save_state(user_state)
    return user_state[user_id]

# -------------------------------
# Generate Response
# -------------------------------
severity_mapping = {0: "non", 1: "mild", 2: "moderate", 3: "severe"}
responses = {
    0: "You're doing fine.",
    1: "You're feeling mildly down. Let's talk about it.",
    2: "You're showing signs of moderate distress.",
    3: "You're showing severe signs of distress. Please reach out for help."
}

def resample(waveform, target_sample_rate=16000):
    return torchaudio.functional.resample(waveform, orig_freq=waveform.shape[-1], new_freq=target_sample_rate)

def transcribe_audio(waveform, sample_rate=48000):
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        sample_rate = 16000

    inputs = whisper_processor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    # Avoid ambiguous tensor condition
    if "input_features" in inputs:
        model_inputs = inputs["input_features"]
    elif "input_values" in inputs:
        model_inputs = inputs["input_values"]
    else:
        raise ValueError("Expected 'input_features' or 'input_values' in Whisper inputs.")

    with torch.no_grad():
        generated_ids = whisper_model.generate(model_inputs)

    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


def run_prediction(audio_input, text, user_id):
    start_time = time.time()

    waveform = None
    if isinstance(audio_input, str): 
        try:
            waveform, sample_rate = torchaudio.load(audio_input)
            if waveform.numel() == 0:
                raise ValueError("Empty waveform loaded.")
            if sample_rate != 16000:
                waveform = resample(waveform, target_sample_rate=16000)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Normalise audio
            waveform = waveform / torch.max(torch.abs(waveform))
            
        except Exception as e:
            print(f"Error loading audio: {e}")
            return "Error", f"Could not load or process audio file. Error: {e}", "N/A"

    elif isinstance(audio_input, np.ndarray):  # Recorded audio (NumPy array)
        try:
            # Convert to Tensor and move to correct device
            waveform = torch.tensor(np.frombuffer(base64.b64decode(audio_input), dtype=np.float32)).to(DEVICE)

            # Fix shape: (channels, samples) -> (batch, channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] == 1:
                waveform = waveform.unsqueeze(0)

            # Convert to mono (if stereo)
            if waveform.shape[1] > 1:
                waveform = torch.mean(waveform, dim=1, keepdim=True)

            # Resample to 16 kHz if needed
            if waveform.shape[-1] != 16000:
                waveform = resample(waveform, target_sample_rate=16000)

            # Normalise audio (avoid silent recording)
            waveform = waveform / torch.max(torch.abs(waveform))
            
            if waveform.numel() == 0:
                raise ValueError("Empty waveform loaded.")
        except Exception as e:
            print(f"Error processing recorded audio: {e}")
            return "Error", f"Could not process recorded audio. Error: {e}", "N/A"

    audio_emb = extract_audio_embeddings(waveform) if waveform is not None else torch.tensor([]).to(DEVICE)
    text_emb = extract_text_embeddings(text)

    with torch.no_grad():
        output = fusion_model(audio_emb, text_emb)

    severity = torch.argmax(output, dim=1).item()
    updated_state = update_emotional_state(user_id, output.squeeze().cpu().numpy())
    latency = f"{(time.time() - start_time):.2f} s"
    response = responses[severity]

    return severity_mapping[severity], response, latency


app = Flask(__name__)

# Severity mapping
SEVERITY_MAP = {
    "non": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3
}
REVERSE_MAP = {v: k for k, v in SEVERITY_MAP.items()}
severity_scores = []

@app.route("/send_message", methods=["POST"])
def send_message():
    try:
        print("Received request at /send_message")

        # Parse JSON data from request
        try:
            data = request.get_json(force=True)
            if not data:
                print("Empty or malformed JSON.")
                return jsonify({"error": "Invalid JSON input"}), 400
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            traceback.print_exc()
            return jsonify({"error": "Failed to parse JSON"}), 400

        # Extract inputs
        text = data.get("text", "").strip()
        audio_base64 = data.get("audio")

        print(f"Text received: '{text}'")
        print(f"Audio received: {'Yes' if audio_base64 else 'No'}")

        # Handle empty input
        if not text and not audio_base64:
            print("No text or audio provided.")
            return jsonify({
                "message": "Please enter text or provide an audio recording.",
                "severity": "N/A",
                "transcription": ""
            }), 200

        waveform = None
        transcription = text  # fallback

        # Handle audio input
        if audio_base64:
            try:
                print("Decoding base64 audio...")
                audio_bytes = base64.b64decode(audio_base64)
                audio_buffer = io.BytesIO(audio_bytes)

                # Convert non-WAV formats like webm to WAV using pydub
                print("Converting recorded audio to WAV using pydub...")
                audio_segment = AudioSegment.from_file(audio_buffer)
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                waveform, sample_rate = torchaudio.load(wav_io)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                waveform = waveform / torch.max(torch.abs(waveform))
                transcription = transcribe_audio(waveform, sample_rate)
                print(f"Transcription: {transcription}")

            except Exception as e:
                print("Audio decoding/transcription failed:")
                traceback.print_exc()
                return jsonify({
                    "message": "Audio processing failed.",
                    "severity": "N/A",
                    "transcription": text
                }), 500
                
        if text.lower() == "end chat":
            print("User ended chat.")
            return end_chat()
            
        # Combine text and transcription
        combined_input = f"{text} {transcription}".strip()

        print("Running model prediction...")
        
        severity, response_msg, latency = run_prediction(waveform if audio_base64 else None, combined_input, user_id="frontend_user")
        print(f" Prediction: severity={severity}, message={response_msg}, latency={latency}")

        if severity in SEVERITY_MAP:
            severity_scores.append(SEVERITY_MAP[severity])
        else:
            severity = "N/A"

        return jsonify({
            "message": response_msg,
            "severity": severity,
            "transcription": transcription,
            "latency": latency
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "message": "Internal server error.",
            "severity": "N/A",
            "transcription": ""
        }), 500

@app.route("/end_chat", methods=["POST"])
def end_chat():
    try:
        if not severity_scores:
            return jsonify({
                "summary": "No responses recorded.",
                "average_severity": "N/A"
            })

        avg_score = round(sum(severity_scores) / len(severity_scores))
        average_severity = REVERSE_MAP.get(avg_score, "N/A")
        severity_scores.clear()

        return jsonify({
            "summary": "Thank you for talking with me.",
            "average_severity": average_severity
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True, use_reloader=False)
