import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel

torch.cuda.empty_cache()
torch.manual_seed(42)

# -------------------------------
# Step 1: Load Pretrained Models
# -------------------------------
wav2vec_processor = Wav2Vec2Processor.from_pretrained("/workspace/checkpoint-3420")
wav2vec_model = Wav2Vec2Model.from_pretrained("/workspace/checkpoint-3420")
wav2vec_model.eval()

deepseek_tokenizer = AutoTokenizer.from_pretrained("/workspace/models79%/deepseek_classification/checkpoint-9360")
deepseek_model = AutoModel.from_pretrained("/workspace/models79%/deepseek_classification/checkpoint-9360")
deepseek_model.eval()

device = torch.device("cpu")
wav2vec_model.to(device)
deepseek_model.to(device)

# -------------------------------
# Step 2: Extract Audio & Text Embeddings
# -------------------------------
class TextProjection(nn.Module):
    """Reduces DeepSeek output from 4096 to 768 dimensions."""
    def __init__(self, input_dim=4096, output_dim=768):
        super(TextProjection, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

text_projection = TextProjection().to(device)

def get_audio_embedding(audio_files):
    waveforms = []
    
    for file in audio_files:
        waveform, sample_rate = torchaudio.load(file)  
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)  
        waveform = waveform.squeeze(0)  
        waveforms.append(waveform)

    waveforms = torch.stack(waveforms)
    inputs = wav2vec_processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.squeeze(1)  

    with torch.no_grad():
        outputs = wav2vec_model(input_values.to(device))

    return outputs.last_hidden_state.mean(dim=1)  

def get_text_embedding(texts):
    inputs = deepseek_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = deepseek_model(**inputs.to(device))
    
    embeddings = outputs.last_hidden_state.mean(dim=1)  
    projected_embeddings = text_projection(embeddings)  

    return projected_embeddings

# -------------------------------
# Step 3: Cross-Attention Fusion Model with Softmax Weighting
# -------------------------------
class CrossAttentionFusionModel(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=4, num_heads=8, ff_dim=2048, dropout=0.1):
        super(CrossAttentionFusionModel, self).__init__()

        self.text_self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.audio_self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        self.text_to_audio_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.audio_to_text_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        # **Learnable Weights with Softmax for Equal Contribution**
        self.modality_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # Learnable weights for [Text, Audio]

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes)
        )

    def forward(self, text_emb, audio_emb):
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            text_emb = text_emb.unsqueeze(0)  
            audio_emb = audio_emb.unsqueeze(0)  

            text_emb, _ = self.text_self_attn(text_emb, text_emb, text_emb)
            audio_emb, _ = self.audio_self_attn(audio_emb, audio_emb, audio_emb)

            text_to_audio, text_to_audio_attn_weights = self.text_to_audio_attn(text_emb, audio_emb, audio_emb)
            audio_to_text, audio_to_text_attn_weights = self.audio_to_text_attn(audio_emb, text_emb, text_emb)

            # **Softmax-based Weighting for Balance**
            modality_weights = torch.softmax(self.modality_weights, dim=0)  # Normalize [text, audio] weights
            text_weight, audio_weight = modality_weights[0], modality_weights[1]

            # Print weight values for debugging
            print(f"Text Weight: {text_weight.item():.4f}, Audio Weight: {audio_weight.item():.4f}")

            weighted_text_emb = text_weight * text_to_audio.squeeze(0)
            weighted_audio_emb = audio_weight * audio_to_text.squeeze(0)

            fused_representation = torch.cat([weighted_text_emb, weighted_audio_emb], dim=1)

            fused_representation = F.layer_norm(fused_representation, fused_representation.shape)

            logits = self.fc(fused_representation)

            audio_confidence_boost = torch.max(audio_to_text_attn_weights) * audio_weight
            text_confidence_boost = torch.max(text_to_audio_attn_weights) * text_weight
            logits[:, 3] += audio_confidence_boost  
            logits[:, :-1] += text_confidence_boost  

        return logits, text_to_audio_attn_weights, audio_to_text_attn_weights

# -------------------------------
# Step 4: Run Inference with Balanced Modality Contribution
# -------------------------------
def predict_depression(audio_files, text_inputs):
    num_classes = 4  
    fusion_model = CrossAttentionFusionModel(hidden_dim=768, num_classes=num_classes).to(device)
    fusion_model.eval()

    audio_embeddings = get_audio_embedding(audio_files).to(device)
    text_embeddings = get_text_embedding(text_inputs).to(device)

    with torch.no_grad():
        logits, text_to_audio_attn, audio_to_text_attn = fusion_model(text_embeddings, audio_embeddings)

        predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)

    weighted_text_score = logits[:, :-1].mean().item()  
    weighted_audio_score = logits[:, -1].item()  

    text_to_audio_importance = text_to_audio_attn.mean(dim=1).mean().item()
    audio_to_text_importance = audio_to_text_attn.mean(dim=1).mean().item()

    if weighted_audio_score > weighted_text_score:
        dominant_modality = "Audio had stronger influence"
    else:
        dominant_modality = "Text had stronger influence"

    depression_labels = ["Non", "Mild", "Moderate", "Severe"]
    
    for i, prediction in enumerate(predictions):
        print(f"Sample {i+1}: Predicted Depression Level: {depression_labels[prediction.item()]}")

    print(f"Attention Scores - Text → Audio: {text_to_audio_importance:.4f}, Audio → Text: {audio_to_text_importance:.4f}")
    print(f"True Weighted Influence - Text: {weighted_text_score:.4f}, Audio: {weighted_audio_score:.4f}")
    print(f"Model Decision: {dominant_modality}")

    return [depression_labels[pred.item()] for pred in predictions]

# -------------------------------
# Step 5: Run the Script
# -------------------------------
if __name__ == "__main__":
    audio_files = ["/workspace/346_10.wav"]
    text_inputs = ["I feel tired and have no motivation to do anything."]

    results = predict_depression(audio_files, text_inputs)
    print(f"Predictions: {results}")
