import torch
import torch.nn as nn


class SequenceBackbone(nn.Module):
    def __init__(self, feature_size=32, hidden_size=96, heads=4, layers=3, dropout=0.1):
        super().__init__()

        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        transformer_block = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.sequence_encoder = nn.TransformerEncoder(
            transformer_block,
            num_layers=layers
        )

        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        embedded_features = self.feature_embedding(inputs) 

        encoded_sequence = self.sequence_encoder(embedded_features)

        pooled_embedding = encoded_sequence.mean(dim=1)

        return self.output_norm(pooled_embedding)


class PredictionHead(nn.Module):
    def __init__(self, hidden_size=96, prediction_size=32, dropout=0.1):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_size)
        )

    def forward(self, embedding):
        logits = self.predictor(embedding)
        return torch.sigmoid(logits)


class InferenceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = SequenceBackbone()
        self.head = PredictionHead()

        self.eval()

    def get_embedding(self, inputs):
        with torch.no_grad():
            return self.backbone(inputs)

    def predict_from_embedding(self, embedding):
        with torch.no_grad():
            return self.head(embedding)

    def forward(self, inputs):
        embedding = self.get_embedding(inputs)
        prediction = self.predict_from_embedding(embedding)

        return prediction