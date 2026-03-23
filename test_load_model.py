import torch
from torch import nn
from transformers import RobertaModel, RobertaPreTrainedModel, AutoTokenizer

# ---- Custom Model Class (must match training) ----
class RobertaForMultiLabelClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


# ---- Load Checkpoint ----
checkpoint_path = r"D:\FYP\Emoi-Track\outputs\roberta-base-emopillars-negative\checkpoint-31000"

model = RobertaForMultiLabelClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

print("Model loaded successfully!")
print("Num labels:", model.config.num_labels)
