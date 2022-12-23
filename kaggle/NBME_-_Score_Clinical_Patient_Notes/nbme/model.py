import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class NBMEModel(nn.Module):
    def __init__(self, config_path: str = None, use_pretrained: bool = True, checkpoint: str = 'microsoft/deberta-base', 
                 output_hidden_states: bool = True, fc_dropout: float = 0.2, num_labels: int = 1, **kwargs) -> None:
        super().__init__()

        # set configuration
        if config_path is None:
            self.config = AutoConfig.from_pretrained(checkpoint, output_hidden_states=output_hidden_states)
        else:
            self.config = torch.load(config_path)

        # set backbone model
        if use_pretrained:
            self.backbone = AutoModel.from_pretrained(checkpoint, config=self.config)
        else:
            self.backbone = AutoModel(self.config)

        # set head (classification layer)
        self.fc = nn.Sequential(nn.Dropout(p=fc_dropout),
                                nn.Linear(in_features=self.config.hidden_size, out_features=num_labels, bias=True))
    
        # weights initialization for last linear layer
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=self.config.initializer_range)
                if m.bias:
                    m.bias.data.zero_()

        self.fc.apply(_init_weights)

    def get_features(self, inputs: dict) -> torch.tensor:
        # get last hidden state
        backbone_output = self.backbone(**inputs)
        last_hidden_states = backbone_output.get("last_hidden_state")
        return last_hidden_states

    def forward(self, inputs: dict) -> torch.tensor:
        # get classification results
        features = self.get_features(inputs)
        output = self.fc(features)
        return output
