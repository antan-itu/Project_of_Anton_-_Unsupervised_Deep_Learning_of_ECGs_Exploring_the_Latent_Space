import os
import json
import torch
import torch.nn as nn
import numpy as np

# ================================
# 1. Model Definition
# ================================
class ConvAutoencoder(nn.Module):
    def __init__(self, seq_len, in_channels, latent_dim, base_filters, kernel_size,
                 num_layers, pool_size, activation, dropout_rate, norm_type, pooling_type, masking_ratio=0.0):
        super(ConvAutoencoder, self).__init__()
        
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.masking_ratio = masking_ratio 
        padding = kernel_size // 2
        
        encoder_layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            filters = base_filters * (2**i)
            conv_stride = pool_size if pooling_type == 'stride' else 1
            encoder_layers.append(nn.Conv1d(current_channels, filters, kernel_size, stride=conv_stride, padding=padding))
            if norm_type == 'layer':
                encoder_layers.append(nn.GroupNorm(1, filters))
            elif norm_type == 'batch':
                encoder_layers.append(nn.BatchNorm1d(filters))
            if activation == 'leaky_relu':
                encoder_layers.append(nn.LeakyReLU())
            else:
                encoder_layers.append(nn.ReLU())
            if pooling_type == 'max':
                encoder_layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            elif pooling_type == 'average':
                encoder_layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_size))
            if dropout_rate > 0.0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            current_channels = filters
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        dummy_input = torch.zeros(1, in_channels, seq_len)
        dummy_output = self.encoder(dummy_input)
        self.shape_before_flatten = dummy_output.shape[1:]
        flattened_size = int(np.prod(self.shape_before_flatten))
        
        self.fc_latent = nn.Linear(flattened_size, latent_dim)
        self.fc_decoder_input = nn.Linear(latent_dim, flattened_size)
        
        decoder_layers = []
        if activation == 'leaky_relu':
            decoder_layers.append(nn.LeakyReLU())
        else:
            decoder_layers.append(nn.ReLU())
            
        for i in reversed(range(num_layers)):
            filters = base_filters * (2**i)
            out_channels_next = base_filters * (2**(i-1)) if i > 0 else in_channels
            if pooling_type in ['max', 'average']:
                decoder_layers.append(nn.Upsample(scale_factor=pool_size))
                conv_stride = 1
            else:
                conv_stride = pool_size
            decoder_layers.append(nn.ConvTranspose1d(current_channels, out_channels_next, kernel_size, 
                                                     stride=conv_stride, padding=padding, output_padding=conv_stride-1 if conv_stride > 1 else 0))
            if i > 0: 
                if norm_type == 'layer':
                    decoder_layers.append(nn.GroupNorm(1, out_channels_next))
                elif norm_type == 'batch':
                    decoder_layers.append(nn.BatchNorm1d(out_channels_next))
                if activation == 'leaky_relu':
                    decoder_layers.append(nn.LeakyReLU())
                else:
                    decoder_layers.append(nn.ReLU())
                if dropout_rate > 0.0:
                    decoder_layers.append(nn.Dropout(dropout_rate))
            current_channels = out_channels_next

        self.decoder = nn.Sequential(*decoder_layers)
        self.final_conv = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x):
        if self.training and self.masking_ratio > 0.0:
            mask = (torch.rand_like(x) > self.masking_ratio).float()
            x_input = x * mask
        else:
            x_input = x

        encoded = self.encoder(x_input)
        flattened = encoded.view(encoded.size(0), -1)
        latent = self.fc_latent(flattened)
        decoded_input = self.fc_decoder_input(latent)
        reshaped = decoded_input.view(decoded_input.size(0), *self.shape_before_flatten)
        decoded = self.decoder(reshaped)
        
        if decoded.size(2) > self.seq_len:
            decoded = decoded[:, :, :self.seq_len]
        elif decoded.size(2) < self.seq_len:
            pad_size = self.seq_len - decoded.size(2)
            decoded = torch.nn.functional.pad(decoded, (0, pad_size))
            
        out = self.final_conv(decoded)
        return out, latent


# ================================
# 2. Configuration & Paths
# ================================
RUN_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/experiments/GridRun_003_1804_1354"

CONFIG_PATH = os.path.join(RUN_DIR, "config.json")
MODEL_PATH = os.path.join(RUN_DIR, "best_fold_model.pth")
ONNX_PATH = os.path.join(RUN_DIR, "autoencoder_architecture.onnx")

SEQ_LEN = 5000
IN_CHANNELS = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================
# 3. Load & Export Execution
# ================================
def main():
    print(f"Loading configuration from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    print("Instantiating model architecture...")
    model = ConvAutoencoder(
        seq_len=SEQ_LEN,
        in_channels=IN_CHANNELS,
        latent_dim=config['latent_dim'],
        base_filters=config['base_filters'],
        kernel_size=config['kernel_size'],
        num_layers=config['num_layers'],
        pool_size=config['pool_size'],
        activation=config['activation'],
        dropout_rate=config['dropout_rate'],
        norm_type=config['norm_type'],
        pooling_type=config['pooling_type'],
        masking_ratio=0.0 # Set to 0.0 for inference/export
    ).to(DEVICE)

    print(f"Loading weights from: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    print("Generating ONNX graph...")
    dummy_input = torch.randn(1, IN_CHANNELS, SEQ_LEN).to(DEVICE)

    torch.onnx.export(
        model,               
        dummy_input,              
        ONNX_PATH,               
        export_params=True,       
        opset_version=11,          
        input_names=['ECG_Input'],   
        output_names=['Reconstruction', 'Latent_Vector'] 
    )
    print(f"Success! ONNX model saved to: {ONNX_PATH}")
    print("You can now drag and drop this file into netron.app in your browser.")

if __name__ == "__main__":
    main()