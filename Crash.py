import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class AudioConv_AE(nn.Module):
    def __init__(self):
        super(AudioConv_AE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size=26, stride=17),
            nn.Conv1d(40, 40, kernel_size=13, stride=8),
            nn.Conv1d(40, 40, kernel_size=7, stride=4),
            nn.Conv1d(40, 40, kernel_size=4, stride=2),
            nn.Hardtanh(min_val=-1.0, max_val=1.0)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(40, 40, kernel_size=4, stride=2),
            nn.ConvTranspose1d(40, 40, kernel_size=7, stride=4),
            nn.ConvTranspose1d(40, 40, kernel_size=13, stride=8),
            nn.ConvTranspose1d(40, 1, kernel_size=26, stride=17),
            nn.Hardtanh(min_val=-1.0, max_val=1.0)
        )

    def forward(self, input):
        latent = self.encoder(input)
        output = self.decoder(latent)
        output = F.pad(output, (0, input.size(2) - output.size(2)))
        return output


# Create the model instance
model = AudioConv_AE()
model.float()
model.train()
print(model)

# Generate some fake data
batch = 16
audio = 44_100
dataset = TensorDataset(torch.randn(batch, 1, audio) * 2 - 1)

def test(device):
    global dataset, model
    print(f"device={device}")
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    # Training Loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for inputs, in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


test('mps')
test('cpu')
test('mps')
test('cpu')



# Loss function
loss_function = nn.MSELoss()


# Save the model after training
torch.save(model.state_dict(), "audio_conv_ae_model.pth")
