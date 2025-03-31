import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

# Simulating data (Replace with actual data)
num_samples = 1000  # Example dataset size
X = np.random.rand(num_samples, 4, 2).astype(np.float32)  # Shape: (samples, time steps, features)
y = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)  # Binary outcome

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define LSTM model using PyTorch Lightning
class LSTMModel(pl.LightningModule):
    def __init__(self, hidden_size=16, lr=0.001):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = lr
        self.criterion = nn.BCELoss()
        self.save_hyperparameters()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
        x = torch.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)





# Load model and run inference
def load_model_and_predict(X_test):
    model = LSTMModel(hidden_size=best_params['hidden_size'], lr=best_params['lr']).to(device)
    model.load_state_dict(torch.load('final_lstm_model.pth'))
    model.eval()
    X_test_tensor = torch.tensor(X_test).to(device)
    with torch.no_grad():
        predictions = (model(X_test_tensor) > 0.5).cpu().numpy().astype(int).flatten()
    return predictions


# Example usage
X_test = np.random.rand(10, 4, 2).astype(np.float32)  # Replace with real test data
predictions = load_model_and_predict(X_test)
print("Predictions:", predictions)
