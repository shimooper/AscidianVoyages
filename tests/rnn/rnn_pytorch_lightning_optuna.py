import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score
import pickle

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
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


# Hyperparameter optimization with Optuna
def objective(trial):
    hidden_size = trial.suggest_categorical('hidden_size', [8, 16, 32])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mean_mcc = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = LSTMModel(hidden_size=hidden_size, lr=lr)
        trainer = pl.Trainer(max_epochs=20, enable_checkpointing=False, logger=False)
        trainer.fit(model, train_loader)

        model.eval()
        with torch.no_grad():
            y_val_pred = torch.cat([model(x) for x, _ in val_loader]).cpu().numpy().astype(int).flatten()

        mcc = matthews_corrcoef(y_val, y_val_pred)
        mean_mcc.append(mcc)

    return np.mean(mean_mcc)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Train final model with best hyperparameters
best_model = LSTMModel(hidden_size=best_params['hidden_size'], lr=best_params['lr'])
best_model_state = best_model.state_dict()
torch.save(best_model_state, 'best_lstm_model.pth')

print("Best model saved as 'best_lstm_model.pth'")


# Load model and run inference
def load_model_and_predict(X_test):
    model = LSTMModel(hidden_size=best_params['hidden_size'], lr=best_params['lr']).to(device)
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    model.eval()
    X_test_tensor = torch.tensor(X_test).to(device)
    with torch.no_grad():
        predictions = (model(X_test_tensor) > 0.5).cpu().numpy().astype(int).flatten()
    return predictions


# Example usage
X_test = np.random.rand(10, 4, 2).astype(np.float32)  # Replace with real test data
predictions = load_model_and_predict(X_test)
print("Predictions:", predictions)
