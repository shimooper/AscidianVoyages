import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, f1_score, average_precision_score
import pandas as pd
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
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))

# Hyperparameter grid
hyperparameter_grid = {
    'hidden_size': [8, 16, 32],
    'lr': [1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64]
}

best_model = None
best_score = -np.inf
best_params = None
all_results = []

# Grid search
for hidden_size in hyperparameter_grid['hidden_size']:
    for lr in hyperparameter_grid['lr']:
        for batch_size in hyperparameter_grid['batch_size']:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = LSTMModel(hidden_size=hidden_size, lr=lr)
            checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, dirpath='checkpoints',
                                                  filename='best_model')
            trainer = pl.Trainer(
                max_epochs=20,
                enable_checkpointing=True,
                logger=False,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min'), checkpoint_callback]
            )
            trainer.fit(model, train_loader, val_loader)

            best_model_path = checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(best_model_path))
            model.eval()

            with torch.no_grad():
                y_val_pred_probs = torch.cat([model(x) for x, _ in val_loader]).cpu().numpy().flatten()
                y_val_pred = (y_val_pred_probs > 0.5).astype(int)

            mcc = matthews_corrcoef(y_val, y_val_pred)
            f1 = f1_score(y_val, y_val_pred)
            auprc = average_precision_score(y_val, y_val_pred_probs)

            all_results.append(
                {'hidden_size': hidden_size, 'lr': lr, 'batch_size': batch_size, 'mcc': mcc, 'f1': f1, 'auprc': auprc})

            if mcc > best_score:
                best_score = mcc
                best_model = model
                best_params = {'hidden_size': hidden_size, 'lr': lr, 'batch_size': batch_size}

# Save all hyperparameter results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv('hyperparameter_results.csv', index=False)
print("Hyperparameter search results saved to 'hyperparameter_results.csv'")

print(f"Best hyperparameters: {best_params}")

# Retrain on the full dataset with the best hyperparameters
final_dataset = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
final_loader = DataLoader(final_dataset, batch_size=best_params['batch_size'], shuffle=True)

final_model = LSTMModel(hidden_size=best_params['hidden_size'], lr=best_params['lr'])
checkpoint_callback_final = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, dirpath='checkpoints',
                                            filename='best_final_model')
final_trainer = pl.Trainer(
    max_epochs=20,
    enable_checkpointing=True,
    logger=False,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min'), checkpoint_callback_final]
)
final_trainer.fit(final_model, final_loader)

# Load best final model before saving
best_final_model_path = checkpoint_callback_final.best_model_path
final_model.load_state_dict(torch.load(best_final_model_path))
final_model.eval()

# Save final model
final_model_state = final_model.state_dict()
torch.save(final_model_state, 'final_lstm_model.pth')
print("Final model saved as 'final_lstm_model.pth'")


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
