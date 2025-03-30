import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score
import pickle

# Simulating data (Replace with actual data)
num_samples = 1000  # Example dataset size
X = np.random.rand(num_samples, 4, 2).astype(np.float32)  # Shape: (samples, time steps, features)
y = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)  # Binary outcome


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, hidden_size=16, lr=0.001):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
        x = torch.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)


# Hyperparameter grid
param_grid = {
    'hidden_size': [8, 16, 32],
    'lr': [0.001, 0.0005, 0.0001]
}

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = []
best_model_state = None
best_score = -float('inf')

best_params = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for params in ParameterGrid(param_grid):
    print(f"Testing params: {params}")

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/5")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_tensor = torch.tensor(X_train).to(device)
        y_train_tensor = torch.tensor(y_train).unsqueeze(1).to(device)
        X_val_tensor = torch.tensor(X_val).to(device)
        y_val_tensor = torch.tensor(y_val).unsqueeze(1).to(device)

        model = LSTMModel(hidden_size=params['hidden_size'], lr=params['lr']).to(device)

        # Training loop
        epochs = 20
        for epoch in range(epochs):
            model.train()
            model.optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = model.criterion(outputs, y_train_tensor)
            loss.backward()
            model.optimizer.step()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            y_val_pred = (model(X_val_tensor) > 0.5).cpu().numpy().astype(int).flatten()

        mcc = matthews_corrcoef(y_val, y_val_pred)
        auprc = average_precision_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)

        fold_metrics.append({'mcc': mcc, 'auprc': auprc, 'f1': f1})

    mean_mcc = np.mean([m['mcc'] for m in fold_metrics])
    mean_auprc = np.mean([m['auprc'] for m in fold_metrics])
    mean_f1 = np.mean([m['f1'] for m in fold_metrics])

    metrics = {
        'mean_mcc_on_held_out_folds': mean_mcc,
        'mean_auprc_on_held_out_folds': mean_auprc,
        'mean_f1_on_held_out_folds': mean_f1,
        'params': params
    }
    cv_metrics.append(metrics)

    if mean_mcc > best_score:  # Save best model based on mean MCC
        best_score = mean_mcc
        best_model_state = model.state_dict()
        best_params = params

torch.save(best_model_state, 'best_lstm_model.pth')
with open('cv_metrics.pkl', 'wb') as f:
    pickle.dump(cv_metrics, f)

print(f"Best model saved as 'best_lstm_model.pth' with params {best_params}.")
print("Cross-validation metrics saved.")


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
