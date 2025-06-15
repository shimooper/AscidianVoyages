import pandas as pd
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analyze_expirement_results.utils import convert_data_to_tensor_for_rnn


# Define LSTM model using PyTorch Lightning
class LSTMModel(L.LightningModule):
    def __init__(self, hidden_size=16, num_layers=1, lr=0.001):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
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
        y_hat = self(x).flatten()
        loss = self.criterion(y_hat, y.float())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).detach().cpu().flatten()
        loss = self.criterion(y_hat, y.float())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

        y_pred = (y_hat > 0.5).numpy().astype(int)

        val_mcc = matthews_corrcoef(y, y_pred)
        val_f1 = f1_score(y, y_pred)
        val_auprc = average_precision_score(y, y_hat)

        self.log('val_mcc', val_mcc, prog_bar=True, on_epoch=True)  # Log MCC
        self.log('val_f1', val_f1, prog_bar=True, on_epoch=True)
        self.log('val_auprc', val_auprc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def train_lstm_with_hyperparameters_train_val(logger, config, output_dir, hidden_size, num_layers, lr, batch_size,
                                              X_train, y_train, X_val, y_val, device):
    if config.balance_classes:
        logger.info(f"Resampling train examples using RandomUnderSampler. Before: {y_train.value_counts()}")
        rus = RandomUnderSampler(random_state=config.random_state, sampling_strategy=config.max_classes_ratio)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        logger.info(f"After resampling: {y_train.value_counts()}")

    # Standardize the data
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

    X_train_tensor = convert_data_to_tensor_for_rnn(X_train, device)
    X_val_tensor = convert_data_to_tensor_for_rnn(X_val, device)

    train_dataset = TensorDataset(X_train_tensor, torch.tensor(y_train.values, device=device))
    val_dataset = TensorDataset(X_val_tensor, torch.tensor(y_val.values, device=device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    model = LSTMModel(hidden_size=hidden_size, num_layers=num_layers, lr=lr)
    model.to(device)

    checkpoint_callback = ModelCheckpoint(monitor=f'val_{config.metric}', mode='max', save_top_k=1,
                                          dirpath=output_dir / 'checkpoints',
                                          filename='best_model-epoch-{epoch}')
    trainer = L.Trainer(
        max_epochs=config.nn_max_epochs,
        logger=True,
        default_root_dir=output_dir,  # logs directory
        callbacks=[EarlyStopping(monitor=f'val_loss', patience=3, mode='min'), checkpoint_callback],
        deterministic=True
    )
    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    best_model = LSTMModel.load_from_checkpoint(best_model_path)
    best_model.to(device)
    best_model.eval()

    with torch.no_grad():
        train_loader_for_evaluation = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        y_train_pred_probs = torch.cat(
            [best_model(x) for x, _ in train_loader_for_evaluation]).cpu().numpy().flatten()
        y_train_pred = (y_train_pred_probs > 0.5).astype(int)
        y_val_pred_probs = torch.cat([best_model(x) for x, _ in val_loader]).cpu().numpy().flatten()
        y_val_pred = (y_val_pred_probs > 0.5).astype(int)

    train_mcc = matthews_corrcoef(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auprc = average_precision_score(y_train, y_train_pred_probs)
    val_mcc = matthews_corrcoef(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auprc = average_precision_score(y_val, y_val_pred_probs)

    return train_mcc, train_f1, train_auprc, val_mcc, val_f1, val_auprc, best_model_path
