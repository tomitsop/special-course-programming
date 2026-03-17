#%%
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import seaborn as sns
import matplotlib.pyplot as plt

Number_of_days = 30*6
Number_of_hours = 24*Number_of_days
PTC_flows = False
NP = True
train_val_share = 0.5 # how much data should be used for training and validation

ML_results_path = 'ML_results/' + str(Number_of_hours) + '_hours'
if not os.path.exists(ML_results_path):
    os.makedirs(ML_results_path)

# Load the data
X = pd.read_csv('data\X.csv', index_col=0)  

if NP:
    Y = pd.read_parquet(f'data\Y_NP_FBMC.parquet')

# Set random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # If using CUDA

# For deterministic behavior (optional, slightly reduces performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add 1 to the X index
X.index = X.index + 1

# Take the index from Y and extract the corresponding rows from X
X = X.loc[Y.index]  # Ensure all indices in Y exist in X to avoid KeyErrors

# Create the training/validation and test splits
split = int(train_val_share * len(Y))  
X_train_val = X.iloc[:split]
Y_train_val = Y.iloc[:split]
X_test = X.iloc[split:]
Y_test = Y.iloc[split:]

# Further split the training/validation set into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_val, Y_train_val, test_size=0.2, random_state=random_seed, shuffle=True
)

# Display the sizes of the resulting datasets for validation
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, Y_train: {Y_train.shape}, Y_val: {Y_val.shape}")
print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")

# Identify columns with 'load' in the name
load_columns = [col for col in X_train.columns if 'load' in col]

# Initialize scalers
scalerMinMax = MinMaxScaler()
scalerNorm = StandardScaler()  # Note: `scalerNorm` is defined but not used in the current script

# Fit and transform only the 'load' columns for train, validation, and test sets
X_train_scaled = X_train.copy()
X_train_scaled[load_columns] = scalerNorm.fit_transform(X_train[load_columns])

X_val_scaled = X_val.copy()
X_val_scaled[load_columns] = scalerNorm.transform(X_val[load_columns])

X_test_scaled = X_test.copy()
X_test_scaled[load_columns] = scalerNorm.transform(X_test[load_columns])

# Scale Y (MinMaxScaler is used for Y)
Y_train_scaled = scalerMinMax.fit_transform(Y_train)  # Fit and transform Y_train
Y_val_scaled = scalerMinMax.transform(Y_val)          # Transform Y_val
Y_test_scaled = scalerMinMax.transform(Y_test)        # Transform Y_test

# Convert scaled Y back to DataFrame (optional for consistency)
Y_train_scaled = pd.DataFrame(Y_train_scaled, columns=Y_train.columns, index=Y_train.index)
Y_val_scaled = pd.DataFrame(Y_val_scaled, columns=Y_val.columns, index=Y_val.index)
Y_test_scaled = pd.DataFrame(Y_test_scaled, columns=Y_test.columns, index=Y_test.index)


# Neural Network Implementation

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_scaled.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val_scaled.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_scaled.values, dtype=torch.float32)

# Create DataLoader for training and validation
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train_tensor.shape[1]
output_size = Y_train_tensor.shape[1]
model = NeuralNet(input_size, output_size)

criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10)# 

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)  # Adjust learning rate based on validation loss

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Evaluate on test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    predictions = []
    actuals = []
    for X_batch, Y_batch in test_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        test_loss += loss.item()
        predictions.append(outputs.numpy())
        actuals.append(Y_batch.numpy())
test_loss /= len(test_loader)

# Flatten the predictions and actuals for analysis
predictions = torch.cat([torch.tensor(p) for p in predictions]).numpy()
actuals = torch.cat([torch.tensor(a) for a in actuals]).numpy()

# Convert predictions and actuals back to their original scale using inverse_transform
predictions_original = scalerMinMax.inverse_transform(predictions)
actuals_original = scalerMinMax.inverse_transform(actuals)

print(f"Test Loss: {test_loss:.4f}")

# Add timestep and line identifiers to the comparison DataFrame
comparison_df = pd.DataFrame({
    "Timestep": np.repeat(Y_test.index.values, Y_test.shape[1]),  # Repeat each timestep for all lines
    "Line": np.tile(Y_test.columns.values, len(Y_test)),         # Tile the lines for all timesteps
    "Actual": actuals_original.flatten(),
    "Predicted": predictions_original.flatten()
})

# Display the first few rows of the enhanced DataFrame
print(comparison_df.head())

#make a prediction data frame with time step as index and line as columns
prediction_df = pd.DataFrame(predictions_original, index=Y_test.index, columns=Y_test.columns)

#pritn the first few rows
print(prediction_df.head())

if PTC_flows:
    comparison_df.to_csv(f'{ML_results_path}\\comparison_PTC.csv')
    prediction_df.to_csv(f'{ML_results_path}\\predictions_PTC.csv')
if NP:
    comparison_df.to_csv(f'{ML_results_path}\\comparison_NP.csv')
    prediction_df.to_csv(f'{ML_results_path}\\predictions_NP.csv')



#calculate the mean absolute error
comparison_df['MAE'] = abs(comparison_df['Actual'] - comparison_df['Predicted'])
comparison_df['MSE'] = comparison_df['MAE']**2
print('The mean absolute error is: ', np.round(comparison_df['MAE'].mean(), 2))
print('The mean squared error is: ', np.round(comparison_df['MSE'].mean(), 2))

#plot the distribution of the mean absolute error

sns.histplot(comparison_df['MAE'], bins=50, kde=True)
plt.xlabel('Mean Absolute Error')
plt.ylabel('Frequency')
plt.title('Mean Absolute Error Distribution')
plt.show()


# Define error metrics
def normalized_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)) / (np.mean(np.abs(y_true)) + 1e-8)

def relative_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)) / (np.max(y_true) - np.min(y_true) + 1e-8)

# Compute metrics per line
mae_metrics_per_line = (
    comparison_df.groupby("Line", group_keys=False)
    .apply(lambda df: pd.Series({
        "Normalized MAE": normalized_mae(df["Actual"], df["Predicted"]),
        "Relative MAE": relative_mae(df["Actual"], df["Predicted"])
    }))
    .reset_index()
)

# Optional: sort by one of the metrics
mae_metrics_per_line = mae_metrics_per_line.sort_values("Normalized MAE", ascending=False)
print(mae_metrics_per_line)

print(f"The mean Normalized MAE is: {mae_metrics_per_line['Normalized MAE'].mean():.4f}")
print(f"The mean Relative MAE is: {mae_metrics_per_line['Relative MAE'].mean():.4f}")


# %%