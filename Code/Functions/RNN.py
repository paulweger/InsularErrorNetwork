import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=10, output_dim=1, prior=[0.66, 0.34],
                 num_layers=2, dropout_rate=0.2, lr=0.001, cell_type='lstm', use_batchnorm=False, init_weights=None, fixed_skip_weight=None):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.initial_lr = lr
        self.cell_type = cell_type.lower()
        self.fixed_skip_weight = fixed_skip_weight

        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        else:
            raise ValueError("Invalid cell type. Choose 'lstm' or 'gru'.")
        
        # fc at end, skip transform with skip weight
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.skip_transform = nn.Linear(input_dim, output_dim)
        if fixed_skip_weight is None: #learned weight.
            self.skip_weight = nn.Parameter(torch.tensor(0.5))  # Learnable weight
        else: #fixed weight.
            self.skip_weight = torch.tensor(fixed_skip_weight)

        if init_weights is not None:
            self.load_state_dict(init_weights)
        else: self.apply(self._init_weights)

        # Loss, optimizer & scheduler
        class_prior = torch.tensor(prior)
        pos_weight = torch.tensor([class_prior[1] / class_prior[0]])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer =  optim.Adam(self.parameters(), lr=lr, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.75)
        
        self.hidden_state = None
        
    def _init_weights(self, m):
        """ Apply Kaiming initialization to linear layers. """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    
    
    def forward(self, trial):
        """Forward pass for a single trial."""
        trial = trial.clone().detach().to(torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions

        if self.hidden_state is None:
            if self.cell_type == 'lstm':
                self.hidden_state = (torch.zeros(self.num_layers, 1, self.hidden_dim),
                                     torch.zeros(self.num_layers, 1, self.hidden_dim))
            else:
                self.hidden_state = torch.zeros(self.num_layers, 1, self.hidden_dim)

        output, self.hidden_state = self.rnn(trial, self.hidden_state)
        output = self.fc(output)  

        # Weighted skip connections 
        transformed_input = self.skip_transform(trial[:, :, :])
        output = (1 - self.skip_weight) * output + self.skip_weight * transformed_input

        return output[:, -1, :]
    

    def fit(self, X, y, epochs=60, batch_size=32):
        """Trains the RNN model."""
        self.train()  # Set model to training mode
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape y for loss calculation

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False) #shuffle false to keep temporal order

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                self.hidden_state = None #reset hidden state at the beginning of each batch.

                outputs = []
                for trial in batch_X:
                    output = self.forward(trial)
                    outputs.append(output)

                outputs = torch.stack(outputs).squeeze(2)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            #print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")
            self.scheduler.step()
            

        
    def evaluate(self, X_test):
        """Evaluates the RNN model on a test set and returns probabilities of movement."""
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            X_test = torch.tensor(X_test, dtype=torch.float32)

            probabilities = []
            self.reset_hidden_state()  # Reset hidden state at the beginning
            for trial in X_test:
                logits = self.forward(trial)  # Use forward to get logits
                prob = torch.sigmoid(logits)  # Get probability of class 1
                probabilities.append(prob.item())  # Append the probability (as a scalar)

            return np.array(probabilities)


    def predict_proba(self, trial):
        """Predicts the probability of a single trial and updates hidden state."""
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            trial = torch.tensor(np.squeeze(trial), dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            
            if self.hidden_state is None:
                if self.cell_type == 'lstm':
                    self.hidden_state = (torch.zeros(self.num_layers, 1, self.hidden_dim),
                                         torch.zeros(self.num_layers, 1, self.hidden_dim))
                else:
                    self.hidden_state = torch.zeros(self.num_layers, 1, self.hidden_dim)

            output, self.hidden_state = self.rnn(trial, self.hidden_state)
            output = self.fc(output[:, -1, :])  # Take the last time step's output
            probs = torch.sigmoid(output).squeeze().item()  # Apply sigmoid and squeeze
            rest_prob = 1 - probs

        return np.array([[rest_prob], [probs]]).T
    

    def predict(self, trial):
        """Predicts the class (0 or 1) of a single trial using predict_proba."""
        probability = self.predict_proba(trial)  # Use predict_proba
        prediction = 1 if probability >= 0.5 else 0  # Classify based on probability
        return prediction
    
    
    def reset_hidden_state(self):
        """Resets the hidden state of the RNN."""
        self.hidden_state = None