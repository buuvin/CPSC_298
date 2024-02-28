import re
import torch
import torch.nn as nn

class Tokenizer:
    def __init__(self):
        pass
    
    def tokenize(self, text):
        """
        Tokenize the input text into a list of tokens.
        
        Args:
            text (str): The input text to tokenize.
        
        Returns:
            list: A list of tokens extracted from the input text.
        """
        return re.findall(r'\b\w+\b', text)

    def preprocess(self, text):
        """
        Preprocess the input text before tokenization.
        
        Args:
            text (str): The input text to preprocess.
        
        Returns:
            str: The preprocessed text.
        """
        # Remove special characters and convert to lowercase
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

class MoE(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, hidden_dim=64):
        """
        Initialize the Mixture of Experts model.
        
        Args:
            num_experts (int): Number of expert networks.
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output predictions.
            hidden_dim (int): Dimension of hidden layer.
        """
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.expert_networks = nn.ModuleList([nn.Sequential(
                                                nn.Linear(input_dim, hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(hidden_dim, output_dim)
                                            ) for _ in range(num_experts)])
        self.gating_network = nn.Sequential(
                                    nn.Linear(input_dim, num_experts),
                                    nn.Softmax(dim=-1)
                                )
    
    def forward(self, x):
        """
        Perform forward pass through the Mixture of Experts model.
        
        Args:
            x (Tensor): Input data tensor.
        
        Returns:
            Tensor: Output predictions tensor.
        """
        gate_outputs = self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.expert_networks], dim=-1)
        output = torch.matmul(gate_outputs.unsqueeze(-2), expert_outputs).squeeze(-2)
        return output
    
    def train(self, x, y, optimizer):
        """
        Train the Mixture of Experts model.
        
        Args:
            x (Tensor): Input data tensor.
            y (Tensor): Target output tensor.
            optimizer (torch.optim.Optimizer): Optimizer for training.
        """
        optimizer.zero_grad()
        predictions = self.forward(x)
        loss = nn.MSELoss()(predictions, y)
        loss.backward()
        optimizer.step()
      
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the Simple Neural Network model.
        
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int or list): Dimension(s) of hidden layers.
            output_dim (int): Dimension of output predictions.
        """
        super(SimpleNN, self).__init__()
        
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        
        layers = []
        for i in range(len(hidden_dim)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim[i]))
            else:
                layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Perform forward pass through the Simple Neural Network model.
        
        Args:
            x (Tensor): Input data tensor.
        
        Returns:
            Tensor: Output predictions tensor.
        """
        return self.model(x)
