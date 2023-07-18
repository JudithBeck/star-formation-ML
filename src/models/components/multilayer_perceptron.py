from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int = 4302,
        lin1_size: int = 2048,
        lin2_size: int = 1024,
        lin3_size: int = 512,
        lin4_size: int = 256,
        lin5_size: int = 64,
        lin6_size: int = 32,
        output_size: int = 5,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            # nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            # nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            # nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin4_size),
            # nn.BatchNorm1d(lin4_size),
            nn.ReLU(),
            nn.Linear(lin4_size, lin5_size),
            # nn.BatchNorm1d(lin5_size),
            nn.ReLU(),
            nn.Linear(lin5_size, lin6_size),
            # nn.BatchNorm1d(lin6_size),
            nn.ReLU(),
            nn.Linear(lin6_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    _ = MLP()


#In the folder 'components' inside 'models' within the 'src' directory, you have a Python class named MLP. This class defines a Multi-Layer Perceptron (MLP) neural network using PyTorch's nn.Module as the base class.

#The MLP class constructor initializes the MLP model with default layer sizes. The model consists of six fully connected (linear) layers followed by ReLU activation functions. The commented-out lines with nn.BatchNorm1d indicate that batch normalization layers are not used in this model.

#The forward method defines how data flows through the model. It applies the sequence of layers defined in the constructor to the input 'x', returning the output of the model.

#The provided script (__name__ == "__main__") checks if this module is run as the main program and instantiates the MLP class. However, it doesn't perform any specific operations in the current context.
