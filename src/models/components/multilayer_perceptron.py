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
