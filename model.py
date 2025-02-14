import torch
import torch.nn as nn


class PINNs(nn.Module):
    def __init__(self, hyper_param):
        super().__init__()
        if hyper_param['is_res']:
            self.nn = ResNet(hyper_param)
        else:
            self.nn = MLP(hyper_param)
    
    def forward(self, x):
        return self.nn(x)


class MLP(nn.Module):
    def __init__(self, hyper_param):
        super().__init__()
        self.init_layer = nn.ModuleList([nn.Linear(5, hyper_param["nb_neurons"])])
        self.hiden_layers = nn.ModuleList(
            [
                nn.Linear(hyper_param["nb_neurons"], hyper_param["nb_neurons"])
                for _ in range(hyper_param["nb_layers"] - 1)
            ]
        )
        self.final_layer = nn.ModuleList([nn.Linear(hyper_param["nb_neurons"], 3)])
        self.layers = self.init_layer + self.hiden_layers + self.final_layer
        self.initial_param()

    def forward(self, x):
        for k, layer in enumerate(self.layers):
            if k != len(self.layers) - 1:
                x = torch.tanh(layer(x))
            else:
                x = layer(x)
        return x  # Retourner la sortie

    def initial_param(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class ResBlock(nn.Module):
    def __init__(self, hyper_param):
        super().__init__()
        self.nb_layer_block = hyper_param["nb_layer_block"]
        self.nb_neurons = hyper_param["nb_neurons"]
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.nb_neurons, self.nb_neurons)
                for _ in range(self.nb_layer_block - 1)
            ]
        )
        self.initial_param()

    def forward(self, x):
        x_ = x
        for k, layer in enumerate(self.layers):
            x_ = torch.tanh(layer(x_))
        return x + x_  # Retourner la sortie

    def initial_param(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class ResNet(nn.Module):
    def __init__(self, hyper_param):
        super().__init__()
        self.init_layer = nn.ModuleList([nn.Linear(5, hyper_param["nb_neurons"])])
        self.hiden_layers = nn.ModuleList(
            ResBlock(hyper_param) for k in range(hyper_param['nb_blocks'])
        )
        self.final_layer = nn.ModuleList([nn.Linear(hyper_param["nb_neurons"], 3)])
        self.initial_param()
        self.layers = self.init_layer + self.hiden_layers + self.final_layer

    def forward(self, x):
        for k, layer in enumerate(self.layers):
            x = layer(x)
        return x  # Retourner la sortie

    def initial_param(self):
        nn.init.xavier_uniform_(self.init_layer[0].weight)
        nn.init.zeros_(self.init_layer[0].bias)
        nn.init.xavier_uniform_(self.final_layer[0].weight)
        nn.init.zeros_(self.final_layer[0].bias)


if __name__ == "__main__":
    hyper_param = {"nb_layers": 12, "nb_neurons": 64}
    piche = PINNs(hyper_param)
    nombre_parametres = sum(p.numel() for p in piche.parameters() if p.requires_grad)
    print(nombre_parametres)
