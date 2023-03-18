import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size,
        n_neurons,
        n_layers,
        output_size=None,
        activation="elu",
        last_layer_linear=True,
    ):

        super(MLP, self).__init__()
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.output_size = output_size
        self.activation = activation
        self.last_layer_linear = last_layer_linear
        layer_list = []
        for i in range(n_layers):
            if i == 0 and n_layers == 1:
                layer = torch.nn.Linear(
                    input_size, output_size if output_size is not None else n_neurons
                )
            elif i == 0:
                layer = torch.nn.Linear(input_size, n_neurons)
            elif i == n_layers - 1:
                layer = torch.nn.Linear(
                    n_neurons, output_size if output_size is not None else n_neurons
                )
            else:
                layer = torch.nn.Linear(n_neurons, n_neurons)
            torch.nn.init.kaiming_uniform_(layer.weight)
            layer_list.append(layer)
            if i != n_layers - 1 or not self.last_layer_linear:
                layer_list.append(self.get_activation(activation))
        self.mlp = torch.nn.Sequential(*layer_list)

    def forward(self, input):
        return torch.tanh(self.mlp(input))

    def get_activation(self, act_name):
        if act_name.lower() == "relu":
            return torch.nn.ReLU()
        elif act_name.lower() == "leakyrelu":
            return torch.nn.LeakyReLU()
        elif act_name.lower() == "elu":
            return torch.nn.ELU()
        elif act_name.lower() == "sigmoid":
            return torch.nn.Sigmoid()
        elif act_name.lower() == "tanh":
            return torch.nn.Tanh()
        elif act_name.lower() == "softplus":
            return torch.nn.Softplus()
        else:
            raise Exception("act_name {} is not valid!".format(act_name))
