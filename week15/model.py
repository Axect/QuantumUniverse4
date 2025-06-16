from torch import nn


def create_net(sizes):
    net = []
    for i in range(len(sizes) - 1):
        net.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            # Batch Normalization
            net.append(nn.BatchNorm1d(sizes[i + 1]))
            net.append(nn.GELU())
    return nn.Sequential(*net)


@torch.compile
class DeepONet(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super().__init__()

        nodes = hparams["nodes"]
        layers = hparams["layers"]
        branches = hparams["branches"]

        input_size = 100
        query_size = 1

        self.branch_net = create_net(
            [input_size] + [nodes] * layers + [branches]
        )
        self.trunk_net = create_net(
            [query_size] + [nodes] * layers + [branches]
        )

    def forward(self, dNdE, E):
        B, _ = dNdE.shape
        window = E.shape[1]
        branch_out = self.branch_net(dNdE) # B x p
        E_reshaped = E.reshape(-1, 1) # (B x w) x 1
        trunk_out_all = self.trunk_net(E_reshaped) # (B x w) x p
        trunk_out = trunk_out_all.reshape(B, window, -1) # B x w x p
        pred = torch.einsum("bp,bwp->bw", branch_out, trunk_out)
        return pred


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        - x: (B, W, d_model)
        - self.pe: (1, M, d_model)
        - self.pe[:, :x.size(1), :]: (1, W, d_model)
        - output: (B, W, d_model)
        """
        x = x + self.pe[:, : x.size(1), :] # pyright: ignore
        return x


class TFEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # self.pos_encoder = LearnablePositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers, norm=nn.LayerNorm(d_model)
        )

    def forward(self, x):
        """
        - x: (B, W1, 1)
        - x (after embedding): (B, W1, d_model)
        - out: (B, W1, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        return out


class TFDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers, norm=nn.LayerNorm(d_model)
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, memory):
        """
        - x: (B, W2, 1)
        - x (after embedding): (B, W2, d_model)
        - memory: (B, W1, d_model)
        - out: (B, W2, d_model)
        - out (after fc): (B, W2, 1)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        out = self.transformer_decoder(x, memory)
        out = self.fc(out)
        return out.squeeze(-1)


@torch.compile
class TraONet(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super().__init__()

        d_model = hparams["d_model"]
        nhead = hparams["n_head"]
        num_layers = hparams["num_layers"]
        dim_feedforward = hparams["d_ff"]
        dropout = hparams["dropout"]

        self.branch_net = TFEncoder(
            d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.trunk_net = TFDecoder(d_model, nhead, num_layers, dim_feedforward, dropout)

    def forward(self, u, y):
        """
        - u: (B, W1)
        - y: (B, W2)
        - u (after reshape): (B, W1, 1)
        - y (after reshape): (B, W2, 1)
        - memory: (B, W1, d_model)
        - o: (B, W2)
        """
        B, W1 = u.shape
        _, W2 = y.shape
        u = u.view(B, W1, 1)
        y = y.view(B, W2, 1)

        # Encoding
        memory = self.branch_net(u)

        # Decoding
        o = self.trunk_net(y, memory)
        return o
