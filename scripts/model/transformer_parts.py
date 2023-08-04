import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0, max_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        if self.use_mask:
            print("Using forward mask for transformer decoder.")

    def forward(self, src):
        # Apply positional encoding to the source tensor
        src = self.pos_encoder(src)

        # Create a target tensor with the same shape as the source tensor
        tgt = src.clone()

        # Apply positional encoding to the target tensor
        tgt = self.pos_encoder(tgt)

        # Generate a square subsequent mask to prevent the decoder from attending to future positions
        if self.use_mask:
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        else:
            tgt_mask = None

        # Pass the source and target tensors through the transformer model
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return output

if __name__ == "__main__":
    # Define the transformer model
    model = Seq2SeqTransformer(
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        
    )

    # Example input sequence (sequence_length=10, batch_size=1, d_model=512)
    src = torch.rand(10, 1, 512)

    # Pass the input through the transformer model
    output = model(src)
    print(output.shape)