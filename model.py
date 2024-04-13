import torch
from torch import nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_length, d_model)
        pos_e = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pos_e[:, 0::2] = torch.sin(position * div_term)
        pos_e[:, 1::2] = torch.cos(position * div_term)

        pos_e = pos_e.unsqueeze(0) # (1, seq_lenght, d_model)

        self.register_buffer('pos_e', pos_e)

    def forward(self, x):
        x = x + (self.pos_e[:, x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    
class LayerNorm(nn.Module):

    def __init__(self, eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, dff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff) ##nn.Linear take the in_feature and out_features are inputs AND bias is true by default
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, dff) -> (batch_size, seq_length, d_model)
        return self.linear2(self.dropout(self.linear1(x)))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.h = h
        self.d_model = d_model
        assert d_model % h == 0, "The dimensions must be divisble by the number of attention heads"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # batcch, h, seq_length, seq_length

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_length, d_model) -> batch, h, seq_length, d_k
        query = query.view(query.shape[0], query.shape[1]. self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, scr_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x, x, x, scr_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block, cross_attention_block, feed_forward_network, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_network = feed_forward_network
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, src_mask, encoder_output, tgt_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x:self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_network)

class Decoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, scr_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, scr_mask, encoder_output, tgt_mask)
        return self.norm(x)
    
class ProjectLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj_layer(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, scr_embdding, tgt_embdedding, scr_positional_encoding, tgt_positional_encoding, projection_layer):
        super().__init__()
        self.encder = encoder
        self.decoder = decoder
        self.scr_embed = scr_embdding
        self.tgt_embed = tgt_embdedding
        self.scr_pos = scr_positional_encoding
        self.tgt_pos = tgt_positional_encoding
        self.proj_layer = projection_layer

    def encode(self, scr, scr_mask):
        scr = self.scr_embed(scr)
        scr = self.scr_pos(scr)
        return self.encoder(scr, scr_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def projection(self, x):
        return self.proj_layer(x)
    
def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):
    src_embedding=InputEmbeddings(d_model, src_vocab_size)
    tgt_embdedding=InputEmbeddings(d_model, tgt_vocab_size)

    src_positional_encoding=PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_positional_embedding=PositionalEncoding(d_model, tgt_seq_len, dropout)

    # encoder = EncoderBlocks x N
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, encoder_feed_forward, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, decoder_feed_forward,dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = nn.ModuleList(encoder_blocks)
    decoder = nn.ModuleList(decoder_blocks)
    projection_layer = ProjectLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embedding, tgt_embdedding, src_positional_encoding, tgt_positional_embedding, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

build_transformer(512,512, 256, 256)