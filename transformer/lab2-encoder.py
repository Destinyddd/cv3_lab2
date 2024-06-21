class TransformerEncoder(nn.Layer):
    def __init__(self, feats, mlp_hidden, head=8, dropout=0., attn_dropout=0.):
        super(TransformerEncoder, self).__init__()
        self.layer1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout, attn_dropout=attn_dropout)
        self.layer2 = nn.LayerNorm(feats)
        self.mlp = Mlp(feats, mlp_hidden)

    def forward(self, x):
        out = self.msa(self.layer1(x)) + x
        out = self.mlp(self.layer2(out)) + out
        return out
