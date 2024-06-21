class MultiHeadSelfAttention(nn.Layer):
    def __init__(self, feats, head=8, dropout=0., attn_dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats ** 0.5
        self.qkv = nn.Linear(feats,
                             feats * 3)
        self.out = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + [self.head, self.feats//self.head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        b, n, f = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multi_head, qkv)
        attn = F.softmax(paddle.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d, axis=-1)
        attn = self.attn_dropout(attn)
        attn = paddle.einsum("bhij, bhjf->bihf", attn, v)
        out = self.dropout(self.out(attn.flatten(2)))
        return out
