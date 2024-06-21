class ViT(nn.Layer):
    def __init__(self, in_c=3, num_classes=10, img_size=32, patch=8, dropout=0., attn_dropout=0.0, num_layers=7, hidden=384, mlp_hidden=384*4, head=8, is_cls_token=True):
        super(ViT, self).__init__()
        self.patch = patch
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // self.patch
        self.patches = Patches(self.patch_size)
        f = (img_size // self.patch) ** 2 * 3
        num_tokens = (self.patch ** 2) + 1 if self.is_cls_token else (self.patch ** 2)

        self.emb = nn.Linear(f, hidden)
        self.cls_token  = paddle.create_parameter(
            shape = [1, 1, hidden],
            dtype = 'float32',
            default_initializer=nn.initializer.Assign(paddle.randn([1, 1, hidden]))
        ) if is_cls_token else None

        self.pos_embedding  = paddle.create_parameter(
            shape = [1,num_tokens, hidden],
            dtype = 'float32',
            default_initializer=nn.initializer.Assign(paddle.randn([1,num_tokens, hidden]))
        )

        encoder_list = [TransformerEncoder(hidden, mlp_hidden=mlp_hidden, dropout=dropout, attn_dropout=attn_dropout, head=head) for _ in range(num_layers)]
        self.encoder = nn.Sequential(*encoder_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )

    def forward(self, x):
        out = self.patches(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = paddle.concat([self.cls_token.tile([out.shape[0],1,1]), out], axis=1)
        out = out + self.pos_embedding
        out = self.encoder(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out
