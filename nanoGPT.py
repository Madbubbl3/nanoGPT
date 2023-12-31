"""import libraries and data"""
# import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# load the data
path = "datasets/tinyshakespear.txt"
# path = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
with open(file=path, mode="r", encoding="utf-8") as file:
    text = file.read()

# create the encoder and decoder
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: "".join([itos[i] for i in s])
decode(encode("salut"))

# activate GPU if available
device = "cpu"
if torch.cuda.is_available():
    print("training on GPU: CUDA")
    device = "cuda"
if torch.backends.mps.is_available():
    print("training on GPU: Apple Metal")
    device = "mps"

# create the datasets
data = torch.tensor(data=encode(text), dtype=torch.int64)
n = int(len(data)) // 10 * 9
train_data = data[:n]
val_data = data[n:]

# --------------------------------------------------------------------------------------------------------------------
"""Hyperparameters"""
block_size = 64
batch_size = 128
steps = 5000  # = max_iters
# steps = 50  # = max_iters
# steps = 0  # = max_iters
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 198  # number of embedding dimension
n_head = 6
n_layers = 6
head_size = n_embd // n_head
dropout = 0.2

# --------------------------------------------------------------------------------------------------------------------
"""helper functions"""


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(low=0, high=(len(data) - block_size), size=(batch_size,))
    x = torch.stack(tensors=[data[i : (i + block_size)] for i in ix])
    y = torch.stack(tensors=[data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device=device), y.to(device=device)
    return x, y


# estimate the loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # set the model in evaluation mode
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()  # set the model in training mode
    return out


# print the number of parameters of the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------------------------------------
"""Build the Model"""


class Head(nn.Module):
    """Implementation of a single head of self attention"""

    def __init__(self, fan_in: int = n_embd, fan_out: int = head_size) -> None:
        super().__init__()
        # (fan_in,fan_out)
        self.query = nn.Linear(in_features=fan_in, out_features=fan_out, bias=False)
        # (fan_in,fan_out)
        self.key = nn.Linear(in_features=fan_in, out_features=fan_out, bias=False)
        # dim =(fan_in,fan_out)
        self.value = nn.Linear(in_features=fan_in, out_features=fan_out, bias=False)
        self.register_buffer(
            name="tril", tensor=torch.tril(torch.ones(size=(block_size, block_size)))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.query(x)  # (batch_size, n_embd, head_size)
        k = self.key(x)  # (batch_size, n_embd, head_size)
        v = self.value(x)  # (batch_size, n_embd, head_size)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # (batch_size, n_embd, n_emd)
        # wei = wei.masked_fill(mask=self.tril == 0, value=float("-inf"))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(input=wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    """Implementation of multiple heads of attention"""

    def __init__(self, n_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            modules=[Head(fan_out=head_size) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int = n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_embd, out_features=n_embd * 4),
            nn.ReLU(),
            nn.Linear(in_features=n_embd * 4, out_features=n_embd),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """one transformer block: communication followed by computation"""

    def __init__(self, n_emb: int = n_embd, n_head: int = n_head) -> None:
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_heads=n_head, head_size=head_size)
        self.ffw = FeedForward(n_embd=n_emb)
        self.ln1 = nn.LayerNorm(normalized_shape=n_emb)
        self.ln2 = nn.LayerNorm(normalized_shape=n_emb)

    def forward(self, x: torch.Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Implementation of our language model"""

    def __init__(self):
        super().__init__()
        # lookup table: each token looks directly for the next following one
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_embd
        )
        self.position_embedding_table = nn.Embedding(
            num_embeddings=block_size, embedding_dim=n_embd
        )
        # self.sa_heads = MultiHeadAttention(n_heads=4, head_size=n_embd // 4)
        self.blocks = nn.Sequential(
            *[Block(n_emb=n_embd, n_head=n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final normalisation layer
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)

    def forward(self, inputs: torch.Tensor, targets=None):
        B, T = inputs.shape
        # Tensor of size B, T and C (n_embd)
        tok_embd: torch.Tensor = self.token_embedding_table(inputs)
        # Tensor of size (T, C=n_embd)
        pos_embd: torch.Tensor = self.position_embedding_table(
            torch.arange(T, device=device)
        )
        x = tok_embd + pos_embd  # broadcasting pos_embd: (T, C) -> (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits: torch.Tensor = self.lm_head(x)  # Tensor of size B, T and C (vocab_size)

        if targets is None:
            loss = torch.zeros(size=(0,))
        else:
            # PyTorch API: inputs should be of shape (N,C) = B, C, target should be of shape (N) = B
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            yhat = targets.view(B * T)
            loss = F.cross_entropy(input=logits, target=yhat)
        # return the logits and the loss
        return logits, loss

    def generate(self, idx: torch.Tensor, maxNewTokens: int):
        # idx is the (B, T) arrays of indices in the current context
        for _ in range(maxNewTokens):
            logits, loss = self.forward(inputs=idx[:, -block_size:])
            # focus on the last one -> dim becomes (B,C)
            logits = logits[:, -1, :]
            # translate to probabilities
            probs = F.softmax(input=logits, dim=-1)
            # sample (1 sample) from the probabilities
            new_idx = probs.multinomial(num_samples=1, replacement=True)  # (B, 1)
            # complete the existing string of indices
            idx = torch.cat(tensors=(idx, new_idx), dim=1)

        return idx


# --------------------------------------------------------------------------------------------------------------------
"""create and optimize the model"""

# instance of a bigram model
model = GPTLanguageModel()
model = model.to(device=device)
# create an optimizer object
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
# print the number of parameters of the model
print(f"Number of parameters of the Model: {count_parameters(model=model) / 1e6} Mio.")
# optimize the model
loss = torch.zeros(size=())
for step in range(steps):
    # get a new batch sample
    xb, yb = get_batch()
    # forward pass
    logits, loss = model(inputs=xb, targets=yb)
    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # check loss from time to time
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"Step {step}: Loss on training set = {losses['train']}; loss on validation set = {losses['val']}"
        )


# --------------------------------------------------------------------------------------------------------------------
"""sample from the model"""


# helper function: sample from model
def sample(
    model=model,
    context=torch.zeros(size=(1, 1), dtype=torch.int64, device=device),
    maxNewToken=100,
):
    idx = context
    prediction = model.generate(idx=idx, maxNewTokens=maxNewToken).view(-1)
    return decode([i.item() for i in prediction])


# sample from the model
print(sample(maxNewToken=500))
