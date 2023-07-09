# import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# load the data
with open(file="datasets/tinyshakespear.txt", mode="r", encoding="utf-8") as file:
    text = file.read()

# create the encoder and decoder
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: "".join([itos[i] for i in s])
decode(encode("salut"))
n_embd = 32  # number of embedding dimension

# create the datasets
data = torch.tensor(data=encode(text), dtype=torch.int64)
n = int(len(data)) // 10 * 9
train_data = data[:n]
val_data = data[n:]

# hyperparameters
block_size = 8
batch_size = 32
steps = 1000
# steps = 10000
eval_iters = 200
eval_interval = 300


# helper functions
def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(low=0, high=(len(data) - block_size), size=(batch_size,))
    x = torch.stack(tensors=[data[i : (i + block_size)] for i in ix])
    y = torch.stack(tensors=[data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


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


# Create a class for the language model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # lookup table: each token looks directly for the next following one
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_embd
        )
        self.position_embedding_table = nn.Embedding(
            num_embeddings=block_size, embedding_dim=n_embd
        )
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)

    def forward(self, inputs: torch.Tensor, targets=None):
        B, T = inputs.shape
        tok_embd: torch.Tensor = self.token_embedding_table(
            inputs
        )  # Tensor of size B, T and C (n_embd)
        pos_embd: torch.Tensor = self.position_embedding_table(
            torch.arange(T)
        )  # Tensor of size (T, C=n_embd)
        x = tok_embd + pos_embd  # broadcasting pos_embd: (T, C) -> (B, T, C)
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
            # gets the prediction
            logits, loss = self.forward(inputs=idx)
            # focus on the last one -> dim becomes (B,C)
            logits = logits[:, -1, :]
            # translate to probabilities
            probs = F.softmax(input=logits, dim=-1)
            # sample (1 sample) from the probabilities
            new_idx = probs.multinomial(num_samples=1, replacement=True)  # (B, 1)
            # complete the existing string of indices
            idx = torch.cat(tensors=(idx, new_idx), dim=1)  # (B, T+1)
        return idx


# instance of a bigram model
model = BigramLanguageModel()


# helper function: sample from model
def sample(
    model=model, context=torch.zeros(size=(1, 1), dtype=torch.int64), maxNewToken=100
):
    idx = torch.zeros(size=(1, 1), dtype=torch.int64)
    prediction = model.generate(idx=idx, maxNewTokens=maxNewToken).view(-1)
    return decode([i.item() for i in prediction])


# optimization
# create an optimizer object
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
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
            f"Loss on training set = {losses['train']}; loss on validation set = {losses['val']}"
        )

print(sample(maxNewToken=300))
