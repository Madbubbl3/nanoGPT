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

# create the datasets
data = torch.tensor(data=encode(text), dtype=torch.int64)
n = int(len(data))
train_data = data[:n]
val_data = data[n:]

# hyperparameters
block_size = 8
batch_size = 32
steps = 10000


# helper functions
def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(low=0, high=(len(data) - block_size), size=(batch_size,))
    x = torch.stack(tensors=[data[i : (i + block_size)] for i in ix])
    y = torch.stack(tensors=[data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# Create a class for the language model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # lookup table: each token looks directly for the next following one
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )

    def forward(self, inputs: torch.Tensor, targets=None):
        logits: torch.Tensor = self.token_embedding_table(
            inputs
        )  # Tensor of size B, T and C
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
model = BigramLanguageModel(vocab_size=vocab_size)


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
for _ in range(steps):
    # get a new batch sample
    xb, yb = get_batch()
    # forward pass
    logits, loss = model(inputs=xb, targets=yb)
    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Loss = {loss.item()}")
print(sample(maxNewToken=300))
