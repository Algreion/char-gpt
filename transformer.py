import torch
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from time import time

SEED = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Hyperparameters
BLOCK = 64
LEARNING = 3e-4
EMBEDIMS = 120
HEAD_SIZE = EMBEDIMS
HEAD_NUMBER = 6
SINGLE_HEAD_SIZE = HEAD_SIZE // HEAD_NUMBER # Usually multiple heads are smaller
FFWD_HIDDEN = 4
DROPOUT = 0.05
N_LAYERS = 6
assert EMBEDIMS % HEAD_NUMBER == 0, "Embedding dimensions must be divisible by head number."

# Default stats
BATCH = 64
MAXTOKENS = 300
TRAINING_STEPS = 5000
EPSILON = 1e-5

# Default generation
GEN_SEED = '\n\n' # Following Shakespeare's format

class Data:
    """Character-level transformer | Shared data infrastructure."""
    def __init__(self, file: str = '', context: int = BLOCK):
        self.data = ''
        if file:
            try:
                with open(file,'r',encoding='utf-8') as f:
                    self.data = f.read()
            except OSError as e:
                print(f"Couldn't read '{file}' ({e}). Skipping data.")
        self.lookup = dict(enumerate(sorted(set(self.data)))) # index -> char
        self.rlookup = dict((v,k) for k,v in self.lookup.items()) # char -> index
        self.vocab_size = len(self.lookup)
        self.encode = lambda s: [self.rlookup[c] for c in s]
        self.decode = lambda x: "".join(self.lookup[i] for i in x)
        self.context = context
        self.data = torch.tensor(self.encode(self.data), dtype = torch.long)
        self.splitdata()
    def __repr__(self):
        return f"Data(vocabSize={self.vocab_size},trainingData={len(self.traindata)},devData={len(self.devdata)})"
    
    def splitdata(self, n: float = 0.9) -> None:
        """Splits the data into training and development data."""
        n = int(len(self.data)*n)
        self.traindata = self.data[:n]
        self.devdata = self.data[n:]
    
    def trainingset(self, batch_size: int = 4, split: str = 'train') -> tuple:
        """Generates a batch of training/validation data. (split = train,dev,none)"""
        data = self.traindata if split=='train' else self.devdata if split=='dev' else self.data
        index = torch.randint(len(data)-self.context, (batch_size,))
        inputs = torch.stack([data[i:i+self.context] for i in index])
        labels = torch.stack([data[i+1:i+self.context+1] for i in index])
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        return inputs,labels

class Head(nn.Module):
    """Single head of self-attention"""
    def __init__(self, head_size: int = SINGLE_HEAD_SIZE, context: int = BLOCK, n_embed: int = EMBEDIMS):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False) # Every char has info about itself
        self.query = nn.Linear(n_embed, head_size, bias=False) # Every char wants to know specific things
        self.value = nn.Linear(n_embed, head_size, bias=False) # True embedding that is refined by key-queries
        self.dropout_p = DROPOUT
        # self.register_buffer('tril',torch.tril(torch.ones(context, context)))
        # self.dropout = nn.Dropout(DROPOUT)

    def forward(self, inp):
        # B,T,C = inp.shape if self.training else (1, *inp.shape) # Batch size | Tokens/Context | Character vectors
        k = self.key(inp) # BxTxH | H = HeadSize
        q = self.query(inp) # BxTxH
        v = self.value(inp)
        # Attention scores (affinities) as dot products
            # weights = q @ k.transpose(-2,-1) * C**-0.5 # BxTxH @ BxHxT = BxTxT | Layer by layer, TxH @ HxT -> TxT | Dot product between all k-q, how relevant each char is to another
            # if self.training: weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # Cannot communicate with future tokens
            # weights = F.softmax(weights, dim = -1)
            # weights = self.dropout(weights)
            # Weighted aggregation of values (Values are true embedded tokens)
            # return weights @ v # BxTxT @ BxTxH = BxTxH
        # Same as ^, but more efficient:
        return F.scaled_dot_product_attention(q, k, v, dropout_p = self.dropout_p if self.training else 0.0, is_causal = True)

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention acting in parallel."""
    def __init__(self, n_heads: int = HEAD_NUMBER, head_size: int = SINGLE_HEAD_SIZE, context: int = BLOCK, n_embed: int = EMBEDIMS):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, context, n_embed) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed) # Projection (residual connections)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, inp):
        out = torch.cat([h(inp) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Basic non-linear layer"""
    def __init__(self, n_embed: int = EMBEDIMS):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_embed, n_embed * FFWD_HIDDEN),
            nn.GELU(),
            nn.Linear(n_embed * FFWD_HIDDEN, n_embed),
            nn.Dropout(DROPOUT)
        )
    def forward(self, inp):
        return self.layer(inp)

class Block(nn.Module):
    """Transformer block | Communication -> Computation"""
    def __init__(self, head_n: int = HEAD_NUMBER, head_size: int = SINGLE_HEAD_SIZE, context: int = BLOCK, n_embed = EMBEDIMS):
        super().__init__()
        self.sa = MultiHeadAttention(head_n, head_size, context, n_embed)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed,eps=EPSILON)
        self.ln2 = nn.LayerNorm(n_embed,eps=EPSILON)
    
    def forward(self, inp):
        inp = inp + self.sa(self.ln1(inp))
        return inp + self.ffwd(self.ln2(inp))

class GPT(Data, nn.Module):
    def __init__(self, file: str = '', model: str = '', context: int = BLOCK):
        """GPT Transformer architecture. To load model, you can use the vocab file for file."""
        Data.__init__(self, file, context)
        nn.Module.__init__(self)
        self.token_embedding = nn.Embedding(self.vocab_size, EMBEDIMS) # Embedding lookup for each character
        self.position_embedding = nn.Embedding(self.context, EMBEDIMS) # Embedding lookup for positions (from 0 to context)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.lnf = nn.LayerNorm(EMBEDIMS,eps=EPSILON) # Final layer norm
        self.lm_head = nn.Linear(EMBEDIMS, self.vocab_size) # ^ Correct possibly skewed dimensions
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING) # Gradient descent but better
        if model: self.load(model)

    def __repr__(self):
        return f'GPT(vocabSize={self.vocab_size},context={self.context},size={self.size()})'
    
    def sample(self, word: str | int = '', maxlen: int | str = MAXTOKENS, temperature: float = 1.0, top_k: int | None = None, number: int = 0):
        n = maxlen
        if type(word)==int: word, maxlen = '', word
        if type(n)==str: word = n
        return self.generateN(number, word, maxlen, temperature=temperature, top_k=top_k) if number > 0 else self.generate(word, maxlen, temperature=temperature, top_k=top_k)

    def inference(self, inputs):
        T = inputs.numel()
        token_embeddings = self.token_embedding(inputs) #  token into N-dimensional space.
        pos_embeddings = self.position_embedding(torch.arange(T, device=inputs.device)) # Embed also its position.
        activation = token_embeddings + pos_embeddings # Sum ^ and ^^ to get all info about individual token
        refined = self.blocks(activation) # Refine it with self-attention (context of other tokens)
        normalized = self.lnf(refined)
        return self.lm_head(normalized)
    
    def forward(self, inputs, labels) -> tuple:
        """Returns the logits and loss of the model."""
        B, T = inputs.shape
        token_embeddings = self.token_embedding(inputs) # [Batch,Context,VocabSize] | Prediction of char given previous
        pos_embeddings = self.position_embedding(torch.arange(T, device=inputs.device))
        activation = token_embeddings + pos_embeddings
        refined = self.blocks(activation)
        normalized = self.lnf(refined)
        logits = self.lm_head(normalized)
        B, T, C = logits.shape
        logits = logits.view(B*T,C) # Every row is different example's logits.
        labels = labels.view(-1) # Flatten labels
        loss = F.cross_entropy(logits, labels)
        return logits, loss
    
    def fit(self, n: int = TRAINING_STEPS, batchsize: int = BATCH, eval_gaps: int | None = TRAINING_STEPS//10, plot: bool = False) -> list[tuple]:
        """Trains model for n steps. Returns [(step, train_loss, dev_loss), ...]"""
        if len(self.traindata) <= self.context+1:
            print("No data to train with!")
            return
        self.train()
        history, start = [], time()
        for i in range(n):
            X, Y = self.trainingset(batchsize)
            self.optimizer.zero_grad()
            _, loss = self(X,Y)
            loss.backward()
            self.optimizer.step()
            if eval_gaps is not None and (i%eval_gaps == 0 or i==n-1):
                est = self.estimate_loss(50)
                history.append((i, est['train'], est['dev']))
                print(f"step {i+1 if i in [0,n-1] else i:>{len(str(n))}}: train {est['train']:.3f} | dev {est['dev']:.3f} | {round(time()-start)}s", end='\r', flush=True)
        if plot:
            steps, train, dev = zip(*history)
            plt.figure()
            plt.plot(steps, train, label='train'); plt.plot(steps, dev, label='dev')
            plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
        return history

    @torch.no_grad()
    def estimate_loss(self, batches: int = 100, batchsize: int = BATCH):
        """Mean loss over random batches. Checks train and dev."""
        prev_training = self.training
        self.eval()
        result = {}
        for split in ('train','dev'):
            losses = torch.zeros(batches)
            for k in range(batches):
                X,Y = self.trainingset(batchsize, split)
                losses[k] = self(X,Y)[1].item()
            result[split] = losses.mean().item()
        self.train(prev_training)
        return result
    
    @torch.no_grad()
    def generate(self, word: str = '', maxlen: int = MAXTOKENS, temperature=1.0, top_k=None) -> str:
        """Generate one word based on training data."""
        if not word:
            context = torch.tensor(self.encode(GEN_SEED), device=DEVICE)
        else: context = torch.tensor(self.encode(word),device=DEVICE)
        if temperature==0: top_k=1 # Greedy decoding
        if top_k is not None and top_k<1: top_k=None
        self.eval()
        for _ in range(maxlen):
            logits = self.inference(context[-self.context:])[-1] # Feed last character
            logits = logits / temperature # <1 predictable, >1 creative
            if top_k is not None:
                k = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < k, float('-inf'))
            probs = F.softmax(logits, dim=-1)
            index = torch.multinomial(probs, 1)
            context = torch.cat((context,index), 0) # Add one to context (till maxlen)
        context = context.tolist()
        if not word: context = context[2:]
        return self.decode(context)
    
    def generateN(self, n: int = 10, word: str = '', maxlen: int = MAXTOKENS, temperature=1.0, top_k=None) -> list:
        """Generate a list of N words based on training data."""
        res = []
        for _ in range(n):
            res.append(self.generate(word, maxlen,temperature=temperature,top_k=top_k))
        return res

    @torch.no_grad()
    def save(self, file: str, vocabfile: str = ''):
        """Save the model's parameters onto a file. It is recommended to also save its vocab."""
        with open(file, 'w', encoding='utf-8') as f:
            validation = f"[{self.context},{EMBEDIMS},{HEAD_SIZE},{HEAD_NUMBER},{SINGLE_HEAD_SIZE},{FFWD_HIDDEN},{N_LAYERS},{self.vocab_size}]\n"
            f.write(validation)
            for m in self.parameters():
                f.write("\n".join(map(lambda x: str(x), m.view(-1).tolist())))
                f.write("\n")
        if vocabfile:
            with open(vocabfile,'w',encoding='utf-8') as f:
                f.write("".join(c for c in self.rlookup))
    
    def _updatedata(self, file: str, clear: bool = True):
        """By default, erases current data. Make sure the file has no unknown characters!"""
        with open(file, 'r', encoding='utf-8') as f:
            if clear:
                self.data = torch.tensor(self.encode(f.read()), dtype = torch.long)
            else:
                self.data = torch.cat((self.data, torch.tensor(self.encode(f.read()), dtype = torch.long)))
        self.splitdata()

    @torch.no_grad()
    def load(self, file: str):
        """Load the model's parameters."""
        with open(file, 'r', encoding='utf-8') as f:
            validation = f"[{self.context},{EMBEDIMS},{HEAD_SIZE},{HEAD_NUMBER},{SINGLE_HEAD_SIZE},{FFWD_HIDDEN},{N_LAYERS},{self.vocab_size}]"
            first = f.readline().strip()
            if validation != first:
                try:
                    b,e,h1,h2,shs,ff,nl,vs = map(int,first.removeprefix('[').removesuffix(']').split(','))
                except ValueError:
                    raise ValueError("Given file doesn't contain matching validation!")
                raise ValueError(f"""Given model doesn't match! Required hyper-parameters:
                context={b}, embedDims={e}, headSize={h1},headNum={h2},singleHeadSize={shs}
                ffwdHidden={ff},nLayers={nl},vocabSize={vs}""")
            for m in self.parameters():
                data = [float(f.readline()) for _ in range(m.numel())]
                m.data.copy_(torch.tensor(data, dtype=m.dtype).view(m.shape))
                m.grad = None
    
    @torch.no_grad()
    def plot(self, vectors: torch.Tensor | None = None, basicflatten: bool = False) -> None:
        """Shows the way the model maps the characters as vectors."""
        check =  dict([(i,str(i)) for i in range(1000)])
        if vectors is None:
            C = self.token_embedding.weight
            check = self.lookup
        else: C = vectors
        if len(C[0]) > 2:
            if basicflatten:
                C = C[:,:2]
            else: C = self.flatten(C)
        C = C.to('cpu')
        plt.figure(figsize=(8,8))
        plt.scatter(C[:,0].data, C[:,1].data, s = 200)
        for i in range(C.shape[0]):
            plt.text(C[i,0].item(), C[i,1].item(), check[i] ,ha="center",va="center",color="white")
        plt.grid(True)
    
    def flatten(self, vectors: torch.Tensor) -> torch.Tensor:
        """Reduces N-dimensional vectors to 2D (with PCA)"""
        U, S, _ = torch.linalg.svd(vectors-vectors.mean(dim=0,keepdim=True))
        return U[:, :2] @ torch.diag(S[:2])
    
    def size(self) -> int:
        """Returns total parameter size of model."""
        n = 0
        for p in self.parameters():
            n += p.numel()
        return n

def run_training(data:str='./models/shakespeare.txt', out:str='./models/transformer-shspr.txt', vocab:str='./models/vocab.txt',
    training_steps = 20000) -> GPT:
    """Retrains the demo model on Shakespeare's dataset."""
    torch.manual_seed(SEED)
    g = GPT(data).to(DEVICE)
    g.fit(training_steps, plot=True, eval_gaps=500)
    plt.savefig('loss.png', dpi=150)
    est = g.estimate_loss(200)
    print()
    print(f"final train: {est['train']:.3f} | final dev: {est['dev']:.3f}")
    g.save(out, vocab)
    return g

def run_demo(model:str='./models/transformer-shspr.txt', vocab:str='./models/vocab.txt'):
    """Samples the demo model."""
    g = GPT(vocab, model).to(DEVICE) # Dataset isn't needed
    torch.manual_seed(SEED)
    print(g.sample(300, temperature=0.8)+'...')


if __name__ == '__main__':
    import sys
    MSG = "char-gpt | Run 'py transformer.py demo/train' to view a demo on Shakespeare."
    if len(sys.argv) == 2:
        match sys.argv[1].lower():
            case 'train' | 't':
                g = run_training()
            case 'demo' | 'sample' | 'd':
                run_demo()
            case _:
                print(MSG)
    else:
        print(MSG)