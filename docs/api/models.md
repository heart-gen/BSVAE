# Model API

This section summarizes the primary modeling components exposed by BSVAE. Consult the source files in `src/bsvae/models/` for implementation specifics.

---

## `StructuredFactorVAE`

Defined in [`models/structured.py`](../../src/bsvae/models/structured.py), this class composes an encoder and decoder tailored to gene-expression matrices.

### Constructor

```python
StructuredFactorVAE(
    n_genes: int,
    n_latent: int,
    hidden_dims: Sequence[int] = (512, 256, 128),
    dropout: float = 0.1,
    init_sd: float = 0.02,
    learn_var: bool = False,
    L: Optional[torch.Tensor] = None,
)
```

- **`n_genes`** – Number of observed genes.
- **`n_latent`** – Number of latent biological modules.
- **`hidden_dims`** – Encoder hidden-layer sizes.
- **`dropout`** – Dropout rate applied in the encoder.
- **`init_sd`** – Standard deviation for decoder weight initialization.
- **`learn_var`** – Enables gene-specific decoder variances.
- **`L`** – Optional Laplacian matrix used for smoothness regularization.

### Methods

- `forward(x)` → `(recon_x, mu, logvar, z, log_var)`
- `encode(x)` / `decode(z)` delegated to the encoder and decoder submodules.
- `group_sparsity_penalty(l1_strength)` computes the L1 decoder penalty.
- `laplacian_penalty(L, lap_strength)` applies Laplacian smoothing.
- `reset_parameters(activation="relu")` reinitializes weights.

---

## `StructuredEncoder`

Defined in [`models/encoder.py`](../../src/bsvae/models/encoder.py), maps normalized gene expression (log-CPM/TPM) to latent Gaussian parameters.

### Constructor

```python
StructuredEncoder(
    n_genes: int,
    n_latent: int,
    hidden_dims: list[int] = [512, 256, 128],
    dropout: float = 0.1,
)
```

- **`n_genes`** – Number of input genes (features).
- **`n_latent`** – Dimensionality of latent space (number of biological modules).
- **`hidden_dims`** – Sizes of hidden layers.
- **`dropout`** – Dropout probability applied after hidden layers.

### Methods

- `forward(x)` → `(mu, logvar)` – Returns latent mean and log-variance tensors.

---

## `StructuredDecoder`

Defined in [`models/decoder.py`](../../src/bsvae/models/decoder.py), reconstructs gene expression from latent codes with support for sparsity and Laplacian penalties.

### Constructor

```python
StructuredDecoder(
    n_genes: int,
    n_latent: int,
    mask: torch.Tensor = None,
    init_sd: float = 0.02,
    learn_var: bool = False,
)
```

- **`n_genes`** – Number of genes (G).
- **`n_latent`** – Number of latent factors/modules (K).
- **`mask`** – Optional binary mask of shape (G, K) where 1 means gene↔factor allowed.
- **`init_sd`** – Standard deviation for normal initialization of decoder weights.
- **`learn_var`** – If True, learns gene-specific variance parameters.

### Methods

- `forward(z)` → `(recon_x, log_var)` – Reconstructs gene expression from latent codes.
- `group_sparsity_penalty(l1_strength)` – Computes L1 sparsity penalty on decoder weights.
- `laplacian_penalty(L, lap_strength)` – Computes Laplacian smoothness penalty: tr(W^T L W).

### Attributes

- `W` – Decoder weight matrix of shape (n_genes, n_latent).
- `bias` – Per-gene bias terms.
- `log_var` – Per-gene log-variance (if `learn_var=True`).

---

## Loss Functions

### `BaseLoss`
Located in [`models/losses.py`](../../src/bsvae/models/losses.py), `BaseLoss` integrates reconstruction, KL divergence, and biological regularizers.

```python
BaseLoss(
    beta: float = 1.0,
    l1_strength: float = 1e-3,
    lap_strength: float = 1e-4,
    record_loss_every: int = 50,
)
```

Calling the loss with `(x, recon_x, mu, logvar, model, L=None, storer=None, is_train=True)` returns a scalar objective:

- **Reconstruction term** uses mean squared error or Gaussian NLL if the decoder learns log-variance.
- **KL divergence** encourages latent distributions to remain close to a standard Normal prior.
- **Sparsity penalty** invokes `model.decoder.group_sparsity_penalty` to shrink decoder loadings.
- **Laplacian regularizer** applies `model.decoder.laplacian_penalty` when a PPI Laplacian is attached.

💡 **Tip:** Adjust `beta`, `l1_strength`, and `lap_strength` jointly to balance disentanglement, sparsity, and graph coherence.
