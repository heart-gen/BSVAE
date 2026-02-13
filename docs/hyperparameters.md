# Hyperparameter Configuration

BSVAE reads experiment defaults from INI-style configuration files. Each section defines a coherent preset, and values can be overridden with CLI flags. The parser uses Python's `ast.literal_eval` to interpret numbers, booleans, and lists.

## File Structure
An `.ini` file may contain multiple sections sharing common settings. Include shared blocks (e.g., `[Common_genenet]`) and reference them by copying key/value pairs into your custom section or by using the provided presets.

## Sample Configuration
The bundled `hyperparam.ini` illustrates recommended defaults. Comments highlight the role of each block.

```ini
[Custom]
    # General options
    log_level = "info"
    no_progress_bar = False
    no_cuda = False
    seed = 1234

    # Training options
    epochs = 100
    batch_size = 64
    lr = 5e-4
    checkpoint_every = 30
    dataset = "genenet"
    experiment = "structured"

    # Model options
    model = "StructuredFactorVAE"
    latent_dim = 30
    hidden_dims = [256, 128]   # encoder hidden layers
    dropout = 0.1
    init_sd = 0.02
    learn_var = True           # decoder per-gene variance

    # Loss options
    loss = "beta"
    beta = 1.0                 # KL weight
    l1_strength = 1e-3         # sparsity regularizer
    lap_strength = 1e-4        # Laplacian smoothness
    coexpr_strength = 0.1    # co-expression preservation

    # Evaluations
    is_metrics = True
    no_test = False
    is_eval_only = False
    eval_batchsize = 512


# ### DATASET COMMON ###
[Common_genenet]
    dataset = "genenet"
    checkpoint_every = 100
    epochs = 400


# ### LOSS COMMON ###
[Common_VAE]
    loss = "VAE"
    lr = 5e-4
[Common_beta]
    loss = "beta"
    beta = 4.0
    lr = 5e-4


# ### EXPERIMENT SPECIFIC ###

[VAE_genenet]
    dataset = "genenet"
    model = "StructuredFactorVAE"
    loss = "VAE"
    latent_dim = 8
    epochs = 400
    checkpoint_every = 50
    lr = 1e-4
    l1_strength = 1e-3
    lap_strength = 1e-4

[beta_genenet]
    dataset = "genenet"
    model = "StructuredFactorVAE"
    loss = "beta"
    beta = 4.0
    latent_dim = 8
    epochs = 400
    checkpoint_every = 50
    lr = 1e-4
    l1_strength = 5e-4
    lap_strength = 5e-5

[debug]
    epochs = 1
    log_level = "debug"
    no_test = True
```

## Key Parameters
| Parameter | Meaning | Typical Range / Effect |
| --- | --- | --- |
| `latent_dim` | Number of latent factors. | Default is 30; typical range 30–50 for module recovery tasks. |
| `beta` | KL divergence weight. | Larger β enforces disentanglement; too large may underfit reconstructions. |
| `l1_strength` | Sparsity penalty on decoder loadings. | Higher values yield sparser, more interpretable factors. |
| `lap_strength` | Graph Laplacian smoothness weight. | Controls PPI coherence; increase when using dense interaction networks. |
| `coexpr_strength` | Co-expression preservation loss weight. | Default 0.1; recommended range 0.1–1.0 for stronger correlation matching. |
| `learn_var` | Learn per-gene reconstruction variance. | Enabled by default (`True`) for heteroscedastic Gaussian reconstruction. |
| `epochs`, `batch_size`, `lr` | Training schedule and optimizer settings. | Adjust to balance convergence speed and stability. |

## Debugging Presets
Use the `[debug]` section for fast smoke tests. It shortens training to a single epoch, enables verbose logging, and skips evaluation. Customize similar sections for automated CI pipelines or hardware checks.

💡 **Tip:** Keep experiment-specific overrides minimal—store the shared defaults in `[Custom]` and create new sections that only redefine parameters that change between studies.


Soft-thresholded network extraction supports a `soft_power` parameter (default 6.0) for WGCNA-style adjacency shaping.
