"""bsvae-train and bsvae-sweep-k must build identical models and losses."""

from bsvae.cli import train as train_cli
from bsvae.cli import sweep_k as sweep_cli
from bsvae.cli._common import gmm_loss_kwargs, model_kwargs


def test_train_and_sweep_share_loss_and_model_config(tmp_path):
    common = [
        "--dataset", "x.csv", "--outdir", str(tmp_path), "--no-cuda",
        "--normalize-input", "--bal-type", "uniform", "--latent-corr-strength", "0.5",
        "--masked-recon", "--hier-strength", "0.2", "--tx2gene", "t.tsv",
        "--isoform-stratified", "--free-bits", "0.0",
    ]
    a = train_cli.parse_args(["run"] + common)
    b = sweep_cli.parse_args(["sweep"] + common)
    assert gmm_loss_kwargs(a) == gmm_loss_kwargs(b)
    assert model_kwargs(a, 50) == model_kwargs(b, 50)


def test_defaults_agree_between_cli_entry_points(tmp_path):
    a = train_cli.parse_args(["run", "--dataset", "x.csv"])
    b = sweep_cli.parse_args(["sweep", "--dataset", "x.csv"])
    assert gmm_loss_kwargs(a) == gmm_loss_kwargs(b)
    assert a.free_bits == 0.0 and b.free_bits == 0.0
    assert a.bal_type == "entropy" and a.kl_per_element is True
