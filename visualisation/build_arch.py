"""Build the architecture files read by the visualiser (visualisation/arch/).

Instantiates the real model of a submission on real data (``prepare`` builds
the index, the pairs and the per gene and per run tables), measures it with
one forward on a few rows, and describes the diagram.

- ``hybrid``: the gene tower, the GLM effect tables and the run tower, side
  by side, then the interaction, log mu, log sigma and the NB loss.
- ``transformer``: the tokens of a row, the blocks (or the mlp mixer), the
  head and the per-level additive effects, then log mu, log sigma and the NB
  loss.

    python visualisation/build_arch.py                       # both, debug data
    python visualisation/build_arch.py --model transformer --param norm_place=pre
    python visualisation/build_arch.py --mode train          # real sizes (pod, 16 GB+)
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "src"))

from archkit import Probe, link, node, write  # noqa: E402

SUBMISSIONS = {
    "hybrid": ROOT / "submissions" / "hybrid_esm" / "submission.py",
    "transformer": ROOT / "submissions" / "transformer" / "submission.py",
}
B = 37  # rows of the measuring forward, a size no real dimension takes

TABLES = {
    "b0": ("Constante", "1 valeur", "Niveau moyen de log μ, avant tout effet."),
    "a_genome": ("Génome", "niveaux", "Niveau propre à chaque génome."),
    "a_gene": ("Gène", "niveaux",
               "Niveau propre au gène, masqué au hasard pendant "
               "l'entraînement (1 famille sur 10) pour que la tour gène "
               "apprenne à le deviner."),
    "b_project": ("Projet", "niveaux", "Le laboratoire, ses habitudes."),
    "b_condition": ("Condition", "niveaux",
                    "Mêmes conditions curées dans un génome. Masquée pour 1 "
                    "condition sur 10 pendant l'entraînement."),
    "b_run": ("Run", "niveaux", "Niveau propre au run."),
    "c_gp": ("Gène × projet", "paires",
             "Le niveau du gène dans les runs d'un projet. Une paire vue "
             "moins de 2 fois reste à 0."),
    "c_gc": ("Gène × condition", "paires",
             "Le niveau du gène dans une condition. Une paire vue moins de "
             "2 fois reste à 0."),
    "rho0": ("ρ constante", "1 valeur", "Dispersion moyenne, log σ."),
    "rho_genome": ("ρ génome", "niveaux", "Dispersion propre au génome."),
    "rho_gene": ("ρ gène", "niveaux", "Dispersion propre au gène."),
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_data(mode):
    """The dataset's own get_data, as the platform runs it."""
    from evaluate_local import lean_read_parquet

    dataset = load_module(ROOT / "benchmark" / "datasets" /
                          "bacterial_expression.py", "bacterial_expression")
    home = ROOT / "benchmark" / "data"
    dataset.get_data_path = (lambda key=None: (home / key).resolve()
                             if key else home.resolve())
    dataset.pd.read_parquet, original = lean_read_parquet, \
        dataset.pd.read_parquet
    try:
        return dataset.Dataset.get_instance(mode=mode).get_data()
    finally:
        dataset.pd.read_parquet = original


def table(net, key):
    label, unit, doc = TABLES[key]
    n = net.glm[key].weight.shape[0]
    count = unit if n == 1 else f"{n:,} {unit}"
    return node(label, "embed", module=f"glm.{key}", info=[count], doc=doc)


def linear(path, label, layer):
    return node(label, "linear", module=path,
                info=[f"{layer.in_features} → {layer.out_features}"])


def describe(net, n_in_gene, n_in_run, dim):
    tw = net.towers
    gene_tower = node(
        "Tour gène", "stack", module="towers.gene", scale=2,
        info=[f"{n_in_gene} + 8 → {tw.gene[0].out_features} → "
              f"{tw.gene[2].out_features} → {tw.gene[4].out_features}"],
        doc="Prédit ce que le GLM ne sait pas d'un gène jamais vu : son "
            "niveau, sa dispersion et son vecteur de réponse aux conditions.",
        children=[
            node("Entrées du gène", "input", shape=f"[G, {n_in_gene}]",
                 info=["39 signaux (CAI, opéron, origine...)",
                       "embedding ESM-2 de la protéine (ACP)",
                       "2 indicateurs de présence"]),
            node("Embedding génome", "embed", module="towers.genome",
                 id="genome_emb", info=["8 dimensions"],
                 doc="Partagé entre les deux tours."),
            node("Concat", "concat"),
            linear("towers.gene.0", "Linear 1", tw.gene[0]),
            node("GELU", "act", module="towers.gene.1"),
            linear("towers.gene.2", "Linear 2", tw.gene[2]),
            node("GELU", "act", module="towers.gene.3"),
            linear("towers.gene.4", "Sortie", tw.gene[4]),
            node("s_gène, ρ_gène, u_gène", "output",
                 info=["niveau, dispersion", f"vecteur de {dim}"]),
        ])
    run_tower = node(
        "Tour run", "stack", module="towers.run", scale=2,
        info=[f"{n_in_run} + 8 → {tw.run[0].out_features} → "
              f"{tw.run[2].out_features}"],
        doc="Prédit le niveau d'un run jamais vu et son vecteur de "
            "conditions, à partir de sa fiche de curation.",
        children=[
            node("Entrées du run", "input", shape=f"[R, {n_in_run}]",
                 info=["conditions curées en one-hot",
                       "température, pH, agitation, OD600",
                       "indicateurs de valeur connue"]),
            node("Embedding génome", "embed", id="genome_emb_run",
                 info=["partagé avec la tour gène"]),
            node("Concat", "concat"),
            linear("towers.run.0", "Linear 1", tw.run[0]),
            node("GELU", "act", module="towers.run.1"),
            linear("towers.run.2", "Sortie", tw.run[2]),
            node("s_run, v_run", "output", info=["niveau", f"vecteur de {dim}"]),
        ])
    glm = node(
        "GLM, chemin résiduel", "stack", scale=2, layout="col", flow="none",
        info=["une valeur libre par niveau", "pénalité L2, 0 si jamais vu"],
        doc="Le modèle de référence. Ses effets passent tout droit jusqu'à "
            "log μ, les tours n'apprennent que ce qu'il ne sait pas.",
        children=[
            node("Effets du gène", "group", layout="row", flow="none",
                 children=[table(net, "a_genome"), table(net, "a_gene")]),
            node("Effets du run", "group", layout="row", flow="none",
                 children=[table(net, k) for k in
                           ("b_project", "b_condition", "b_run")]),
            node("Interactions", "group", layout="row", flow="none",
                 children=[table(net, "c_gp"), table(net, "c_gc")]),
            node("Dispersion", "group", layout="row", flow="none",
                 children=[table(net, k) for k in
                           ("rho0", "rho_genome", "rho_gene")]),
            node("Constante", "group", layout="row", flow="none",
                 children=[table(net, "b0")]),
        ])
    branches = node("branches", "group", box=False, layout="row", flow="par",
                    children=[gene_tower, glm, run_tower])
    heads = node("sorties", "group", box=False, layout="row", flow="par",
                 children=[
                     node("log μ", "head", shape="[B]",
                          info=["offset + GLM + s_gène + s_run",
                                f"+ ⟨u_gène, v_run⟩ / √{dim}"]),
                     node("log σ", "head", shape="[B]",
                          info=["ρ du GLM + ρ_gène", "+ δ du cas (7 cas)"],
                          doc="δ est calibré après l'entraînement sur des "
                              "holdouts construits comme les régions A, B, "
                              "C et D. Il rend σ plus petit pour un gène ou "
                              "un run inconnu."),
                 ])
    root = node("Modèle hybride ESM-2", "stack", layout="row", children=[
        node("Observation", "input", scale=3, shape="[B]",
             info=["un gène dans un run", "offset = profondeur + longueur"]),
        branches,
        node("Interaction", "mul", info=[f"⟨u_gène, v_run⟩ / √{dim}"]),
        heads,
        node("NB-NLL", "loss", scale=3,
             info=["−log P(y | μ, σ)", "plafond 100 nats"],
             doc="La métrique du challenge, utilisée telle quelle comme "
                 "perte."),
    ])
    links = [link("genome_emb", "genome_emb_run", "partagé", kind="embed")]
    return root, links


def describe_transformer(net, cfg, tables):
    """Tokens, blocks, head and id biases of the expression transformer."""
    d = cfg.d_model

    def attention(i, att):
        path = f"blocks.{i}.attn"
        kids = [node("Q", "linear", module=f"{path}.q",
                     info=[f"{d} → {att.h} têtes × {att.hd}"]),
                node("K", "linear", module=f"{path}.k",
                     info=[f"{d} → {att.kv} × {att.hd}"]),
                node("V", "linear", module=f"{path}.v",
                     info=[f"{d} → {att.kv} × {att.hd}"])]
        if cfg.qk_norm:
            kids += [node("QK-norm (q)", "norm", module=f"{path}.q_norm"),
                     node("QK-norm (k)", "norm", module=f"{path}.k_norm")]
        kids.append(node("SDPA" if att.sdpa else "softmax(QKᵀ/√d) V",
                         "sdpa" if att.sdpa else "softmax",
                         info=([f"softcap {cfg.softcap:g}"] if cfg.softcap
                               else ["tous les tokens se voient"])))
        kids.append(node("Sortie", "linear", module=f"{path}.out"))
        return node("Attention", "attention", module=path,
                    badge=cfg.attn.upper(),
                    info=[f"{att.h} têtes, {att.kv} KV"], children=kids)

    def ffn(i, f):
        path = f"blocks.{i}.ffn"
        if cfg.moe_experts:
            experts = [node(f"Expert {j}", "expert",
                            module=f"{path}.experts.{j}")
                       for j in range(cfg.moe_experts)]
            return node("MoE", "moe", module=path,
                        badge=f"top-{cfg.moe_top_k}",
                        info=[f"{cfg.moe_experts} experts {cfg.ffn}"],
                        children=[node("Routeur", "router",
                                       module=f"{path}.router"),
                                  node("Experts", "group", layout="grid",
                                       flow="none", cols=4,
                                       children=experts)])
        return node("FFN", "ffn", module=path, badge=cfg.ffn,
                    info=[f"{d} → {f.down.in_features} → {d}"],
                    children=[node("Montée", "linear", module=f"{path}.up"),
                              node(cfg.ffn, "act"),
                              node("Descente", "linear",
                                   module=f"{path}.down")])

    def norm(i, name):
        return node(cfg.norm, "norm", module=f"blocks.{i}.{name}")

    blocks = []
    for i, blk in enumerate(net.blocks):
        a, f = attention(i, blk.attn), ffn(i, blk.ffn)
        if cfg.norm_place == "post":
            kids = [a, node("+", "add"), norm(i, "norm1"),
                    f, node("+", "add"), norm(i, "norm2")]
        elif cfg.norm_place == "pre":
            kids = [norm(i, "norm1"), a, node("+", "add"),
                    norm(i, "norm2"), f, node("+", "add")]
        else:
            kids = [norm(i, "norm1"), a, norm(i, "post1"), node("+", "add"),
                    norm(i, "norm2"), f, norm(i, "post2"), node("+", "add")]
        blocks.append(node(f"Bloc {i + 1}", "block", module=f"blocks.{i}",
                           info=[f"norme {cfg.norm_place}"], children=kids))

    n_fields = tables["run_cat"].shape[1]
    sources = node("Sources", "group", layout="grid", cols=2, flow="none",
                   children=[
        node("Gène", "embed", module="gene_emb",
             info=[f"{tables['n_gene']:,} + MASK"],
             doc="Un vecteur libre par gène vu. Un gène jamais vu (région "
                 "C) lit le token MASK, appris avec la dropout de niveau."),
        node("Protéine et signaux", "linear", module="features",
             info=[f"{tables['x_gene'].shape[1]} → {d}",
                   "ESM-2 (ACP), CAI, opéron..."]),
        node("Génome", "embed", module="genome_emb",
             info=[f"{tables['n_genome']} génomes"]),
        node("Run", "embed", module="run_emb",
             info=[f"{tables['n_run']:,} + MASK"]),
        node("Projet", "embed", module="prj_emb",
             info=[f"{tables['n_prj']:,} + MASK"]),
        node("Condition", "embed", module="cond_emb",
             info=[f"{tables['n_cond']:,} + MASK + aucune"]),
        node("Champs de la fiche", "embed", module="field_emb",
             info=[f"{n_fields} champs, {tables['n_vocab']} valeurs"],
             doc="Milieu, oxygène, phase de croissance, perturbation..."),
        node("Nombres", "linear", module="numbers",
             info=["température, pH, agitation, OD600",
                   "z·w + b, ou token manquant"]),
    ])
    if getattr(net, "n_signals", 0):
        sources["children"].append(node(
            "Signaux en PLR", "embed", module="signal_plr",
            info=[f"{net.n_signals} signaux × {cfg.plr_dim}",
                  f"{cfg.plr_freq} fréquences apprises"],
            doc="Chaque signal du gène passe par sin et cos de fréquences "
                "apprises, puis un Linear et un ReLU qui lui sont propres."))
    if cfg.glm_eta == "token":
        sources["children"].append(node(
            "η du GLM", "linear", module="glm_tokens",
            info=["log μ et log σ du GLM gelé", "masqués comme les ids"]))
    tokens = node("Tokens", "stack", scale=2,
                  info=[f"{net.n_tokens} tokens de {d}"],
                  children=[sources, node("Concat", "concat"),
                            node("CLS + slots", "pos", module="slots",
                                 info=["un embedding par champ"])])
    if cfg.mixer == "mlp":
        kids = [node("Aplatir + Linear", "linear", module="mixer.inp",
                     info=[f"{net.n_tokens} × {d} → {cfg.mlp_hidden}"])]
        for i in range(cfg.n_layers):
            kids.append(node(f"Bloc MLP {i + 1}", "block",
                             module=f"mixer.blocks.{i}", children=[
                node(cfg.norm, "norm", module=f"mixer.blocks.{i}.norm"),
                node("FFN", "ffn", module=f"mixer.blocks.{i}.ffn",
                     badge=cfg.ffn, children=[
                    node("Montée", "linear",
                         module=f"mixer.blocks.{i}.ffn.up"),
                    node(cfg.ffn, "act"),
                    node("Descente", "linear",
                         module=f"mixer.blocks.{i}.ffn.down")]),
                node("+", "add")]))
        kids += [node(cfg.norm, "norm", module="mixer.norm"),
                 node("Linear", "linear", module="mixer.out",
                      info=[f"{cfg.mlp_hidden} → {d}"])]
        mixer = node("MLP sans attention", "stack", module="mixer",
                     info=[f"{cfg.n_layers} blocs de {cfg.mlp_hidden}"],
                     doc="Les mêmes tokens, aplatis : aucune attention. "
                         "Sert à mesurer ce que l'attention apporte.",
                     children=kids)
        body = [tokens, mixer]
    else:
        body = [tokens, node(f"Blocs ×{cfg.n_layers}", "stack",
                             children=blocks)]
    if cfg.norm_place != "post":
        body.append(node("Norme finale", "norm", module="final_norm"))
    body.append(node("Tête", "head", module="head",
                     info=["CLS → log μ, log σ", "à 0 au départ"]))
    transformer = node("Transformer", "stack", scale=2, children=body)
    if cfg.id_bias:
        biases = node("Effets additifs par niveau", "embed", module="bias",
                      scale=2,
                      info=["b0 + génome + gène + run + projet",
                            "ρ0 + ρ gène", "masqués si jamais vus"],
                      doc="Une valeur libre par niveau, apprise dans le "
                          "réseau et démarrée aux estimations de moments. "
                          "Ce n'est pas un GLM séparé : le transformer "
                          "apprend ces effets avec le reste.")
    else:
        biases = node("Constantes b0 et ρ0", "embed", module="bias", scale=2,
                      info=["niveau moyen et dispersion moyenne",
                            "effets par niveau coupés (id_bias=false)"],
                      doc="Les ablations du 2026-11-05 ont montré que les "
                          "effets additifs libres par gène et par run "
                          "gênent : sans eux, le score local passe de 7.567 "
                          "à 7.505. Les embeddings portent seuls le niveau "
                          "de chaque gène et de chaque run.")
    root = node("Transformer d'expression", "stack", layout="row", children=[
        node("Observation", "input", scale=3, shape="[B]",
             info=["un gène dans un run", "offset = profondeur + longueur"]),
        node("branches", "group", box=False, layout="row", flow="par",
             children=[transformer, biases] + ([node(
                 "GLM additif gelé", "embed",
                 info=["η et ρ ajoutés tels quels",
                       "ajusté avant le transformer"],
                 doc="Le transformer n'apprend que l'écart à ce GLM.")]
                 if cfg.glm_eta == "offset" else [])),
        node("+", "add"),
        node("sorties", "group", box=False, layout="row", flow="par",
             children=[node("log μ", "head", shape="[B]",
                            info=["offset + effets + tête"]),
                       node("log σ", "head", shape="[B]",
                            info=["ρ + tête", "+ δ du cas (7 cas)"])]),
        node("NB-NLL", "loss", scale=3,
             info=["−log P(y | μ, σ)", "plafond 100 nats"]),
    ])
    return root, []


def _meta(data):
    meta = {k: v for k, v in data.items()
            if k not in ("train_df", "samples_df", "genomes_dir", "test_sets")}
    meta["submission_dir"] = None
    return meta


def build_transformer(sub, data, mode, params):
    # the submitted configuration, unless --param says otherwise
    cfg = sub.TConfig(**{**getattr(sub, "PRESET", {}), **params})
    model = sub.TransformerModel(_meta(data), cfg)
    gene, run, y, off = model.prepare(data["train_df"], data["samples_df"])
    ix = model.ix
    bias = model._bias_init(gene, run, y, off, np.ones(len(y), bool))
    torch.manual_seed(0)
    net = sub._ExprTransformer(cfg, model.tables, bias).eval()

    rows = np.random.default_rng(0).choice(len(y), B, replace=False)
    g = torch.as_tensor(gene[rows]).long()
    r = torch.as_tensor(run[rows]).long()
    on = [torch.ones(B, dtype=torch.bool) for _ in range(4)]
    probe = Probe(net, lambda: net(g, r, *on),
                  symbols={"B": B, "T": net.n_tokens})
    root, links = describe_transformer(net, cfg, model.tables)
    n_blocks = sum(p.numel() for p in net.blocks.parameters())
    write(HERE / "arch" / "transformer.json", root, probe, name="transformer",
          title="Transformer d'expression",
          subtitle=f"une ligne (gène, run) lue comme {net.n_tokens} tokens, "
                   f"données {mode}",
          summary=[("lignes d'entraînement", f"{len(y):,}"),
                   ("gènes", f"{ix.n_gene:,}"), ("runs", f"{ix.n_run:,}"),
                   ("tokens par ligne", f"{net.n_tokens}"),
                   ("d_model", f"{cfg.d_model}"),
                   ("couches", f"{cfg.n_layers}"),
                   ("têtes", f"{cfg.n_heads} ({cfg.attn})"),
                   ("normalisation", f"{cfg.norm}, {cfg.norm_place}"),
                   ("FFN", cfg.ffn if not cfg.moe_experts else
                    f"MoE {cfg.moe_experts} experts, top {cfg.moe_top_k}"),
                   ("paramètres des blocs", f"{n_blocks:,}")],
          links=links, source="submissions/transformer/submission.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="all",
                        choices=["all", "hybrid", "transformer"])
    parser.add_argument("--mode", default="debug", choices=["debug", "train"])
    parser.add_argument("--features", type=Path, default=None,
                        help="gene_features.npz (default: the clean data "
                        "folder of the mode)")
    parser.add_argument("--param", action="append", default=[],
                        help="TConfig field of the transformer, name=value")
    args = parser.parse_args()

    from evaluate_local import parse_value
    features = args.features or (ROOT / "benchmark" / "data" / "bacteria" /
                                 "clean" / args.mode / "gene_features.npz")
    if features.exists():
        os.environ["BACTERIA_GENE_FEATURES"] = str(features)
    data = load_data(args.mode)
    if args.model in ("all", "transformer"):
        params = {k: parse_value(v) for k, v in
                  (p.split("=", 1) for p in args.param)}
        build_transformer(load_module(SUBMISSIONS["transformer"],
                                      "submission_tf"), data, args.mode,
                          params)
    if args.model in ("all", "hybrid"):
        build_hybrid(load_module(SUBMISSIONS["hybrid"], "submission"), data,
                     args.mode)


def build_hybrid(sub, data, mode):
    model = sub.HybridModel(_meta(data))
    gene, run, y, off = model.prepare(data["train_df"], data["samples_df"])
    ix = model.ix
    init = model._init_params(gene, run, y, off, np.ones(len(y), bool))
    torch.manual_seed(0)
    net = sub._Net(init, model.towers).eval()

    rows = np.random.default_rng(0).choice(len(y), B, replace=False)
    g = torch.as_tensor(gene[rows]).long()
    r = torch.as_tensor(run[rows]).long()
    gp, gc = model.gp.idx[rows], model.gc.idx[rows]
    inputs = dict(
        x_gene=torch.as_tensor(model.x_gene),
        genome_of_gene=torch.as_tensor(ix.genome_of_gene).long(),
        x_run=torch.as_tensor(model.x_run),
        genome_of_run=torch.as_tensor(np.maximum(ix.genome_of_run, 0)).long(),
        g=g, r=r, G=torch.as_tensor(ix.genome_of_gene[gene[rows]]).long(),
        p=torch.as_tensor(ix.prj_of_run[run[rows]]).long(),
        c=torch.as_tensor(np.where(ix.cond_of_run[run[rows]] >= 0,
                                   ix.cond_of_run[run[rows]], ix.n_cond)).long(),
        gp=torch.as_tensor(np.where(gp >= 0, gp, model.gp.n)).long(),
        gc=torch.as_tensor(np.where(gc >= 0, gc, model.gc.n)).long(),
        off=torch.as_tensor(off[rows]))
    probe = Probe(net, lambda: net(**inputs),
                  symbols={"B": B, "G": ix.n_gene, "R": ix.n_run})

    n_in_gene, n_in_run = model.x_gene.shape[1], model.x_run.shape[1]
    root, links = describe(net, n_in_gene, n_in_run, model.dim)
    write(HERE / "arch" / "hybrid-esm.json", root, probe, name="hybrid-esm",
          title="Modèle hybride ESM-2",
          subtitle=f"GLM binomiale négative + tour gène (ESM-2) + tour run, "
                   f"données {mode}",
          summary=[("lignes d'entraînement", f"{len(y):,}"),
                   ("gènes", f"{ix.n_gene:,}"), ("runs", f"{ix.n_run:,}"),
                   ("projets", f"{ix.n_prj:,}"),
                   ("conditions", f"{ix.n_cond:,}"),
                   ("gène × projet", f"{model.gp.n:,}"),
                   ("gène × condition", f"{model.gc.n:,}"),
                   ("entrées tour gène", f"{n_in_gene}"),
                   ("dimension des tours", f"{model.dim}")],
          links=links, source="submissions/hybrid_esm/submission.py")


if __name__ == "__main__":
    main()
