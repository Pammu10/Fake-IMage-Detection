import json
import os
from datetime import date

RESULTS_PATH = os.path.join("outputs", "results.json")
LATEX_DIR = os.path.join("outputs", "latex")
TEX_PATH = os.path.join(LATEX_DIR, "ai_vs_real_slides.tex")
PDF_PATH = os.path.join(LATEX_DIR, "ai_vs_real_slides.pdf")


def esc(text: str) -> str:
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    out = text
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def p(path: str) -> str:
    return path.replace("\\", "/")


def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError("Run train.py first so outputs/results.json exists.")

    os.makedirs(LATEX_DIR, exist_ok=True)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        r = json.load(f)

    ds = r["dataset"]
    best = r["best_config"]
    fm = r["final_test_metrics"]
    kf = r["kfold_results"]
    sweep = r["hyperparameter_sweep"]
    g = r["graphs"]

    table_rows = "\n".join(
        [
            f"{esc(row['Config'])} & {row['Learning Rate']} & {row['Batch Size']} & {row['Epochs']} & {row['Best Val Acc']:.4f} & {row['Best Val Loss']:.4f} & {row['Best Val AUC']:.4f} \\\\" for row in sweep
        ]
    )

    config_summary = ", ".join(
      [
        f"{row['Config']}({row['Learning Rate']}, {row['Batch Size']}, {row['Epochs']})"
        for row in sweep
      ]
    )

    kfold_rows = "\n".join(
        [
            f"{row['fold']} & {row['val_accuracy']:.4f} & {row['val_auc']:.4f} \\\\" for row in kf["folds"]
        ]
    )

    tex = f"""
\\documentclass[aspectratio=169]{{beamer}}
\\usetheme{{Madrid}}
\\usecolortheme{{seahorse}}
\\setbeamertemplate{{navigation symbols}}{{}}

\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{tikz}}
\\usetikzlibrary{{positioning}}
\\usepackage{{array}}
\\usepackage{{amsmath}}

\\title{{AI vs Real Image Detection}}
\\author{{Student Name: <Your Name>}}
\\date{{{date.today()}}}

\\begin{{document}}

\\begin{{frame}}
  \\titlepage
\\end{{frame}}

\\begin{{frame}}{{Problem Statement}}
\\begin{{itemize}}
  \\item Goal: classify images as real photos or AI-generated images.
  \\item Why it matters: synthetic media is growing and can spread misinformation.
  \\item Challenge: AI images are realistic but contain subtle global inconsistencies.
\\end{{itemize}}
\\end{{frame}}

\\begin{{frame}}{{Dataset (CIFAKE)}}
\\begin{{itemize}}
  \\item Total samples used: {ds['total_samples']}
  \\item Classes: {esc(', '.join(ds['class_names']))} (balanced)
  \\item Split: Train={ds['split']['train']}, Val={ds['split']['val']}, Test={ds['split']['test']}
\\end{{itemize}}
\\end{{frame}}

\\begin{{frame}}{{Architecture: ViT + Custom Head}}
\\centering
\\begin{{tikzpicture}}[scale=0.85, node distance=1.2cm, auto, >=latex]
  \\tiny
  \\node[draw, rounded corners, align=center, text width=1.2cm] (in) {{Input\\\\224×224\\\\RGB}};
  \\node[draw, rounded corners, align=center, right=of in, text width=1.3cm] (vit) {{Frozen ViT\\\\Base}};
  \\node[draw, rounded corners, align=center, right=of vit, text width=1.2cm] (cls) {{CLS\\\\Token\\\\768-d}};
  \\node[draw, rounded corners, align=center, right=of cls, text width=1.4cm] (h1) {{Linear 768\\\\$\\to$ 256\\\\+ ReLU + Drop}};
  \\node[draw, rounded corners, align=center, right=of h1, text width=1.2cm] (out) {{Linear 256\\\\$\\to$ 2\\\\Fake/Real}};

  \\draw[->] (in) -- (vit);
  \\draw[->] (vit) -- (cls);
  \\draw[->] (cls) -- (h1);
  \\draw[->] (h1) -- (out);
\\end{{tikzpicture}}

\\vspace{{0.3cm}}
\\small ViT backbone frozen; only custom head trained. CLS token summarizes image for classification.
\\end{{frame}}

\\begin{{frame}}{{Training Setup}}
\\begin{{itemize}}
  \\item Loss: {esc(r['training_setup']['loss'])}
  \\item Optimizer: {esc(r['training_setup']['optimizer'])}
  \\item Scheduler: {esc(r['training_setup']['scheduler'])}
  \\item Configs from latest run: {esc(config_summary)}
\\end{{itemize}}
\\end{{frame}}

\\begin{{frame}}{{Hyperparameter Comparison}}
\\scriptsize
\\begin{{tabular}}{{lcccccc}}
\\toprule
Config & LR & Batch & Epochs & Val Acc & Val Loss & Val AUC \\\\
\\midrule
{table_rows}
\\bottomrule
\\end{{tabular}}
\\end{{frame}}

\\begin{{frame}}{{Training Curves}}
  \\centering
  \\includegraphics[width=0.95\\linewidth,height=0.78\\textheight,keepaspectratio]{{{p(g['training_curves'])}}}
\\end{{frame}}

\\begin{{frame}}{{Confusion Matrix}}
  \\centering
  \\includegraphics[width=0.85\\linewidth,height=0.78\\textheight,keepaspectratio]{{{p(g['confusion_matrix'])}}}
\\end{{frame}}

\\begin{{frame}}{{ROC Curve}}
  \\centering
  \\includegraphics[width=0.85\\linewidth,height=0.78\\textheight,keepaspectratio]{{{p(g['roc_curve'])}}}
\\end{{frame}}

\\begin{{frame}}{{Precision-Recall Curve}}
  \\centering
  \\includegraphics[width=0.85\\linewidth,height=0.78\\textheight,keepaspectratio]{{{p(g['precision_recall_curve'])}}}
\\end{{frame}}

\\begin{{frame}}{{Confidence Score Distribution}}
  \\centering
  \\includegraphics[width=0.9\\linewidth,height=0.78\\textheight,keepaspectratio]{{{p(g['confidence_distribution'])}}}
\\end{{frame}}

\\begin{{frame}}{{K-Fold Cross Validation Results}}
\\small
\\begin{{tabular}}{{lcc}}
\\toprule
Fold & Val Accuracy & Val AUC \\\\
\\midrule
{kfold_rows}
\\midrule
Mean$\\pm$Std & {kf['mean_accuracy']:.4f}$\\pm${kf['std_accuracy']:.4f} & {kf['mean_auc']:.4f}$\\pm${kf['std_auc']:.4f} \\\\
\\bottomrule
\\end{{tabular}}

\\vspace{{0.3cm}}
\\footnotesize K-Fold means train 3 times with different validation splits to show consistency.
\\end{{frame}}

\\begin{{frame}}{{Key Findings and Conclusion}}
\\begin{{itemize}}
  \\item Best config: {esc(best['name'])} (lr={best['lr']}, batch={best['batch_size']}, epochs={best['epochs']})
  \\item Final Test Accuracy: {fm['accuracy']:.4f}
  \\item Final Test AUC: {fm['auc']:.4f}
  \\item K-Fold Mean Accuracy: {kf['mean_accuracy']:.4f}$\\pm${kf['std_accuracy']:.4f}
  \\item External OOD Accuracy: {r.get('external_test_metrics', {}).get('accuracy', 0):.4f}
  \\item Model performs strongly with frozen ViT + lightweight classifier head.
  \\item Future work: unfreeze last ViT blocks, stronger augmentations, larger training budget.
\\end{{itemize}}
\\end{{frame}}

\\begin{{frame}}{{References}}
\\footnotesize
\\begin{{itemize}}
  \\item CNNDetect --- Wang et al., CVPR 2020
  \\item UniversalFakeDetect --- Ojha et al., CVPR 2023
  \\item DIRE --- Wang et al., ICCV 2023
  \\item FreqNet --- Tan et al., AAAI 2024
\\end{{itemize}}
\\end{{frame}}

\\end{{document}}
"""

    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"Wrote LaTeX slides: {TEX_PATH}")
    print(f"Expected PDF after compilation: {PDF_PATH}")


if __name__ == "__main__":
    main()
