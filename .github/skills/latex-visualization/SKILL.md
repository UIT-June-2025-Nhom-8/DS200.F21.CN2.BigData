---
name: latex-visualization
description: 'Create LaTeX figures, tables, and diagrams for the DS200 paper. Use for: create LaTeX figure, write LaTeX table, TikZ diagram, pgfplots chart, confusion matrix, architecture diagram, flowchart, beamer slide, algorithm block, pseudocode, LaTeX equation, format results in LaTeX, full document structure.'
argument-hint: "What to create — e.g. 'LaTeX table for main results', 'TikZ training pipeline', 'pgfplots ROC curve', 'Algorithm block for training loop', 'Full IEEE document shell'"
---

# LaTeX Visualization

## When to Use

- Generating compilable LaTeX for any figure, table, or diagram in the paper
- Converting result CSVs from `reports/` into `booktabs` tables
- Creating TikZ architecture or pipeline diagrams
- Plotting training curves, ROC curves, or robustness charts with `pgfplots`
- Writing pseudocode algorithm blocks
- Producing a full IEEE-style document shell

## Constraints

- **All `\caption{}`, section headings, and descriptive text in LaTeX output must be in Vietnamese.** Technical terms (model names, metric labels, axis labels like "Accuracy", "Loss") may remain in English for clarity.
- Never fabricate metric values — read from `reports/` or `artifacts/`, or ask the user to supply them
- Never use deprecated packages (`epsfig`, `subfigure` → use `subcaption`)
- Save all `.tex` files to `reports/latex/`
- Every `figure` and `table` environment must have `\caption{}` and `\label{}`

## Procedure

### 1. Read Data First

Before generating any table or chart, search `reports/` and `artifacts/` for the relevant numbers.

### 2. Choose the Right Tool

| Need                             | Tool                                  |
| -------------------------------- | ------------------------------------- |
| Metrics table                    | `booktabs` + `siunitx`                |
| Architecture / flowchart         | `tikz` + `positioning`, `arrows.meta` |
| Training curves, ROC, robustness | `pgfplots`                            |
| Confusion matrix heatmap         | `pgfplots` matrix plot                |
| Pseudocode                       | `algorithm2e`                         |
| Equations                        | `amsmath`                             |
| Slides                           | Beamer `metropolis` theme             |
| Full paper                       | `IEEEtran` class                      |

### 3. Templates

#### Results Table

```latex
% Required: \usepackage{booktabs,multirow,siunitx}
\begin{table}[t]
  \centering
  \caption{Kết quả đánh giá trên tập kiểm tra.}
  \label{tab:main_results}
  \begin{tabular}{lSSSSS}
    \toprule
    \textbf{Model} & {\textbf{Acc (\%)}} & {\textbf{P}} & {\textbf{R}} & {\textbf{F1}} & {\textbf{AUC}} \\
    \midrule
    % rows — use \textbf{} for best value per column
    \bottomrule
  \end{tabular}
\end{table}
```

#### TikZ Block Diagram (architecture / pipeline)

```latex
% Required: \usepackage{tikz}
% \usetikzlibrary{shapes.geometric,arrows.meta,positioning,fit,backgrounds}
\begin{figure}[t]
  \centering
  \begin{tikzpicture}[
    block/.style={draw, rounded corners, minimum width=2.2cm, minimum height=0.8cm,
                  fill=blue!10, font=\small},
    arrow/.style={-Stealth, thick},
    node distance=0.6cm and 1.2cm
  ]
    % nodes and \draw[arrow] edges here
  \end{tikzpicture}
  \caption{Kiến trúc mô hình phân loại EfficientNet-B0.}
  \label{fig:architecture}
\end{figure}
```

For the 3-stage diagram: color frozen layers `fill=gray!20`, trainable layers `fill=green!20`.

#### pgfplots Training Curve

```latex
% Required: \usepackage{pgfplots}, \pgfplotsset{compat=1.18}
\begin{figure}[t]
  \centering
  \begin{tikzpicture}
    \begin{axis}[
      xlabel={Epoch}, ylabel={Loss},
      legend pos=north east,
      grid=major, grid style={dashed,gray!30},
      width=0.9\linewidth, height=5cm
    ]
      \addplot[blue,thick] coordinates { }; \addlegendentry{Train}
      \addplot[red,dashed,thick] coordinates { }; \addlegendentry{Val}
    \end{axis}
  \end{tikzpicture}
  \caption{Loss huấn luyện và kiểm định (Stage 1).}
  \label{fig:loss_stage1}
\end{figure}
```

#### Algorithm Block

```latex
% Required: \usepackage[ruled,vlined]{algorithm2e}
\begin{algorithm}[t]
\caption{Quy trình huấn luyện 3 giai đoạn EfficientNet-B0}
\label{alg:training}
\KwIn{Pre-trained EfficientNet-B0 $f_\theta$, datasets $\mathcal{D}$}
\KwOut{Best checkpoint $\theta^*$}
\For{stage $s \in \{1,2,3\}$}{
  Freeze/unfreeze layers per stage $s$\;
  \For{epoch $= 1$ \KwTo $E_{\max}$}{
    \ForEach{$(x,y) \in \mathcal{D}_{\text{train}}$}{
      $\hat{y} \leftarrow f_\theta(x)$ \tcp*{AMP forward}
      $\mathcal{L} \leftarrow \text{BCEWithLogits}(\hat{y},y)$\;
      Update $\theta$ via AdamW\;
    }
    \lIf{val\_loss improved}{save $\theta^*$}
    \lIf{EarlyStopping triggered}{\textbf{break}}
  }
}
\Return $\theta^*$
\end{algorithm}
```

#### Key Equations

```latex
% BCE loss
\begin{equation}
  \mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}
    \bigl[y_i\log\sigma(\hat{y}_i)+(1-y_i)\log(1-\sigma(\hat{y}_i))\bigr]
\end{equation}

% ImageNet normalization
\begin{equation}
  x_{\text{norm}} = \frac{x/255 - \mu}{\sigma},\quad
  \mu=[0.485,0.456,0.406],\quad\sigma=[0.229,0.224,0.225]
\end{equation}
```

#### Full IEEE Document Shell

```latex
% Required: IEEEtran class — install via TeX Live: tlmgr install ieeetran
\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage{booktabs,multirow,siunitx,graphicx,subcaption}
\usepackage{tikz,pgfplots}
\usepackage[ruled,vlined]{algorithm2e}
\usepackage{amsmath,amssymb,hyperref}
\pgfplotsset{compat=1.18}
\usetikzlibrary{shapes.geometric,arrows.meta,positioning,fit,backgrounds}

\title{Detection of AI-Generated Faces Using Multi-Dataset Training and EfficientNet-B0}
\author{\IEEEauthorblockN{...}}

\begin{document}
\maketitle
\begin{abstract}...\end{abstract}
\section{Introduction}
\section{Related Work}
\section{Methodology}
\section{Experiments}
\section{Discussion}
\section{Conclusion}
\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
```

Compile with: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

### 4. Output Format

Always respond with:

1. Compilable LaTeX block with `% Required:` package comment header
2. Compile instruction (`pdflatex` or `xelatex`)
3. Save path under `reports/latex/`
4. What to customize (placeholder values, figure paths, actual data coordinates)
