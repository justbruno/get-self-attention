import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from selfatt import device


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


class TransformerPlotter:
    def __init__(self, decode):
        self.decode = decode
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

    def plot_head(self, input, key, query, value, K, Q, wei, V, output):
        layout = [
            ['Input', 'Query', 'Key', 'Value'],
            ['Q', 'K', 'mask', 'V'],
            ['Output']
        ]

        matrices = {
            'Input': input, 'Key': key, 'Query': query, 'Value': value,
            'K': K, 'Q': Q, 'mask': wei, 'V': V, 'Output': output
        }
        return layout, matrices

    def plot_embeddings(self, idx, tok_emb, pos_emb, x):
        tokens_label = f'Tokens:{self.decode(idx.tolist()[0])}'
        layout = [
            [tokens_label],
            ['Token Emb.', 'Pos Emb.', 'Combined Emb.']
        ]

        matrices = {
            tokens_label: idx, 'Token Emb.': tok_emb, 'Pos Emb.': pos_emb,
            'Combined Emb.': x
        }
        return layout, matrices

    def plot_logits(self, logits):
        layout = [
            ['Logits']
        ]

        matrices = {
            'Logits': logits
        }
        return layout, matrices

    def do_plot(self, ax, layout, matrices, precision=3, font_size=8):
        ax.axis("off")

        cell_w, cell_h = 0.08, 0.08
        y_cursor = 0.95
        vspace = 0.1
        for row in layout:
            x_cursor = 0.05
            row_height = 0
            for name in row:
                M = matrices[name]
                M_fmt = np.vectorize(lambda x: f"{x:.{precision}f}")(M)
                nrows, ncols = M.shape
                width = ncols * cell_w
                height = nrows * cell_h
                table = ax.table(
                    cellText=M_fmt,
                    loc='center',
                    cellLoc='center',
                    bbox=[x_cursor, y_cursor - height, width, height],
                    transform=ax.transAxes
                )
                for _, cell in table.get_celld().items():
                    cell.set_fontsize(font_size)
                    cell.PAD = 0.1
                    ax.text(
                        x_cursor + width / 2,
                        y_cursor + 0.01,
                        name,
                        ha='center', va='bottom',
                        fontsize=font_size + 1,
                        fontweight='bold',
                        transform=ax.transAxes
                    )
                x_cursor += width + 0.03
                row_height = max(row_height, height)
            y_cursor -= row_height + vspace

    def plot_for_input(self, model, idx, n_embd, loss_str=None):
        final_layout = []
        final_matrices = {}
        _, T = idx.shape
        tok_emb = model.token_embedding_table(idx)
        pos_emb = model.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        layout, matrices = self.plot_embeddings(to_numpy(idx), to_numpy(tok_emb)[0],
                                                to_numpy(pos_emb), to_numpy(x)[0])
        final_layout.extend(layout)
        final_matrices = {**final_matrices, **matrices}

        for b in model.blocks:
            for h in b.sa.heads:
                x_input = b.ln1(x)
                K, Q, V, wei, out = h.compute_variables(x_input, dropout=False)
                layout, matrices = self.plot_head(input=to_numpy(x_input)[0],
                                                  key=to_numpy(
                                                      h.key.weight) if not isinstance(
                                                      h.key,
                                                      nn.Identity) else np.ones(
                                                      (n_embd, n_embd)),
                                                  query=to_numpy(
                                                      h.query.weight) if not isinstance(
                                                      h.query, nn.Identity) else np.ones(
                                                      (n_embd, n_embd)),
                                                  value=to_numpy(
                                                      h.value.weight) if not isinstance(
                                                      h.value, nn.Identity) else np.ones(
                                                      (n_embd, n_embd)),
                                                  K=to_numpy(K)[0],
                                                  Q=to_numpy(Q)[0],
                                                  wei=to_numpy(wei)[0],
                                                  V=to_numpy(V)[0],
                                                  output=to_numpy(out)[0]
                                                  )
                final_layout.extend(layout)
                final_matrices = {**final_matrices, **matrices}

        logits, _ = model(idx)
        layout, matrices = self.plot_logits(to_numpy(logits)[0])
        final_layout.extend(layout)
        final_matrices = {**final_matrices, **matrices}

        self.ax.clear()
        if loss_str is not None:
            self.ax.set_title(loss_str)
        self.do_plot(self.ax, final_layout, final_matrices)
