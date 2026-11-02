"""2.5D label-query network.

A study is a bag of 2.5D windows (3 neighbouring slices) from up to 6 series plus 13 zoomed views. One shared timm
backbone turns every window into a 12 x 12 grid of tokens. Each token gets an embedding of its view and of its 3D
position in a common knee frame (X lateral, Y posterior, Z inferior; mm / 70), so all planes and crops line up.
Two readouts, whose logits are averaged:

- label queries: 2 learnable queries per label; 2 pre-LN decoder layers (query self-attention -> cosine
  cross-attention over every token of every window -> FFN). A label's logit is a linear map of the last layer's
  cross-attention output.
- gated attention-MIL (Ilse et al. 2018) over window vectors (mean of the window's tokens + position).

At inference the attention logits of both readouts are multiplied by a temperature (> 1 = sharper), since a dense
test bag has many more windows than a training bag.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

N_SERIES = 6                        # SAG_FS, SAG_N, COR_FS, COR_N, AX_FS, AX_N
PLANE_OF_SERIES = [0, 0, 1, 1, 2, 2]  # 0 sagittal, 1 coronal, 2 axial
HALF_FOV_MM = 70.0                  # the 140 mm crop


def knee_coords(u, v, w, plane):
    """In-plane (u = column, v = row) and through-plane (w = slice) offsets in mm -> (..., 3) knee frame / 70.
    Columns run posterior (sagittal) / lateral (coronal, axial); rows run inferior (sagittal, coronal) / posterior
    (axial); slices run lateral (sagittal) / posterior (coronal) / superior (axial)."""
    X = torch.where(plane == 0, w, u)
    Y = torch.where(plane == 0, u, torch.where(plane == 1, w, v))
    Z = torch.where(plane == 2, -w, v)
    return torch.stack([X, Y, Z], -1) / HALF_FOV_MM


class FourierPE(nn.Module):
    def __init__(self, D, n_freq=8):
        super().__init__()
        self.register_buffer("freq", (2.0 ** torch.arange(n_freq)) * math.pi / 2, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(6 * n_freq, D), nn.GELU(), nn.Linear(D, D))

    def forward(self, c):                     # c (..., 3)
        a = c[..., None] * self.freq          # (..., 3, n_freq)
        return self.mlp(torch.cat([a.sin(), a.cos()], -1).flatten(-2))


class DecoderLayer(nn.Module):
    def __init__(self, D, heads, drop=0.1, ffn_mult=4):
        super().__init__()
        self.h = heads
        self.att_temp = 1.0
        self.log_s = nn.Parameter(torch.full((1, heads, 1, 1), math.log(5.0)))   # cosine-attention scale
        self.n1, self.n2, self.n3, self.nk = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        self.sa = nn.MultiheadAttention(D, heads, dropout=0.0, batch_first=True)
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.ffn = nn.Sequential(nn.Linear(D, ffn_mult * D), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn_mult * D, D))
        self.drop = nn.Dropout(drop)

    def forward(self, q, tok, pos, mask):
        """q (B, Q, D); tok / pos (B, T, D); mask (B, T) True = real token.
        Returns the updated queries and this layer's cross-attention output r (B, Q, D)."""
        x = self.n1(q)
        q = q + self.drop(self.sa(x, x, x, need_weights=False)[0])
        B, Q, D = q.shape
        T, h, dh = tok.shape[1], self.h, D // self.h
        t = self.nk(tok)
        qq = self.q(self.n2(q)).view(B, Q, h, dh).transpose(1, 2)
        kk = self.k(t + pos).view(B, T, h, dh).transpose(1, 2)
        vv = self.v(t).view(B, T, h, dh).transpose(1, 2)
        sc = (self.log_s.exp().clamp(max=20.0) * self.att_temp).to(qq.dtype)
        o = F.scaled_dot_product_attention(F.normalize(qq, dim=-1) * sc, F.normalize(kk, dim=-1), vv,
                                           attn_mask=mask[:, None, None, :], scale=1.0)
        r = self.o(o.transpose(1, 2).reshape(B, Q, D))
        q = q + self.drop(r)
        return q + self.drop(self.ffn(self.n3(q))), r


def _pad_rows(x, m):
    """Pad a (n, ...) batch to m rows by repeating its own rows."""
    n = len(x)
    if m <= n:
        return x
    return torch.cat([x, x[torch.arange(m - n, device=x.device) % n]])


class KQNet(nn.Module):
    def __init__(self, arch, roi_planes, D=384, n_lab=12, q_per=2, dec_layers=2, heads=8, drop=0.1):
        super().__init__()
        self.att_temp = 1.0
        self.bb = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3)
        self.n_lab, self.q_per = n_lab, q_per
        self.p4 = nn.Conv2d(self.bb.feature_info[-1]["num_chs"], D, 1)   # stride-32 features -> tokens
        self.tnorm = nn.LayerNorm(D)
        # view slots: the 6 series, then the zoomed views (crops of a parent series, same plane)
        self.register_buffer("plane_of_slot", torch.tensor(PLANE_OF_SERIES + list(roi_planes)), persistent=False)
        self.slot_emb = nn.Embedding(N_SERIES + len(roi_planes), D)
        self.pe = FourierPE(D)
        self.queries = nn.Parameter(torch.randn(n_lab * q_per, D) * 0.02)
        self.dec = nn.ModuleList([DecoderLayer(D, heads, drop) for _ in range(dec_layers)])
        self.qnorm = nn.LayerNorm(D)
        self.qW = nn.Parameter(torch.randn(n_lab, D) * 0.02)
        self.qb = nn.Parameter(torch.zeros(n_lab))
        self.wnorm = nn.LayerNorm(D)
        self.att_v, self.att_u = nn.Linear(D, 256), nn.Linear(D, 256)
        self.att_w = nn.Linear(256, n_lab)
        self.mW = nn.Parameter(torch.randn(n_lab, D) * 0.02)
        self.mb = nn.Parameter(torch.zeros(n_lab))
        self.bb_chunk = 32   # the backbone runs on fixed chunks of this many windows

    def set_att_temp(self, t):
        for m in self.modules():
            if hasattr(m, "att_temp"):
                m.att_temp = float(t)

    def _features(self, x):
        """Backbone in fixed-size chunks (a few input shapes instead of one per study); the last chunk is padded with
        repeated windows whose outputs are dropped."""
        c, outs = self.bb_chunk, []
        for i in range(0, len(x), c):
            xi = x[i:i + c]
            outs.append(self.bb.forward_intermediates(_pad_rows(xi, c), indices=1, intermediates_only=True)[-1][:len(xi)])
        return torch.cat(outs)

    def forward(self, x, wmeta, mask, theta):
        """x (B, W, 3, H, W) normalised windows; wmeta (B, W, 2) [view slot, slice offset mm]; mask (B, W) bool;
        theta (B, W, 2, 3) the affine that produced each window from its cached slice, used to place every token at
        its anatomical position."""
        B, Wn = mask.shape
        t = self.p4(self._features(x[mask]))
        n, D, gh, gw = t.shape
        t = self.tnorm(t.flatten(2).transpose(1, 2))                       # (n, G, D)
        slot = wmeta[..., 0].long()[mask]                                  # (n,)
        plane = self.plane_of_slot[slot]
        emb = self.slot_emb(slot)[:, None]                                 # (n, 1, D)
        ys = (torch.arange(gh, device=x.device) + 0.5) / gh * 2 - 1
        xs = (torch.arange(gw, device=x.device) + 0.5) / gw * 2 - 1
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        g = torch.stack([gx.flatten(), gy.flatten(), torch.ones_like(gx.flatten())], -1)   # (G, 3)
        src = torch.einsum("nij,gj->ngi", theta[mask].float(), g)          # (n, G, 2) position in the cached slice
        w = wmeta[..., 1][mask].float()[:, None].expand(-1, src.shape[1])
        c = knee_coords(src[..., 0] * HALF_FOV_MM, src[..., 1] * HALF_FOV_MM, w, plane[:, None].expand_as(w))
        pos = self.pe(c).to(t.dtype) + emb.to(t.dtype)                     # (n, G, D)
        G = t.shape[1]
        tok = t.new_zeros(B, Wn, G, D); tok[mask] = t
        ps = t.new_zeros(B, Wn, G, D); ps[mask] = pos
        # label queries over every token of every window
        tm = mask[:, :, None].expand(B, Wn, G).reshape(B, Wn * G)
        q = self.queries[None].expand(B, -1, -1).to(t.dtype)
        tk, pk = tok.reshape(B, Wn * G, D), ps.reshape(B, Wn * G, D)
        for layer in self.dec:
            q, r = layer(q, tk, pk, tm)
        r = self.qnorm(r.view(B, self.n_lab, self.q_per, D).mean(2)).float()
        logit_q = (r * self.qW).sum(-1) + self.qb
        # gated attention-MIL over window vectors
        logit_m = self._mil(tok.mean(2) + ps.mean(2), mask)
        return (logit_q + logit_m) / 2

    def _mil(self, hw, mask):
        """Gated attention-MIL over window vectors hw (B, W, D) -> (B, n_lab) logits."""
        h = self.wnorm(hw)
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).float() * self.att_temp
        a = torch.softmax(a.masked_fill(~mask[..., None], -1e4), 1)        # (B, W, n_lab)
        pooled = torch.einsum("bwn,bwd->bnd", a, h.float())
        return (pooled * self.mW).sum(-1) + self.mb
