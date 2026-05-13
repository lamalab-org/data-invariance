"""Manim animation: cross-sample prediction churn (main paper idea).

Style: 3b1b-flavoured -- black background, smooth transforms, stable
spatial composition, math typography in proper LaTeX, large-font-then-
scaled-down for crisp kerning.

Render preview (480p, ~30 s):
    /opt/miniconda3/bin/manim -pql paper/animations/main_idea.py MainIdea

Render high-quality (1080p):
    /opt/miniconda3/bin/manim -pqh paper/animations/main_idea.py MainIdea

The animation deliberately mirrors paper/figures/fig0_overview.pdf so
that figure and animation share the same visual language.  Beats:

  1. Two retrainings of the same training set produce two bootstraps.
  2. Two networks reach near-identical aggregate accuracy.
  3. ...but disagree on individual molecules ("churn").
  4. Heatmaps: ERM noisy stripes vs. twin-bootstrap clean stripes.
  5. Single-line takeaway.
"""
from __future__ import annotations

import random

from manim import (
    BLACK,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    WHITE,
    Arrow,
    Circle,
    Create,
    DecimalNumber,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Indicate,
    LaggedStart,
    MathTex,
    Rectangle,
    ReplacementTransform,
    Scene,
    Tex,
    Transform,
    VGroup,
    Write,
    config,
)

# --- 3b1b-style palette ---------------------------------------------------
BLUE = "#58C4DD"     # class 0
GOLD = "#F0AC5F"     # class 1
RED = "#FC6255"      # accent / disagreement
GREEN = "#83C167"    # accent / improvement
GREY = "#888888"     # muted
PAPER = WHITE        # main text on black background

config.background_color = BLACK


# --- typography helpers ---------------------------------------------------
# 3b1b's signature look is LaTeX-rendered Computer Modern Serif.  Tex
# and MathTex both go through LaTeX, so use them everywhere -- no
# Pango / system fonts, no mixed serif / sans-serif on the same frame.
# Render every label at font_size=72 then .scale(target/72) so the
# rasterizer always works at a high resolution and kerning is clean.
_BIG = 72.0


def _tex(s: str, size: float = 36, color: str = PAPER) -> Tex:
    """Plain text via LaTeX (Computer Modern Serif).  `s` is LaTeX
    source: literal % must be ``\\%``, em-dash is ``--`` etc."""
    t = Tex(s, font_size=_BIG, color=color)
    t.scale(size / _BIG)
    return t


def _bold(s: str, size: float = 36, color: str = PAPER) -> Tex:
    return _tex(rf"\textbf{{{s}}}", size=size, color=color)


def _math(s: str, size: float = 36, color: str = PAPER) -> MathTex:
    m = MathTex(s, font_size=_BIG, color=color)
    m.scale(size / _BIG)
    return m


# --- visual helpers -------------------------------------------------------
def _dot_grid(n: int, cols: int, dot_r: float, gap: float,
              fill_pattern: list[int] | None = None) -> VGroup:
    rows = (n + cols - 1) // cols
    grid = VGroup()
    for k in range(n):
        i, j = k // cols, k % cols
        if fill_pattern is None:
            color, opacity = GREY, 0.55
        else:
            v = fill_pattern[k]
            if v == -1:
                color, opacity = GREY, 0.10
            else:
                color = BLUE if v == 0 else GOLD
                opacity = 0.95
        c = Circle(radius=dot_r, color=color, fill_color=color,
                   fill_opacity=opacity, stroke_width=0)
        c.move_to([(j - (cols - 1) / 2) * gap,
                   ((rows - 1) / 2 - i) * gap, 0])
        grid.add(c)
    return grid


def _bootstrap_pattern(n: int, seed: int) -> list[int]:
    """Two bootstraps share ~63% of indices on expectation; 37% missing.

    -1 = molecule absent from this bootstrap
    0/1 = its class (alternating along k for a clean alternating look).
    """
    rng = random.Random(seed)
    classes = [k % 2 for k in range(n)]
    pattern = []
    for k in range(n):
        if rng.random() < 0.37:
            pattern.append(-1)
        else:
            pattern.append(classes[k])
    return pattern


def _heatmap(rows: int, cols: int, w: float, h: float, churn_p: float,
             seed: int, base_run_class: list[int] | None = None) -> VGroup:
    rng = random.Random(seed)
    cell_w = w / cols
    cell_h = h / rows
    cells = VGroup()
    base_run_class = base_run_class or [
        rng.randint(0, 1) for _ in range(rows)
    ]
    for i in range(rows):
        consensus = base_run_class[i]
        for j in range(cols):
            flipped = rng.random() < churn_p
            cls = 1 - consensus if flipped else consensus
            color = BLUE if cls == 0 else GOLD
            r = Rectangle(
                width=cell_w, height=cell_h,
                stroke_width=0, fill_color=color, fill_opacity=1.0,
            )
            r.move_to([
                -w / 2 + (j + 0.5) * cell_w,
                h / 2 - (i + 0.5) * cell_h,
                0,
            ])
            cells.add(r)
    return cells


# --- the scene -------------------------------------------------------------
class MainIdea(Scene):
    """3b1b-style ~50 s animation; black background, no title card."""

    def construct(self):
        self.beat1_two_bootstraps()
        self.beat2_networks_and_accuracy()
        self.beat3_per_molecule_disagreement()
        self.beat4_magnitude()
        self.beat5_heatmaps()
        self.beat6_takeaway()

    # -- 1. Two bootstraps from the same training set --------------------
    def beat1_two_bootstraps(self):
        n, cols, dot_r, gap = 24, 8, 0.18, 0.45
        full = _dot_grid(n, cols, dot_r, gap)
        full.move_to(ORIGIN)
        cap = _tex("Same training set", size=28, color=GREY
                   ).next_to(full, DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(d, scale=0.6) for d in full],
                              lag_ratio=0.04, run_time=1.6),
                  Write(cap), run_time=1.6)
        self.wait(0.6)

        # Build bootstrap A (left) and bootstrap B (right) by transforming
        # the original grid into its left target and a copy into the right
        # target; this keeps the screen continuous (no fade-and-redraw).
        left_target = _dot_grid(n, cols, dot_r, gap,
                                fill_pattern=_bootstrap_pattern(n, seed=1))
        right_target = _dot_grid(n, cols, dot_r, gap,
                                 fill_pattern=_bootstrap_pattern(n, seed=2))
        left_target.move_to(LEFT * 3.4 + UP * 0.7)
        right_target.move_to(RIGHT * 3.4 + UP * 0.7)

        copy = full.copy()
        a_lab = _tex("Bootstrap A", size=26).next_to(left_target, UP, buff=0.3)
        b_lab = _tex("Bootstrap B", size=26).next_to(right_target, UP, buff=0.3)

        self.play(
            FadeOut(cap),
            Transform(full, left_target),
            Transform(copy, right_target),
            run_time=1.5,
        )
        self.play(Write(a_lab), Write(b_lab), run_time=0.7)
        self.wait(0.6)

        # Persist for next beat.
        self.boot_left = full
        self.boot_right = copy
        self.a_lab = a_lab
        self.b_lab = b_lab

    # -- 2. Two networks reach near-identical aggregate accuracy ---------
    def beat2_networks_and_accuracy(self):
        net_a = Rectangle(width=1.6, height=0.85, color=PAPER,
                          fill_color=BLACK, fill_opacity=1.0,
                          stroke_width=1.5
                          ).move_to(LEFT * 3.4 + DOWN * 1.4)
        net_b = Rectangle(width=1.6, height=0.85, color=PAPER,
                          fill_color=BLACK, fill_opacity=1.0,
                          stroke_width=1.5
                          ).move_to(RIGHT * 3.4 + DOWN * 1.4)
        net_a_lab = _tex("Net A", size=24).move_to(net_a)
        net_b_lab = _tex("Net B", size=24).move_to(net_b)
        arr_a = Arrow(self.boot_left.get_bottom(), net_a.get_top(),
                      buff=0.12, color=GREY, stroke_width=2)
        arr_b = Arrow(self.boot_right.get_bottom(), net_b.get_top(),
                      buff=0.12, color=GREY, stroke_width=2)
        self.play(Create(arr_a), Create(arr_b), run_time=0.8)
        self.play(FadeIn(net_a), FadeIn(net_b),
                  Write(net_a_lab), Write(net_b_lab), run_time=0.7)

        # Numerical accuracy ticking up: use DecimalNumber for the count-up
        # effect.  Pick example values that land inside the paper's quoted
        # range of 1--4 pp accuracy difference (acc_A = 0.805, acc_B =
        # 0.821 -> |Delta| = 1.6 pp).
        acc_a_label = _math(r"\text{acc}_A = ", size=30
                            ).next_to(net_a, DOWN, buff=0.4)
        acc_b_label = _math(r"\text{acc}_B = ", size=30
                            ).next_to(net_b, DOWN, buff=0.4)
        acc_a_num = DecimalNumber(0.0, num_decimal_places=3,
                                  font_size=_BIG, color=PAPER
                                  ).scale(30 / _BIG)
        acc_b_num = DecimalNumber(0.0, num_decimal_places=3,
                                  font_size=_BIG, color=PAPER
                                  ).scale(30 / _BIG)
        acc_a_num.next_to(acc_a_label, RIGHT, buff=0.12)
        acc_b_num.next_to(acc_b_label, RIGHT, buff=0.12)

        self.play(Write(acc_a_label), Write(acc_b_label),
                  FadeIn(acc_a_num), FadeIn(acc_b_num), run_time=0.5)
        self.play(
            acc_a_num.animate.set_value(0.805),
            acc_b_num.animate.set_value(0.821),
            run_time=1.4,
        )
        self.wait(0.4)

        delta = _math(r"|\Delta\text{acc}| = 1.6\,\text{pp}",
                      size=32, color=GREEN).to_edge(DOWN, buff=0.5)
        self.play(Write(delta), run_time=0.8)
        self.play(Indicate(delta, scale_factor=1.15, color=GREEN),
                  run_time=0.7)
        self.wait(0.8)

        self._beat2_group = VGroup(
            self.boot_left, self.boot_right, self.a_lab, self.b_lab,
            net_a, net_b, net_a_lab, net_b_lab, arr_a, arr_b,
            acc_a_label, acc_b_label, acc_a_num, acc_b_num, delta,
        )

    # -- 3. Per-molecule disagreement -----------------------------------
    def beat3_per_molecule_disagreement(self):
        # Sweep beat-2 contents off the screen; replace with a clean
        # Net A / Net B prediction comparison.
        self.play(FadeOut(self._beat2_group, shift=UP * 0.4), run_time=0.8)

        header = _tex("Aggregates match. Per-molecule predictions disagree.",
                      size=30).to_edge(UP, buff=0.7)
        self.play(Write(header), run_time=1.0)

        # 10 molecules, 2 disagreements -> 20% per-molecule churn, inside
        # the paper's 8--22% range.
        n_mol = 10
        cell = 0.78
        x0 = -((n_mol - 1) / 2) * cell

        rng = random.Random(7)
        preds_a = [rng.randint(0, 1) for _ in range(n_mol)]
        preds_b = preds_a.copy()
        flip_idx = sorted(rng.sample(range(n_mol), 2))
        for i in flip_idx:
            preds_b[i] = 1 - preds_a[i]

        def _row(preds, y):
            row = VGroup()
            for i, p in enumerate(preds):
                color = BLUE if p == 0 else GOLD
                sq = Rectangle(width=0.55, height=0.55, stroke_width=0,
                               fill_color=color, fill_opacity=1.0)
                sq.move_to([x0 + i * cell, y, 0])
                row.add(sq)
            return row

        row_a = _row(preds_a, 1.1)
        row_b = _row(preds_b, -0.5)
        a_tag = _tex("Net A", size=24).next_to(row_a, LEFT, buff=0.45)
        b_tag = _tex("Net B", size=24).next_to(row_b, LEFT, buff=0.45)
        molecules_lab = _tex("10 test molecules", size=22, color=GREY
                             ).next_to(row_a, UP, buff=0.4)

        self.play(
            LaggedStart(*[FadeIn(sq, scale=0.7) for sq in row_a],
                        lag_ratio=0.05, run_time=0.9),
            Write(a_tag), Write(molecules_lab),
            run_time=1.0,
        )
        self.play(
            LaggedStart(*[FadeIn(sq, scale=0.7) for sq in row_b],
                        lag_ratio=0.05, run_time=0.9),
            Write(b_tag),
            run_time=1.0,
        )
        self.wait(0.4)

        rings = VGroup(*[
            Circle(radius=0.36, color=RED, stroke_width=4
                   ).move_to(row_a[i].get_center())
            for i in flip_idx
        ] + [
            Circle(radius=0.36, color=RED, stroke_width=4
                   ).move_to(row_b[i].get_center())
            for i in flip_idx
        ])
        churn = _math(r"\text{churn} = \frac{2}{10} = 20\%",
                      size=36, color=RED).move_to(DOWN * 2.0)
        self.play(LaggedStart(*[GrowFromCenter(r) for r in rings],
                              lag_ratio=0.1, run_time=0.9))
        self.play(Write(churn), run_time=0.9)
        self.wait(1.6)

        self._beat3_group = VGroup(header, row_a, row_b, a_tag, b_tag,
                                   molecules_lab, rings, churn)

    # -- 4. Magnitude callout -------------------------------------------
    def beat4_magnitude(self):
        self.play(FadeOut(self._beat3_group, shift=UP * 0.3), run_time=0.7)
        line1 = _tex("Across 8 chemistry datasets",
                     size=32, color=GREY).move_to(UP * 0.5)
        line2 = _math(
            r"8\text{--}22\%\ \text{per-molecule churn}"
            r"\quad\text{vs.}\quad"
            r"1\text{--}4\,\text{pp accuracy difference}",
            size=34, color=PAPER,
        ).move_to(DOWN * 0.5)
        self.play(Write(line1), run_time=0.9)
        self.play(Write(line2), run_time=1.5)
        self.wait(2.2)
        self.play(FadeOut(line1), FadeOut(line2), run_time=0.5)

    # -- 5. Heatmaps: ERM noisy vs Twin-bootstrap clean -----------------
    def beat5_heatmaps(self):
        rows, cols = 40, 10
        h, w = 4.4, 2.6
        rng = random.Random(11)
        consensus = [rng.randint(0, 1) for _ in range(rows)]
        # Match the captions:
        # per-pair disagreement = 2*p*(1-p), so for 16% choose p=0.087,
        # for 6% choose p=0.031.  Using p_ERM=0.09 / p_twin=0.03 keeps the
        # 3x contrast while making the visual flip rate consistent with
        # the percentages quoted underneath the heatmaps.
        erm = _heatmap(rows, cols, w, h, churn_p=0.09, seed=13,
                       base_run_class=consensus)
        twin = _heatmap(rows, cols, w, h, churn_p=0.03, seed=17,
                        base_run_class=consensus)

        # ERM starts centred so the noisy texture reads on its own.
        erm.move_to(DOWN * 0.2)
        erm_title = _tex("ERM", size=32).next_to(erm, UP, buff=0.35)
        cap_erm = _tex(r"$\sim$16\% of molecules flip class",
                       size=24, color=RED
                       ).next_to(erm, DOWN, buff=0.45)
        self.play(Write(erm_title), FadeIn(erm), run_time=1.0)
        self.play(Write(cap_erm), run_time=0.7)
        self.wait(1.6)

        # Move ERM-cluster left as one unit; reveal twin on the right.
        erm_cluster = VGroup(erm, erm_title, cap_erm)
        twin.move_to(RIGHT * 2.4 + DOWN * 0.2)
        twin_title = _tex(r"Twin-bootstrap ($\lambda{=}300$)",
                          size=32).next_to(twin, UP, buff=0.35)
        cap_twin = _tex(r"$\sim$6\% flip rate, same accuracy",
                        size=24, color=GREEN
                        ).next_to(twin, DOWN, buff=0.45)
        self.play(
            erm_cluster.animate.shift(LEFT * 2.4),
            FadeIn(twin, shift=LEFT * 0.4),
            FadeIn(twin_title, shift=LEFT * 0.4),
            run_time=1.3,
        )
        self.play(Write(cap_twin), run_time=0.8)
        self.wait(2.2)

        self._beat5_group = VGroup(erm_cluster, twin, twin_title, cap_twin)

    # -- 6. Takeaway -----------------------------------------------------
    def beat6_takeaway(self):
        self.play(FadeOut(self._beat5_group), run_time=0.8)
        line = _bold("Reducing cross-sample prediction churn", size=42)
        sub = _tex("in scientific machine learning",
                   size=30, color=GREY
                   ).next_to(line, DOWN, buff=0.4)
        self.play(Write(line), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.15), run_time=0.7)
        self.wait(2.2)


_ = ReplacementTransform  # imported for future ablation; keep symbol live
