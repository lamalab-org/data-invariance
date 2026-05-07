"""Manim animation: cross-sample prediction churn (main paper idea).

Render preview (480p, ~30 s):
    /opt/miniconda3/bin/manim -pql paper/animations/main_idea.py MainIdea

Render high-quality (1080p):
    /opt/miniconda3/bin/manim -pqh paper/animations/main_idea.py MainIdea

Storyboard:
    1. Two retrainings on the same training set (bootstrap A, bootstrap B)
    2. Two networks reach near-identical aggregate accuracy
    3. ...but disagree on individual molecules ("churn")
    4. Heatmaps: ERM noisy stripes vs. twin-bootstrap clean stripes

The animation deliberately mirrors paper/figures/fig0_overview.pdf so
that figure and animation share the same visual language.
"""
from __future__ import annotations

import random

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Arrow,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    MathTex,
    Rectangle,
    Scene,
    Tex,
    Text,
    VGroup,
    Write,
    config,
)

# --- palette matched to fig0_overview / fig5_overlap -----------------------
CLASS0 = "#4F7CB6"  # blue
CLASS1 = "#E5824D"  # orange
INK = "#222222"
MUTED = "#9A9A9A"
ACCENT = "#C0392B"
GREEN = "#1F8A4C"

config.background_color = "#FAFAFA"


def _txt(s: str, size: float = 36, color: str = INK, weight: str = "NORMAL") -> Text:
    return Text(s, font_size=size, color=color, weight=weight)


def _math(s: str, size: float = 36, color: str = INK) -> MathTex:
    return MathTex(s, font_size=size, color=color)


def _dot_grid(n: int, cols: int, dot_r: float, gap: float,
              fill_pattern: list[int] | None = None) -> VGroup:
    """Grid of `n` dots in `cols` columns; optional fill_pattern marks
    which dots are coloured (0 = blue, 1 = orange, -1 = grey/missing)."""
    rows = (n + cols - 1) // cols
    grid = VGroup()
    for k in range(n):
        i, j = k // cols, k % cols
        if fill_pattern is None:
            color, opacity = MUTED, 0.7
        else:
            v = fill_pattern[k]
            if v == -1:
                color, opacity = MUTED, 0.10
            else:
                color = CLASS0 if v == 0 else CLASS1
                opacity = 0.85
        c = Circle(radius=dot_r, color=color, fill_color=color,
                   fill_opacity=opacity, stroke_width=0)
        c.move_to([(j - (cols - 1) / 2) * gap,
                   ((rows - 1) / 2 - i) * gap, 0])
        grid.add(c)
    return grid


def _bootstrap_pattern(n: int, seed: int) -> list[int]:
    """Produce a -1/0/1 pattern: -1 = molecule absent from this bootstrap,
    0/1 = its class.  Two bootstraps from the same canonical set share
    ~63% of indices (with replacement); ~37% are missing per bootstrap."""
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
    """Heatmap of `rows` molecules x `cols` retrainings."""
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
            color = CLASS0 if cls == 0 else CLASS1
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


class MainIdea(Scene):
    """Single-shot animation; ~50 s at 30 fps."""

    def construct(self):
        self.s1_title()
        self.clear_scene()
        self.s2_two_retrainings()
        self.clear_scene()
        self.s3_aggregate_vs_individual()
        self.clear_scene()
        self.s4_heatmaps()
        self.clear_scene()
        self.s5_end_card()

    def clear_scene(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)

    # -- 1. Title --------------------------------------------------------
    def s1_title(self):
        title = _txt("Two retrainings on the same data", size=48)
        sub = _txt("...same accuracy, different predictions",
                   size=28, color=MUTED).next_to(title, DOWN, buff=0.4)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.7)
        self.wait(1.4)

    # -- 2. Two bootstraps -> two networks -------------------------------
    def s2_two_retrainings(self):
        n, cols, dot_r, gap = 24, 8, 0.18, 0.45
        full = _dot_grid(n, cols, dot_r, gap)
        full.move_to(UP * 0.6)
        cap = _txt("Training set", size=28, color=MUTED
                   ).next_to(full, DOWN, buff=0.3)
        self.play(FadeIn(full, shift=UP * 0.2), Write(cap), run_time=1.0)
        self.wait(0.6)

        # Two bootstraps side by side
        left = _dot_grid(n, cols, dot_r, gap,
                         fill_pattern=_bootstrap_pattern(n, seed=1))
        right = _dot_grid(n, cols, dot_r, gap,
                          fill_pattern=_bootstrap_pattern(n, seed=2))
        left.move_to(LEFT * 3.4 + UP * 0.6)
        right.move_to(RIGHT * 3.4 + UP * 0.6)
        a_lab = _txt("Bootstrap A", size=24).next_to(left, UP, buff=0.3)
        b_lab = _txt("Bootstrap B", size=24).next_to(right, UP, buff=0.3)

        self.play(
            FadeOut(full, shift=DOWN * 0.2),
            FadeOut(cap),
            FadeIn(left, shift=LEFT * 0.4),
            FadeIn(right, shift=RIGHT * 0.4),
            Write(a_lab), Write(b_lab),
            run_time=1.3,
        )
        self.wait(0.6)

        # Two networks below
        net_a = Rectangle(width=1.6, height=0.9, color=INK,
                          fill_color="#FFFFFF", fill_opacity=1.0,
                          stroke_width=1.5
                          ).move_to(LEFT * 3.4 + DOWN * 1.6)
        net_b = Rectangle(width=1.6, height=0.9, color=INK,
                          fill_color="#FFFFFF", fill_opacity=1.0,
                          stroke_width=1.5
                          ).move_to(RIGHT * 3.4 + DOWN * 1.6)
        net_a_lab = _txt("Net A", size=22).move_to(net_a)
        net_b_lab = _txt("Net B", size=22).move_to(net_b)
        arr_a = Arrow(left.get_bottom(), net_a.get_top(),
                      buff=0.15, color=MUTED, stroke_width=2)
        arr_b = Arrow(right.get_bottom(), net_b.get_top(),
                      buff=0.15, color=MUTED, stroke_width=2)

        self.play(Create(arr_a), Create(arr_b),
                  FadeIn(net_a), FadeIn(net_b),
                  Write(net_a_lab), Write(net_b_lab),
                  run_time=1.0)
        self.wait(0.5)

        # Aggregate accuracy lines below the boxes.
        acc_a = _math(r"\text{acc}_A = 0.811", size=28
                      ).next_to(net_a, DOWN, buff=0.35)
        acc_b = _math(r"\text{acc}_B = 0.813", size=28
                      ).next_to(net_b, DOWN, buff=0.35)
        delta = _math(r"|\Delta\text{acc}| = 0.2\,\text{pp}",
                      size=30, color=GREEN).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(acc_a, shift=UP * 0.1),
                  FadeIn(acc_b, shift=UP * 0.1), run_time=0.8)
        self.play(Write(delta), run_time=0.7)
        self.wait(1.6)

    # -- 3. Aggregate match, individual disagreement ---------------------
    def s3_aggregate_vs_individual(self):
        header = _txt("Same accuracy.  Per-molecule predictions disagree.",
                      size=32).to_edge(UP, buff=0.6)
        self.play(Write(header), run_time=1.0)
        self.wait(0.4)

        n_mol = 12
        cell = 0.7
        x0 = -((n_mol - 1) / 2) * cell

        rng = random.Random(7)
        preds_a = [rng.randint(0, 1) for _ in range(n_mol)]
        preds_b = preds_a.copy()
        flip_idx = sorted(rng.sample(range(n_mol), 3))
        for i in flip_idx:
            preds_b[i] = 1 - preds_a[i]

        def _row(preds, y):
            row = VGroup()
            for i, p in enumerate(preds):
                color = CLASS0 if p == 0 else CLASS1
                sq = Rectangle(width=0.55, height=0.55, stroke_width=0,
                               fill_color=color, fill_opacity=1.0)
                sq.move_to([x0 + i * cell, y, 0])
                row.add(sq)
            return row

        row_a = _row(preds_a, 1.0)
        row_b = _row(preds_b, -0.4)
        a_tag = _txt("Net A", size=24).next_to(row_a, LEFT, buff=0.45)
        b_tag = _txt("Net B", size=24).next_to(row_b, LEFT, buff=0.45)
        molecules_lab = _txt("12 test molecules", size=22, color=MUTED
                             ).next_to(row_a, UP, buff=0.45)

        self.play(Write(molecules_lab), run_time=0.5)
        self.play(FadeIn(row_a, shift=DOWN * 0.2), Write(a_tag),
                  run_time=0.7)
        self.play(FadeIn(row_b, shift=UP * 0.2), Write(b_tag),
                  run_time=0.7)
        self.wait(0.4)

        rings = VGroup()
        for i in flip_idx:
            r1 = Circle(radius=0.36, color=ACCENT, stroke_width=3
                        ).move_to(row_a[i].get_center())
            r2 = Circle(radius=0.36, color=ACCENT, stroke_width=3
                        ).move_to(row_b[i].get_center())
            rings.add(r1, r2)
        churn_text = _math(r"\text{churn} = 3 / 12 = 25\%", size=34,
                           color=ACCENT).move_to(DOWN * 1.7)
        self.play(*[GrowFromCenter(r) for r in rings], run_time=0.7)
        self.play(Write(churn_text), run_time=0.7)
        self.wait(1.5)

        magnitudes = _txt(
            "Across 8 chemistry datasets:  8–22% per-molecule churn  "
            "vs.  1–4 pp accuracy difference",
            size=24, color=INK,
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(magnitudes), run_time=1.0)
        self.wait(2.0)

    # -- 4. Heatmaps ERM vs Twin-bootstrap -------------------------------
    def s4_heatmaps(self):
        rows, cols = 40, 10
        h, w = 4.4, 2.6
        rng = random.Random(11)
        consensus = [rng.randint(0, 1) for _ in range(rows)]
        erm = _heatmap(rows, cols, w, h, churn_p=0.22, seed=13,
                       base_run_class=consensus)
        twin = _heatmap(rows, cols, w, h, churn_p=0.04, seed=17,
                        base_run_class=consensus)
        erm.move_to(LEFT * 2.4 + DOWN * 0.2)
        twin.move_to(RIGHT * 2.4 + DOWN * 0.2)

        erm_title = _txt("ERM", size=30).next_to(erm, UP, buff=0.3)
        twin_title = Tex(r"Twin-bootstrap (\(\lambda{=}300\))",
                         font_size=30, color=INK
                         ).next_to(twin, UP, buff=0.3)
        x_axis = _txt("retraining  1—10", size=20, color=MUTED
                      ).next_to(erm, DOWN, buff=0.35)
        x_axis2 = _txt("retraining  1—10", size=20, color=MUTED
                       ).next_to(twin, DOWN, buff=0.35)

        # ERM first, with a red caption underneath
        self.play(Write(erm_title), FadeIn(erm), Write(x_axis), run_time=1.0)
        cap_erm = _txt("noisy: ~16% of test molecules flip class",
                       size=22, color=ACCENT
                       ).next_to(x_axis, DOWN, buff=0.45)
        self.play(Write(cap_erm), run_time=0.6)
        self.wait(1.6)

        # Twin-bootstrap second, with a green caption.
        self.play(Write(twin_title), FadeIn(twin), Write(x_axis2),
                  run_time=1.0)
        cap_twin = _txt("clean: ~6% flip rate, same accuracy",
                        size=22, color=GREEN
                        ).next_to(x_axis2, DOWN, buff=0.45)
        self.play(Write(cap_twin), FadeOut(cap_erm), run_time=0.7)
        self.wait(2.0)

    # -- 5. End card -----------------------------------------------------
    def s5_end_card(self):
        title = _txt("Reducing cross-sample prediction churn",
                     size=44, weight="BOLD")
        sub = _txt("in scientific machine learning", size=32, color=MUTED
                   ).next_to(title, DOWN, buff=0.35)
        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.7)
        self.wait(2.0)


_ = ORIGIN  # keep manim import warm; ORIGIN unused but other layouts use it
