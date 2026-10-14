"""
╔══════════════════════════════════════════════════════════════════════╗
║         STRING SIMILARITY SEARCH SYSTEM — Final Year Project        ║
║         Built with CustomTkinter | Multi-Algorithm | Analytics      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import threading
import time
import math
import random
import csv
import os
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.metrics import precision_score, recall_score, f1_score

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─────────────────────────────────────────────
#  ALGORITHM ENGINE
# ─────────────────────────────────────────────

class SimilarityEngine:
    """Core engine implementing all similarity algorithms."""

    # ── N-Gram Similarity ──────────────────────────────────────────
    @staticmethod
    def ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
        def get_ngrams(s, n):
            s = s.lower().strip()
            return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else set()
        g1 = get_ngrams(s1, n)
        g2 = get_ngrams(s2, n)
        if not g1 and not g2:
            return 1.0
        if not g1 or not g2:
            return 0.0
        intersection = len(g1 & g2)
        union = len(g1 | g2)
        return intersection / union

    # ── Edit Distance (Levenshtein) ────────────────────────────────
    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        s1, s2 = s1.lower(), s2.lower()
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    @staticmethod
    def edit_distance_similarity(s1: str, s2: str) -> float:
        dist = SimilarityEngine.edit_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1 - dist / max_len

    # ── Cosine Similarity (manual TF-IDF) ─────────────────────────
    @staticmethod
    def cosine_similarity(s1: str, s2: str) -> float:
        def word_freq(s):
            freq = defaultdict(int)
            for w in s.lower().split():
                freq[w] += 1
            return freq
        f1, f2 = word_freq(s1), word_freq(s2)
        vocab = set(f1) | set(f2)
        if not vocab:
            return 1.0
        v1 = np.array([f1.get(w, 0) for w in vocab], dtype=float)
        v2 = np.array([f2.get(w, 0) for w in vocab], dtype=float)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    # ── TF-IDF Cosine (sklearn) ────────────────────────────────────
    @staticmethod
    def tfidf_similarity(query: str, corpus: list) -> list:
        if not corpus:
            return []
        all_docs = [query] + corpus
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
        try:
            tfidf_matrix = vectorizer.fit_transform(all_docs)
            sims = sk_cosine(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            return sims.tolist()
        except Exception:
            return [0.0] * len(corpus)

    # ── Jaccard Similarity ─────────────────────────────────────────
    @staticmethod
    def jaccard_similarity(s1: str, s2: str) -> float:
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    # ── Jaro-Winkler ──────────────────────────────────────────────
    @staticmethod
    def jaro_winkler(s1: str, s2: str) -> float:
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2:
            return 1.0
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        match_dist = max(len1, len2) // 2 - 1
        match_dist = max(0, match_dist)
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        transpositions = 0
        for i in range(len1):
            start = max(0, i - match_dist)
            end = min(i + match_dist + 1, len2)
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
        if matches == 0:
            return 0.0
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        jaro = (matches/len1 + matches/len2 + (matches - transpositions/2)/matches) / 3
        prefix = 0
        for i in range(min(4, len(s1), len(s2))):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        return jaro + prefix * 0.1 * (1 - jaro)

    # ── Soundex ───────────────────────────────────────────────────
    @staticmethod
    def soundex(s: str) -> str:
        s = s.upper()
        if not s:
            return "0000"
        codes = {'BFPV': '1', 'CGJKQSXYZ': '2', 'DT': '3',
                 'L': '4', 'MN': '5', 'R': '6'}
        result = s[0]
        prev = ''
        for char in s[1:]:
            for key, val in codes.items():
                if char in key:
                    if val != prev:
                        result += val
                        prev = val
                    break
            else:
                prev = ''
        return (result + '000')[:4]

    @staticmethod
    def soundex_similarity(s1: str, s2: str) -> float:
        return 1.0 if SimilarityEngine.soundex(s1) == SimilarityEngine.soundex(s2) else 0.0

    # ── Fuzzy (Ratio-based) ───────────────────────────────────────
    @staticmethod
    def fuzzy_ratio(s1: str, s2: str) -> float:
        """Simple token set ratio."""
        t1 = set(s1.lower().split())
        t2 = set(s2.lower().split())
        common = t1 & t2
        if not t1 and not t2:
            return 1.0
        if not t1 or not t2:
            return 0.0
        # Weighted: common tokens + edit distance on full string
        token_score = 2 * len(common) / (len(t1) + len(t2))
        edit_score = SimilarityEngine.edit_distance_similarity(s1, s2)
        return (token_score + edit_score) / 2

    # ── Top-K Search ──────────────────────────────────────────────
    @staticmethod
    def top_k_search(query: str, dataset: list, algorithm: str,
                     k: int = 10, ngram_n: int = 2) -> list:
        """Returns top-k results sorted by similarity (descending)."""
        results = []
        tfidf_scores = None

        if algorithm == "TF-IDF Cosine":
            tfidf_scores = SimilarityEngine.tfidf_similarity(query, dataset)

        for idx, doc in enumerate(dataset):
            if algorithm == "N-Gram":
                score = SimilarityEngine.ngram_similarity(query, doc, ngram_n)
            elif algorithm == "Edit Distance":
                score = SimilarityEngine.edit_distance_similarity(query, doc)
            elif algorithm == "Cosine":
                score = SimilarityEngine.cosine_similarity(query, doc)
            elif algorithm == "TF-IDF Cosine":
                score = tfidf_scores[idx] if tfidf_scores else 0.0
            elif algorithm == "Jaccard":
                score = SimilarityEngine.jaccard_similarity(query, doc)
            elif algorithm == "Jaro-Winkler":
                score = SimilarityEngine.jaro_winkler(query, doc)
            elif algorithm == "Soundex":
                score = SimilarityEngine.soundex_similarity(query, doc)
            elif algorithm == "Fuzzy":
                score = SimilarityEngine.fuzzy_ratio(query, doc)
            else:
                score = 0.0
            results.append((doc, score, idx))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# ─────────────────────────────────────────────
#  METRICS ENGINE
# ─────────────────────────────────────────────

class MetricsEngine:
    @staticmethod
    def compute_metrics(results: list, relevant_threshold: float = 0.5) -> dict:
        scores = [r[1] for r in results]
        if not scores:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
                    "avg_score": 0, "max_score": 0, "min_score": 0}
        predicted = [1 if s >= relevant_threshold else 0 for s in scores]
        true_positive = sum(predicted)
        total = len(predicted)
        avg = np.mean(scores)
        # Simulated ground truth: top 30% are considered relevant
        top_n = max(1, int(total * 0.3))
        ground_truth = [1 if i < top_n else 0 for i in range(total)]
        precision = sum(p and g for p, g in zip(predicted, ground_truth)) / max(sum(predicted), 1)
        recall = sum(p and g for p, g in zip(predicted, ground_truth)) / max(sum(ground_truth), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = sum(p == g for p, g in zip(predicted, ground_truth)) / total
        return {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "avg_score": round(avg * 100, 2),
            "max_score": round(max(scores) * 100, 2),
            "min_score": round(min(scores) * 100, 2),
        }


# ─────────────────────────────────────────────
#  PDF REPORT GENERATOR
# ─────────────────────────────────────────────

class PDFReporter:
    @staticmethod
    def generate(filepath: str, query: str, algorithm: str,
                 results: list, metrics: dict, chart_path: str = None):
        doc = SimpleDocTemplate(filepath, pagesize=A4,
                                leftMargin=0.8*inch, rightMargin=0.8*inch,
                                topMargin=0.8*inch, bottomMargin=0.8*inch)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle('Title', fontSize=20, spaceAfter=6,
                                      textColor=colors.HexColor('#1a1a2e'),
                                      alignment=TA_CENTER, fontName='Helvetica-Bold')
        story.append(Paragraph("String Similarity Search System", title_style))
        story.append(Paragraph("Analysis Report", ParagraphStyle('Sub', fontSize=13,
                                alignment=TA_CENTER, textColor=colors.HexColor('#16213e'),
                                spaceAfter=4, fontName='Helvetica')))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                ParagraphStyle('Date', fontSize=9, alignment=TA_CENTER,
                                               textColor=colors.grey, spaceAfter=16)))
        story.append(Spacer(1, 0.2*inch))

        # Search Info
        info_data = [
            ['Query String', query],
            ['Algorithm Used', algorithm],
            ['Results Retrieved', str(len(results))],
        ]
        info_table = Table(info_data, colWidths=[2*inch, 4.5*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0,0), (0,-1), colors.white),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BACKGROUND', (1,0), (1,-1), colors.HexColor('#f0f4ff')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#ccccdd')),
            ('ROWBACKGROUNDS', (1,0), (1,-1), [colors.HexColor('#f0f4ff'), colors.white]),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.2*inch))

        # Metrics
        story.append(Paragraph("Performance Metrics", ParagraphStyle('H2', fontSize=13,
                                fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=6,
                                textColor=colors.HexColor('#1a1a2e'))))
        metric_data = [['Metric', 'Value'],
                       ['Accuracy', f"{metrics.get('accuracy',0):.2f}%"],
                       ['Precision', f"{metrics.get('precision',0):.2f}%"],
                       ['Recall', f"{metrics.get('recall',0):.2f}%"],
                       ['F1 Score', f"{metrics.get('f1',0):.2f}%"],
                       ['Avg Similarity', f"{metrics.get('avg_score',0):.2f}%"],
                       ['Max Similarity', f"{metrics.get('max_score',0):.2f}%"]]
        m_table = Table(metric_data, colWidths=[3.25*inch, 3.25*inch])
        m_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#ccccdd')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7f9ff')]),
            ('ALIGN', (1,0), (1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(m_table)
        story.append(Spacer(1, 0.2*inch))

        # Chart
        if chart_path and os.path.exists(chart_path):
            story.append(Paragraph("Similarity Score Chart", ParagraphStyle('H2', fontSize=13,
                                    fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=6,
                                    textColor=colors.HexColor('#1a1a2e'))))
            img = RLImage(chart_path, width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.15*inch))

        # Results Table
        story.append(Paragraph("Top Search Results", ParagraphStyle('H2', fontSize=13,
                                fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=6,
                                textColor=colors.HexColor('#1a1a2e'))))
        res_data = [['#', 'Document', 'Similarity Score']]
        for i, (doc, score, _) in enumerate(results[:20], 1):
            truncated = doc[:70] + '...' if len(doc) > 70 else doc
            res_data.append([str(i), truncated, f"{score*100:.2f}%"])
        r_table = Table(res_data, colWidths=[0.4*inch, 5.2*inch, 1.1*inch])
        r_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#ddddee')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7f9ff')]),
            ('ALIGN', (0,0), (0,-1), 'CENTER'),
            ('ALIGN', (2,0), (2,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 5),
            ('WORDWRAP', (1,1), (1,-1), True),
        ]))
        story.append(r_table)
        doc.build(story)


# ─────────────────────────────────────────────
#  DEFAULT DATASET
# ─────────────────────────────────────────────

DEFAULT_DATASET = [
    "machine learning algorithms for data classification",
    "deep learning neural network architectures",
    "natural language processing and text mining",
    "computer vision object detection methods",
    "reinforcement learning reward optimization",
    "support vector machine classification techniques",
    "random forest ensemble learning method",
    "gradient boosting and XGBoost algorithms",
    "k-nearest neighbor clustering approach",
    "principal component analysis dimensionality reduction",
    "convolutional neural network image recognition",
    "recurrent neural network sequence modeling",
    "transformer models attention mechanism BERT",
    "generative adversarial network image synthesis",
    "variational autoencoder latent space representation",
    "decision tree classification regression tree",
    "logistic regression binary classification",
    "linear regression prediction model",
    "naive bayes probabilistic classifier",
    "fuzzy logic membership functions",
    "string similarity algorithms comparison",
    "edit distance levenshtein metric",
    "cosine similarity text documents",
    "jaccard similarity set operations",
    "n-gram language model text",
    "information retrieval search engine ranking",
    "TF-IDF term frequency inverse document",
    "word embeddings word2vec GloVe",
    "semantic similarity sentence transformers",
    "spell checking correction autocomplete",
    "database indexing query optimization",
    "software testing unit test coverage",
    "agile scrum project management methodology",
    "cloud computing distributed systems AWS",
    "cybersecurity threat detection firewall",
    "blockchain decentralized ledger technology",
    "internet of things IoT sensor data",
    "data visualization dashboard analytics",
    "recommendation system collaborative filtering",
    "anomaly detection fraud prevention",
    "time series forecasting ARIMA LSTM",
    "sentiment analysis opinion mining reviews",
    "topic modeling LDA latent Dirichlet",
    "named entity recognition information extraction",
    "question answering reading comprehension",
    "machine translation neural sequence to sequence",
    "speech recognition acoustic model",
    "image segmentation pixel classification",
    "face recognition biometric identification",
    "autonomous vehicle self-driving perception",
]


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class StringSimilarityApp(ctk.CTk):
    ALGORITHMS = ["N-Gram", "Edit Distance", "Cosine", "TF-IDF Cosine",
                  "Jaccard", "Jaro-Winkler", "Soundex", "Fuzzy"]
    PALETTE = {
        "bg_dark": "#0d0d1a",
        "bg_mid": "#12122a",
        "bg_card": "#1a1a3e",
        "accent1": "#6c63ff",
        "accent2": "#00d4ff",
        "accent3": "#ff6584",
        "text": "#e8e8ff",
        "text_dim": "#9090bb",
        "success": "#00e676",
        "warning": "#ffab40",
    }

    def __init__(self):
        super().__init__()
        self.title("String Similarity Search System")
        self.geometry("1400x860")
        self.minsize(1100, 720)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=self.PALETTE["bg_dark"])

        self.dataset = list(DEFAULT_DATASET)
        self.last_results = []
        self.comparison_data = {}
        self.engine = SimilarityEngine()
        self.metrics_engine = MetricsEngine()
        self._particles = []
        self._anim_running = True
        self._search_running = False

        self._build_ui()
        self._start_particle_animation()

    # ── UI BUILD ──────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=self.PALETTE["bg_mid"], height=72, corner_radius=0)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="⬡", font=("Segoe UI", 28), 
                     text_color=self.PALETTE["accent1"]).pack(side="left", padx=(20,6), pady=10)
        ctk.CTkLabel(header, text="STRING SIMILARITY SEARCH SYSTEM",
                     font=("Segoe UI Black", 18, "bold"),
                     text_color=self.PALETTE["text"]).pack(side="left", pady=10)
        ctk.CTkLabel(header, text="  Final Year Project — Multi-Algorithm Analytics Platform",
                     font=("Segoe UI", 11), text_color=self.PALETTE["text_dim"]).pack(side="left", pady=10)

        # Status indicator
        self.status_dot = ctk.CTkLabel(header, text="● READY", font=("Consolas", 11),
                                        text_color=self.PALETTE["success"])
        self.status_dot.pack(side="right", padx=20)

        # ── Main layout ──────────────────────────────────────────
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=0, pady=0)

        # Left sidebar
        self._build_sidebar(main)

        # Right content area (tabbed)
        content = ctk.CTkFrame(main, fg_color=self.PALETTE["bg_dark"], corner_radius=0)
        content.pack(side="left", fill="both", expand=True)

        self.tabview = ctk.CTkTabview(content, fg_color=self.PALETTE["bg_mid"],
                                       segmented_button_fg_color=self.PALETTE["bg_card"],
                                       segmented_button_selected_color=self.PALETTE["accent1"],
                                       text_color=self.PALETTE["text"])
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tabview.add("🔍  Search Results")
        self.tabview.add("📊  Algorithm Comparison")
        self.tabview.add("📈  Analytics Dashboard")
        self.tabview.add("📁  Dataset Manager")

        self._build_results_tab()
        self._build_comparison_tab()
        self._build_analytics_tab()
        self._build_dataset_tab()

    def _build_sidebar(self, parent):
        sidebar = ctk.CTkFrame(parent, fg_color=self.PALETTE["bg_mid"],
                                width=320, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Animated canvas background
        self.canvas_bg = tk.Canvas(sidebar, width=320, height=120,
                                    bg=self.PALETTE["bg_mid"], highlightthickness=0)
        self.canvas_bg.pack(fill="x")
        self._init_particles()

        # ── Query input ──────────────────────────────────────────
        self._section(sidebar, "SEARCH QUERY")
        self.query_text = ctk.CTkTextbox(sidebar, height=80, fg_color=self.PALETTE["bg_card"],
                                          text_color=self.PALETTE["text"],
                                          border_color=self.PALETTE["accent1"], border_width=1,
                                          font=("Consolas", 12))
        self.query_text.pack(fill="x", padx=14, pady=(0, 10))
        self.query_text.insert("end", "machine learning")

        # ── Algorithm ────────────────────────────────────────────
        self._section(sidebar, "ALGORITHM")
        self.algo_var = ctk.StringVar(value="Edit Distance")
        self.algo_menu = ctk.CTkOptionMenu(sidebar, variable=self.algo_var,
                                            values=self.ALGORITHMS,
                                            fg_color=self.PALETTE["bg_card"],
                                            button_color=self.PALETTE["accent1"],
                                            text_color=self.PALETTE["text"],
                                            font=("Segoe UI", 12))
        self.algo_menu.pack(fill="x", padx=14, pady=(0, 10))

        # N-Gram slider
        self.ngram_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        ctk.CTkLabel(self.ngram_frame, text="N-Gram Size",
                     text_color=self.PALETTE["text_dim"], font=("Segoe UI", 10)).pack(anchor="w")
        self.ngram_n = ctk.CTkSlider(self.ngram_frame, from_=1, to=5, number_of_steps=4,
                                      button_color=self.PALETTE["accent1"],
                                      progress_color=self.PALETTE["accent1"])
        self.ngram_n.set(2)
        self.ngram_n.pack(fill="x")
        self.ngram_label = ctk.CTkLabel(self.ngram_frame, text="n = 2",
                                         text_color=self.PALETTE["accent2"], font=("Consolas", 11))
        self.ngram_label.pack()
        self.ngram_n.configure(command=lambda v: self.ngram_label.configure(text=f"n = {int(v)}"))
        self.ngram_frame.pack(fill="x", padx=14, pady=(0, 8))

        self.algo_var.trace_add("write", self._on_algo_change)
        self._on_algo_change()

        # ── Top-K ───────────────────────────────────────────────
        self._section(sidebar, "TOP-K RESULTS")
        k_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        k_frame.pack(fill="x", padx=14, pady=(0, 10))
        self.k_slider = ctk.CTkSlider(k_frame, from_=1, to=50, number_of_steps=49,
                                       button_color=self.PALETTE["accent2"],
                                       progress_color=self.PALETTE["accent2"])
        self.k_slider.set(10)
        self.k_slider.pack(fill="x")
        self.k_label = ctk.CTkLabel(k_frame, text="k = 10",
                                     text_color=self.PALETTE["accent2"], font=("Consolas", 11))
        self.k_label.pack()
        self.k_slider.configure(command=lambda v: self.k_label.configure(text=f"k = {int(v)}"))

        # ── Threshold ────────────────────────────────────────────
        self._section(sidebar, "RELEVANCE THRESHOLD")
        th_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        th_frame.pack(fill="x", padx=14, pady=(0, 14))
        self.threshold_slider = ctk.CTkSlider(th_frame, from_=0, to=1, number_of_steps=20,
                                               button_color=self.PALETTE["accent3"],
                                               progress_color=self.PALETTE["accent3"])
        self.threshold_slider.set(0.3)
        self.threshold_slider.pack(fill="x")
        self.threshold_label = ctk.CTkLabel(th_frame, text="threshold = 0.30",
                                             text_color=self.PALETTE["accent3"],
                                             font=("Consolas", 11))
        self.threshold_label.pack()
        self.threshold_slider.configure(
            command=lambda v: self.threshold_label.configure(text=f"threshold = {float(v):.2f}"))

        # ── Action Buttons ───────────────────────────────────────
        search_btn = ctk.CTkButton(sidebar, text="▶  RUN SEARCH", height=44,
                                    fg_color=self.PALETTE["accent1"],
                                    hover_color="#8880ff",
                                    font=("Segoe UI Black", 13),
                                    command=self._run_search)
        search_btn.pack(fill="x", padx=14, pady=(0, 6))

        compare_btn = ctk.CTkButton(sidebar, text="⇌  COMPARE ALL ALGORITHMS", height=38,
                                     fg_color=self.PALETTE["bg_card"],
                                     hover_color=self.PALETTE["accent1"],
                                     border_color=self.PALETTE["accent1"], border_width=1,
                                     font=("Segoe UI", 12),
                                     command=self._run_comparison)
        compare_btn.pack(fill="x", padx=14, pady=(0, 6))

        export_btn = ctk.CTkButton(sidebar, text="↓  EXPORT CSV", height=34,
                                    fg_color=self.PALETTE["bg_card"],
                                    hover_color="#1e1e4e",
                                    border_color=self.PALETTE["text_dim"], border_width=1,
                                    font=("Segoe UI", 11),
                                    command=self._export_csv)
        export_btn.pack(fill="x", padx=14, pady=(0, 4))

        pdf_btn = ctk.CTkButton(sidebar, text="⎙  EXPORT PDF REPORT", height=34,
                                 fg_color=self.PALETTE["bg_card"],
                                 hover_color="#1e1e4e",
                                 border_color=self.PALETTE["text_dim"], border_width=1,
                                 font=("Segoe UI", 11),
                                 command=self._export_pdf)
        pdf_btn.pack(fill="x", padx=14, pady=(0, 14))

        # ── Progress bar ─────────────────────────────────────────
        self.progress = ctk.CTkProgressBar(sidebar, height=6,
                                            progress_color=self.PALETTE["accent1"],
                                            fg_color=self.PALETTE["bg_card"])
        self.progress.set(0)
        self.progress.pack(fill="x", padx=14, pady=(0, 4))
        self.progress_label = ctk.CTkLabel(sidebar, text="", font=("Segoe UI", 9),
                                            text_color=self.PALETTE["text_dim"])
        self.progress_label.pack()

        # ── Execution time ───────────────────────────────────────
        self.time_label = ctk.CTkLabel(sidebar, text="⏱  Exec time: — ms",
                                        font=("Consolas", 10),
                                        text_color=self.PALETTE["accent2"])
        self.time_label.pack(pady=(6, 2))

        # Dataset count
        self.ds_count_label = ctk.CTkLabel(sidebar, text=f"📂 Dataset: {len(self.dataset)} docs",
                                            font=("Segoe UI", 10),
                                            text_color=self.PALETTE["text_dim"])
        self.ds_count_label.pack(pady=(0, 8))

    def _section(self, parent, text):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(f, text=text, font=("Segoe UI Black", 9),
                     text_color=self.PALETTE["text_dim"]).pack(side="left")
        ctk.CTkFrame(f, height=1, fg_color=self.PALETTE["bg_card"]).pack(
            side="left", fill="x", expand=True, padx=(8, 0))

    # ── RESULTS TAB ───────────────────────────────────────────────

    def _build_results_tab(self):
        tab = self.tabview.tab("🔍  Search Results")

        # Results list + chart split
        pane = ctk.CTkFrame(tab, fg_color="transparent")
        pane.pack(fill="both", expand=True)

        # Results table
        left = ctk.CTkFrame(pane, fg_color=self.PALETTE["bg_card"], corner_radius=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=0)

        ctk.CTkLabel(left, text="Search Results", font=("Segoe UI Black", 13),
                     text_color=self.PALETTE["text"]).pack(pady=(12, 4), padx=14, anchor="w")

        # Column headers
        header_f = ctk.CTkFrame(left, fg_color=self.PALETTE["bg_mid"], corner_radius=6)
        header_f.pack(fill="x", padx=10, pady=(0, 2))
        for col, w in [("#", 35), ("Document", 380), ("Score", 75), ("Bar", 120)]:
            ctk.CTkLabel(header_f, text=col, width=w, anchor="w",
                         font=("Segoe UI Black", 9), text_color=self.PALETTE["text_dim"]).pack(
                             side="left", padx=4, pady=6)

        # Scrollable results
        self.results_scroll = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.results_scroll.pack(fill="both", expand=True, padx=8, pady=4)

        # Right: mini chart
        right = ctk.CTkFrame(pane, fg_color=self.PALETTE["bg_card"],
                              corner_radius=10, width=280)
        right.pack(side="right", fill="y", padx=(4, 0))
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="Score Distribution", font=("Segoe UI Black", 12),
                     text_color=self.PALETTE["text"]).pack(pady=(12, 4), padx=12, anchor="w")

        self.mini_chart_frame = ctk.CTkFrame(right, fg_color="transparent")
        self.mini_chart_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_comparison_tab(self):
        tab = self.tabview.tab("📊  Algorithm Comparison")
        self.comparison_chart_frame = ctk.CTkFrame(tab, fg_color=self.PALETTE["bg_card"],
                                                    corner_radius=10)
        self.comparison_chart_frame.pack(fill="both", expand=True, padx=6, pady=6)
        ctk.CTkLabel(self.comparison_chart_frame,
                     text="Run 'Compare All Algorithms' to see results here",
                     text_color=self.PALETTE["text_dim"],
                     font=("Segoe UI", 13)).pack(expand=True)

    def _build_analytics_tab(self):
        tab = self.tabview.tab("📈  Analytics Dashboard")

        # Metric cards row
        cards_frame = ctk.CTkFrame(tab, fg_color="transparent")
        cards_frame.pack(fill="x", padx=6, pady=(6, 4))

        self.metric_cards = {}
        for metric, color, icon in [
            ("accuracy", self.PALETTE["accent1"], "🎯"),
            ("precision", self.PALETTE["accent2"], "📌"),
            ("recall", self.PALETTE["success"], "🔁"),
            ("f1", self.PALETTE["warning"], "⚡"),
            ("avg_score", self.PALETTE["accent3"], "∑"),
            ("max_score", "#00ffcc", "▲"),
        ]:
            card = ctk.CTkFrame(cards_frame, fg_color=self.PALETTE["bg_card"],
                                corner_radius=10, width=130, height=80)
            card.pack(side="left", fill="x", expand=True, padx=4)
            card.pack_propagate(False)
            ctk.CTkLabel(card, text=icon, font=("Segoe UI", 18)).pack(pady=(8, 0))
            val_lbl = ctk.CTkLabel(card, text="—", font=("Segoe UI Black", 16),
                                    text_color=color)
            val_lbl.pack()
            ctk.CTkLabel(card, text=metric.replace("_", " ").title(),
                         font=("Segoe UI", 9), text_color=self.PALETTE["text_dim"]).pack()
            self.metric_cards[metric] = val_lbl

        # Analytics chart area
        self.analytics_chart_frame = ctk.CTkFrame(tab, fg_color=self.PALETTE["bg_card"],
                                                   corner_radius=10)
        self.analytics_chart_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        ctk.CTkLabel(self.analytics_chart_frame,
                     text="Run a search to populate analytics",
                     text_color=self.PALETTE["text_dim"],
                     font=("Segoe UI", 13)).pack(expand=True)

    def _build_dataset_tab(self):
        tab = self.tabview.tab("📁  Dataset Manager")

        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=6, pady=6)

        ctk.CTkButton(top, text="📂  Upload Dataset (TXT/CSV)", height=38,
                       fg_color=self.PALETTE["accent1"], hover_color="#8880ff",
                       font=("Segoe UI", 12), command=self._upload_dataset).pack(side="left", padx=4)
        ctk.CTkButton(top, text="✕  Clear Dataset", height=38,
                       fg_color=self.PALETTE["bg_card"], hover_color="#3e0000",
                       border_color=self.PALETTE["accent3"], border_width=1,
                       font=("Segoe UI", 12), command=self._clear_dataset).pack(side="left", padx=4)
        ctk.CTkButton(top, text="↺  Restore Default", height=38,
                       fg_color=self.PALETTE["bg_card"], hover_color=self.PALETTE["bg_card"],
                       border_color=self.PALETTE["text_dim"], border_width=1,
                       font=("Segoe UI", 12), command=self._restore_default).pack(side="left", padx=4)

        self.ds_stats = ctk.CTkLabel(tab, text=f"Total entries: {len(self.dataset)}",
                                      text_color=self.PALETTE["text_dim"], font=("Segoe UI", 11))
        self.ds_stats.pack(pady=(0, 4))

        self.ds_scroll = ctk.CTkScrollableFrame(tab, fg_color=self.PALETTE["bg_card"],
                                                 corner_radius=10)
        self.ds_scroll.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._refresh_dataset_view()

    # ── PARTICLE ANIMATION ────────────────────────────────────────

    def _init_particles(self):
        for _ in range(18):
            self._particles.append({
                "x": random.uniform(0, 320),
                "y": random.uniform(0, 120),
                "vx": random.uniform(-0.5, 0.5),
                "vy": random.uniform(-0.4, 0.4),
                "r": random.uniform(1.5, 4),
                "col": random.choice([self.PALETTE["accent1"], self.PALETTE["accent2"],
                                       self.PALETTE["accent3"]]),
            })

    def _start_particle_animation(self):
        def _animate():
            if not self._anim_running:
                return
            try:
                c = self.canvas_bg
                c.delete("all")
                for p in self._particles:
                    p["x"] = (p["x"] + p["vx"]) % 320
                    p["y"] = (p["y"] + p["vy"]) % 120
                    r = p["r"]
                    c.create_oval(p["x"]-r, p["y"]-r, p["x"]+r, p["y"]+r,
                                  fill=p["col"], outline="", tags="particle")
                # Connect nearby particles
                for i, a in enumerate(self._particles):
                    for b in self._particles[i+1:]:
                        dist = math.hypot(a["x"]-b["x"], a["y"]-b["y"])
                        if dist < 60:
                            alpha = int(255 * (1 - dist/60))
                            col = f"#{alpha:02x}{alpha:02x}{'ff'}"
                            c.create_line(a["x"], a["y"], b["x"], b["y"],
                                          fill=self.PALETTE["accent1"],
                                          width=0.4)
                self.after(50, _animate)
            except Exception:
                pass
        _animate()

    # ── SEARCH LOGIC ─────────────────────────────────────────────

    def _on_algo_change(self, *_):
        show = self.algo_var.get() == "N-Gram"
        if show:
            self.ngram_frame.pack(fill="x", padx=14, pady=(0, 8))
        else:
            self.ngram_frame.pack_forget()

    def _run_search(self):
        if self._search_running:
            return
        query = self.query_text.get("1.0", "end").strip()
        if not query:
            messagebox.showwarning("Input Required", "Please enter a query string.")
            return
        if not self.dataset:
            messagebox.showwarning("No Dataset", "Please upload or restore a dataset.")
            return

        self._search_running = True
        self.status_dot.configure(text="● SEARCHING", text_color=self.PALETTE["warning"])
        self.progress.set(0)
        self.progress_label.configure(text="Initializing...")

        def _worker():
            algo = self.algo_var.get()
            k = int(self.k_slider.get())
            n = int(self.ngram_n.get())
            threshold = self.threshold_slider.get()

            self.after(0, lambda: self.progress.set(0.2))
            self.after(0, lambda: self.progress_label.configure(text="Running algorithm..."))

            t0 = time.perf_counter()
            results = SimilarityEngine.top_k_search(query, self.dataset, algo, k, n)
            elapsed = (time.perf_counter() - t0) * 1000

            self.after(0, lambda: self.progress.set(0.7))
            self.after(0, lambda: self.progress_label.configure(text="Computing metrics..."))

            metrics = MetricsEngine.compute_metrics(results, threshold)
            self.last_results = results
            self.last_query = query
            self.last_algo = algo

            self.after(0, lambda: self.progress.set(1.0))
            self.after(0, lambda: self._display_results(results, metrics, elapsed, algo))
            self.after(0, lambda: self.progress_label.configure(text=f"Done — {len(results)} results"))
            self.after(0, lambda: self.time_label.configure(
                text=f"⏱  Exec time: {elapsed:.2f} ms"))
            self.after(0, lambda: self.status_dot.configure(
                text="● READY", text_color=self.PALETTE["success"]))
            self._search_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _display_results(self, results, metrics, elapsed, algo):
        # Clear old results
        for w in self.results_scroll.winfo_children():
            w.destroy()

        colors_bar = [self.PALETTE["accent1"], self.PALETTE["accent2"],
                       self.PALETTE["success"], self.PALETTE["warning"]]

        for i, (doc, score, idx) in enumerate(results):
            row = ctk.CTkFrame(self.results_scroll,
                                fg_color=self.PALETTE["bg_mid"] if i % 2 == 0 else self.PALETTE["bg_card"],
                                corner_radius=6)
            row.pack(fill="x", pady=2, padx=2)

            ctk.CTkLabel(row, text=str(i+1), width=35, anchor="center",
                         font=("Consolas", 10), text_color=self.PALETTE["text_dim"]).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=doc[:75] + ("…" if len(doc) > 75 else ""),
                         width=380, anchor="w", font=("Segoe UI", 11),
                         text_color=self.PALETTE["text"]).pack(side="left", padx=4, pady=5)
            score_pct = score * 100
            col = self.PALETTE["success"] if score_pct >= 70 else \
                  self.PALETTE["warning"] if score_pct >= 40 else \
                  self.PALETTE["accent3"]
            ctk.CTkLabel(row, text=f"{score_pct:.1f}%", width=60,
                         font=("Consolas", 11, "bold"), text_color=col).pack(side="left", padx=4)
            # Mini score bar
            bar_bg = ctk.CTkFrame(row, width=120, height=10, fg_color=self.PALETTE["bg_dark"],
                                   corner_radius=5)
            bar_bg.pack(side="left", padx=4)
            bar_bg.pack_propagate(False)
            fill_w = max(1, int(score * 116))
            ctk.CTkFrame(bar_bg, width=fill_w, height=10, fg_color=col,
                          corner_radius=5).place(x=2, y=0)

        # Update metric cards
        for key, lbl in self.metric_cards.items():
            val = metrics.get(key, 0)
            lbl.configure(text=f"{val:.1f}%")

        # Mini chart (score distribution)
        self._draw_mini_chart(results)
        self._draw_analytics_chart(results, metrics, algo)

    def _draw_mini_chart(self, results):
        for w in self.mini_chart_frame.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(2.8, 5.2), facecolor=self.PALETTE["bg_card"])
        ax.set_facecolor(self.PALETTE["bg_mid"])

        labels = [r[0][:22]+"…" if len(r[0]) > 22 else r[0] for r in results]
        scores = [r[1]*100 for r in results]
        bar_colors = [self.PALETTE["success"] if s >= 70 else
                       self.PALETTE["warning"] if s >= 40 else
                       self.PALETTE["accent3"] for s in scores]

        bars = ax.barh(range(len(scores)), scores, color=bar_colors, height=0.65)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5, color=self.PALETTE["text"])
        ax.set_xlim(0, 105)
        ax.set_xlabel("Similarity %", color=self.PALETTE["text_dim"], fontsize=8)
        ax.tick_params(colors=self.PALETTE["text_dim"], labelsize=7)
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_edgecolor(self.PALETTE["bg_card"])
        ax.axvline(50, color=self.PALETTE["text_dim"], linewidth=0.5, linestyle="--", alpha=0.5)
        fig.tight_layout(pad=0.4)

        canvas = FigureCanvasTkAgg(fig, self.mini_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _draw_analytics_chart(self, results, metrics, algo):
        for w in self.analytics_chart_frame.winfo_children():
            w.destroy()

        fig = plt.figure(figsize=(9, 5.2), facecolor=self.PALETTE["bg_card"])
        gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.5)

        # 1. Score histogram
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.set_facecolor(self.PALETTE["bg_mid"])
        scores = [r[1]*100 for r in results]
        ax1.hist(scores, bins=10, color=self.PALETTE["accent1"], alpha=0.85, edgecolor=self.PALETTE["bg_dark"])
        ax1.set_title("Score Distribution", color=self.PALETTE["text"], fontsize=10, pad=4)
        ax1.set_xlabel("Similarity %", color=self.PALETTE["text_dim"], fontsize=8)
        ax1.tick_params(colors=self.PALETTE["text_dim"], labelsize=7)
        for sp in ax1.spines.values(): sp.set_edgecolor(self.PALETTE["bg_card"])

        # 2. Metrics radar-style bar
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor(self.PALETTE["bg_mid"])
        m_names = ["Accuracy", "Precision", "Recall", "F1"]
        m_vals = [metrics.get("accuracy", 0), metrics.get("precision", 0),
                  metrics.get("recall", 0), metrics.get("f1", 0)]
        m_colors = [self.PALETTE["accent1"], self.PALETTE["accent2"],
                    self.PALETTE["success"], self.PALETTE["warning"]]
        ax2.bar(m_names, m_vals, color=m_colors, width=0.6)
        ax2.set_ylim(0, 110)
        ax2.set_title("Metrics Overview", color=self.PALETTE["text"], fontsize=10, pad=4)
        ax2.tick_params(colors=self.PALETTE["text_dim"], labelsize=7)
        for sp in ax2.spines.values(): sp.set_edgecolor(self.PALETTE["bg_card"])

        # 3. Cumulative score curve
        ax3 = fig.add_subplot(gs[1, 0:2])
        ax3.set_facecolor(self.PALETTE["bg_mid"])
        sorted_scores = sorted(scores, reverse=True)
        ax3.plot(range(1, len(sorted_scores)+1), sorted_scores,
                  color=self.PALETTE["accent2"], linewidth=2, marker='o', markersize=3)
        ax3.fill_between(range(1, len(sorted_scores)+1), sorted_scores,
                          alpha=0.2, color=self.PALETTE["accent2"])
        ax3.set_title("Score Curve (Top-K)", color=self.PALETTE["text"], fontsize=10, pad=4)
        ax3.set_xlabel("Rank", color=self.PALETTE["text_dim"], fontsize=8)
        ax3.set_ylabel("Similarity %", color=self.PALETTE["text_dim"], fontsize=8)
        ax3.tick_params(colors=self.PALETTE["text_dim"], labelsize=7)
        for sp in ax3.spines.values(): sp.set_edgecolor(self.PALETTE["bg_card"])

        # 4. Relevance pie
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_facecolor(self.PALETTE["bg_mid"])
        threshold = self.threshold_slider.get() * 100
        high = sum(s >= threshold for s in scores)
        low = len(scores) - high
        if high + low > 0:
            ax4.pie([high, low],
                     labels=["Relevant", "Not Relevant"],
                     colors=[self.PALETTE["success"], self.PALETTE["accent3"]],
                     autopct="%1.0f%%", textprops={"color": self.PALETTE["text"], "fontsize": 8},
                     startangle=90)
        ax4.set_title("Relevance Split", color=self.PALETTE["text"], fontsize=10, pad=4)

        canvas = FigureCanvasTkAgg(fig, self.analytics_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ── COMPARISON ────────────────────────────────────────────────

    def _run_comparison(self):
        query = self.query_text.get("1.0", "end").strip()
        if not query:
            messagebox.showwarning("Input Required", "Please enter a query string.")
            return
        if not self.dataset:
            messagebox.showwarning("No Dataset", "Dataset is empty.")
            return

        self.status_dot.configure(text="● COMPARING", text_color=self.PALETTE["warning"])

        def _worker():
            k = int(self.k_slider.get())
            n = int(self.ngram_n.get())
            data = {}
            for algo in self.ALGORITHMS:
                t0 = time.perf_counter()
                results = SimilarityEngine.top_k_search(query, self.dataset, algo, k, n)
                elapsed = (time.perf_counter() - t0) * 1000
                scores = [r[1] for r in results]
                data[algo] = {
                    "top_score": max(scores) * 100 if scores else 0,
                    "avg_score": np.mean(scores) * 100 if scores else 0,
                    "time_ms": elapsed,
                    "results": results,
                }
            self.comparison_data = data
            self.after(0, lambda: self._draw_comparison(data))
            self.after(0, lambda: self.status_dot.configure(
                text="● READY", text_color=self.PALETTE["success"]))

        threading.Thread(target=_worker, daemon=True).start()

    def _draw_comparison(self, data):
        for w in self.comparison_chart_frame.winfo_children():
            w.destroy()

        fig = plt.figure(figsize=(11, 6), facecolor=self.PALETTE["bg_card"])
        gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.5)

        algos = list(data.keys())
        top_scores = [data[a]["top_score"] for a in algos]
        avg_scores = [data[a]["avg_score"] for a in algos]
        times = [data[a]["time_ms"] for a in algos]
        bar_colors = [self.PALETTE["accent1"], self.PALETTE["accent2"],
                       self.PALETTE["success"], self.PALETTE["warning"],
                       self.PALETTE["accent3"], "#00ffcc", "#ff9a3c", "#c77dff"]

        # Top score comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(self.PALETTE["bg_mid"])
        bars = ax1.bar(algos, top_scores, color=bar_colors, width=0.6)
        ax1.set_title("Top Score by Algorithm (%)", color=self.PALETTE["text"], fontsize=10)
        ax1.set_ylabel("%", color=self.PALETTE["text_dim"], fontsize=8)
        ax1.set_ylim(0, 115)
        ax1.tick_params(colors=self.PALETTE["text_dim"], labelsize=6.5, axis='x', rotation=30)
        ax1.tick_params(colors=self.PALETTE["text_dim"], labelsize=7, axis='y')
        for sp in ax1.spines.values(): sp.set_edgecolor(self.PALETTE["bg_card"])
        for bar, val in zip(bars, top_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                      f"{val:.1f}", ha='center', va='bottom', fontsize=6.5,
                      color=self.PALETTE["text"])

        # Avg score comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(self.PALETTE["bg_mid"])
        bars2 = ax2.bar(algos, avg_scores, color=bar_colors, width=0.6, alpha=0.85)
        ax2.set_title("Average Score by Algorithm (%)", color=self.PALETTE["text"], fontsize=10)
        ax2.set_ylabel("%", color=self.PALETTE["text_dim"], fontsize=8)
        ax2.set_ylim(0, 115)
        ax2.tick_params(colors=self.PALETTE["text_dim"], labelsize=6.5, axis='x', rotation=30)
        ax2.tick_params(colors=self.PALETTE["text_dim"], labelsize=7, axis='y')
        for sp in ax2.spines.values(): sp.set_edgecolor(self.PALETTE["bg_card"])

        # Execution time
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(self.PALETTE["bg_mid"])
        ax3.barh(algos, times, color=bar_colors, height=0.6)
        ax3.set_title("Execution Time (ms)", color=self.PALETTE["text"], fontsize=10)
        ax3.set_xlabel("ms", color=self.PALETTE["text_dim"], fontsize=8)
        ax3.tick_params(colors=self.PALETTE["text_dim"], labelsize=7)
        for sp in ax3.spines.values(): sp.set_edgecolor(self.PALETTE["bg_card"])

        # Radar-style: top vs avg vs normalized time
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(self.PALETTE["bg_mid"])
        x = np.arange(len(algos))
        w = 0.25
        ax4.bar(x - w, top_scores, w, label="Top Score", color=self.PALETTE["accent1"], alpha=0.9)
        ax4.bar(x,     avg_scores, w, label="Avg Score", color=self.PALETTE["accent2"], alpha=0.9)
        max_t = max(times) or 1
        norm_times = [t/max_t*100 for t in times]
        ax4.bar(x + w, norm_times, w, label="Time (norm %)", color=self.PALETTE["warning"], alpha=0.9)
        ax4.set_title("Multi-Metric Comparison", color=self.PALETTE["text"], fontsize=10)
        ax4.set_xticks(x)
        ax4.set_xticklabels(algos, rotation=30, fontsize=6.5, color=self.PALETTE["text_dim"])
        ax4.tick_params(colors=self.PALETTE["text_dim"], labelsize=7)
        ax4.legend(fontsize=7, facecolor=self.PALETTE["bg_dark"],
                    labelcolor=self.PALETTE["text"], edgecolor=self.PALETTE["bg_card"])
        for sp in ax4.spines.values(): sp.set_edgecolor(self.PALETTE["bg_card"])

        fig.tight_layout(pad=0.5)
        canvas = FigureCanvasTkAgg(fig, self.comparison_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

        self.tabview.set("📊  Algorithm Comparison")

    # ── DATASET MANAGEMENT ────────────────────────────────────────

    def _upload_dataset(self):
        fp = filedialog.askopenfilename(
            title="Upload Dataset",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not fp:
            return
        try:
            entries = []
            if fp.endswith(".csv"):
                df = pd.read_csv(fp)
                for col in df.columns:
                    entries += df[col].dropna().astype(str).tolist()
            else:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    entries = [line.strip() for line in f if line.strip()]
            self.dataset = entries
            self._refresh_dataset_view()
            messagebox.showinfo("Success", f"Loaded {len(entries)} entries from dataset.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _clear_dataset(self):
        if messagebox.askyesno("Confirm", "Clear the entire dataset?"):
            self.dataset = []
            self._refresh_dataset_view()

    def _restore_default(self):
        self.dataset = list(DEFAULT_DATASET)
        self._refresh_dataset_view()

    def _refresh_dataset_view(self):
        for w in self.ds_scroll.winfo_children():
            w.destroy()
        for i, doc in enumerate(self.dataset):
            row = ctk.CTkFrame(self.ds_scroll,
                                fg_color=self.PALETTE["bg_mid"] if i % 2 else self.PALETTE["bg_card"],
                                corner_radius=4)
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=str(i+1), width=40, anchor="center",
                         font=("Consolas", 9), text_color=self.PALETTE["text_dim"]).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=doc, anchor="w", font=("Segoe UI", 10),
                         text_color=self.PALETTE["text"]).pack(side="left", padx=4, pady=3)
        self.ds_stats.configure(text=f"Total entries: {len(self.dataset)}")
        self.ds_count_label.configure(text=f"📂 Dataset: {len(self.dataset)} docs")

    # ── EXPORT ────────────────────────────────────────────────────

    def _export_csv(self):
        if not self.last_results:
            messagebox.showwarning("No Results", "Run a search first.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".csv",
                                           filetypes=[("CSV", "*.csv")])
        if not fp:
            return
        with open(fp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Document", "Similarity Score", "Algorithm"])
            for i, (doc, score, _) in enumerate(self.last_results, 1):
                writer.writerow([i, doc, f"{score*100:.4f}%", getattr(self, "last_algo", "")])
        messagebox.showinfo("Exported", f"Results saved to:\n{fp}")

    def _export_pdf(self):
        if not self.last_results:
            messagebox.showwarning("No Results", "Run a search first.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".pdf",
                                           filetypes=[("PDF", "*.pdf")])
        if not fp:
            return
        # Save chart
        chart_path = fp.replace(".pdf", "_chart.png")
        self._save_chart_image(chart_path)
        metrics = MetricsEngine.compute_metrics(
            self.last_results, self.threshold_slider.get())
        PDFReporter.generate(fp, getattr(self, "last_query", ""),
                              getattr(self, "last_algo", ""),
                              self.last_results, metrics, chart_path)
        if os.path.exists(chart_path):
            os.remove(chart_path)
        messagebox.showinfo("PDF Report", f"Report saved to:\n{fp}")

    def _save_chart_image(self, path: str):
        """Save a simple score chart as image for PDF embedding."""
        if not self.last_results:
            return
        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#ffffff")
        ax.set_facecolor("#f0f4ff")
        scores = [r[1]*100 for r in self.last_results]
        labels = [r[0][:30]+"…" if len(r[0]) > 30 else r[0] for r in self.last_results]
        bar_colors = ["#4CAF50" if s >= 70 else "#FF9800" if s >= 40 else "#f44336"
                       for s in scores]
        ax.barh(range(len(scores)), scores, color=bar_colors, height=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Similarity %")
        ax.invert_yaxis()
        ax.set_title("Top-K Search Results", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def on_close(self):
        self._anim_running = False
        self.destroy()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = StringSimilarityApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
