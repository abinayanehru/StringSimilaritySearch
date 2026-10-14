# 🧬 String Similarity Search System 

## What This System Does

A full-featured, professional desktop application that implements and compares **8 string similarity algorithms** with a modern animated GUI. Features include:

| Feature | Description |
|---|---|
| 8 Algorithms | N-Gram, Edit Distance, Cosine, TF-IDF Cosine, Jaccard, Jaro-Winkler, Soundex, Fuzzy |
| Top-K Search | Configurable number of results (1–50) |
| Metrics | Accuracy, Precision, Recall, F1 Score per search |
| Comparison | Side-by-side algorithm comparison with charts |
| Analytics | Dashboard with histograms, pie charts, score curves |
| Export | CSV results + PDF report with embedded charts |
| Dataset | Upload custom TXT/CSV datasets |
| Animated UI | Particle background, color-coded results, live progress |

---

## ✅ Prerequisites

- **Python 3.10 or newer** (3.11 / 3.12 recommended)
- **VS Code** (with Python extension)
- **Internet connection** (for pip install, first time only)

---

## 📥 Step 1 — Install Python

1. Go to https://www.python.org/downloads/
2. Download **Python 3.12.x** (latest stable)
3. During installation:
   - ✅ Check **"Add Python to PATH"**
   - Click **Install Now**
4. Verify: open **Command Prompt** and type:
   ```
   python --version
   ```
   You should see: `Python 3.12.x`

---

## 📁 Step 2 — Set Up the Project in VS Code

1. Create a folder on your Desktop, e.g., `StringSimilarity`
2. Copy `main.py` and `requirements.txt` into this folder
3. Open VS Code
4. Click **File → Open Folder** → select `StringSimilarity`
5. Open the built-in terminal: **Terminal → New Terminal**

---

## 📦 Step 3 — Install Required Packages

In the VS Code terminal, run this **one command**:

```bash
pip install customtkinter matplotlib numpy pandas scikit-learn Pillow reportlab scipy
```

**Expected output** (takes 1–3 minutes):
```
Successfully installed customtkinter-5.2.2 matplotlib-3.9.0 ...
```

### If pip is not recognized:
```bash
python -m pip install customtkinter matplotlib numpy pandas scikit-learn Pillow reportlab scipy
```

### If you get permission errors (Windows):
```bash
pip install --user customtkinter matplotlib numpy pandas scikit-learn Pillow reportlab scipy
```

---

## ▶️ Step 4 — Run the Application

In the VS Code terminal:

```bash
python main.py
```

The application window will open in 2–5 seconds with the animated GUI.

---

## 🎮 How to Use — Feature Guide

### Basic Search
1. Type your search query in the **"SEARCH QUERY"** text box (left panel)
2. Select an algorithm from the **"ALGORITHM"** dropdown
3. Adjust **Top-K** slider (how many results to show)
4. Click **▶ RUN SEARCH**
5. Results appear in the **"Search Results"** tab with score bars

### N-Gram Settings
- When you select "N-Gram" algorithm, a slider appears for **N-Gram Size (n)**
- n=2 (bigrams) is recommended for most use cases
- n=1 = unigrams, n=3 = trigrams

### Algorithm Comparison
1. Enter your query
2. Click **⇌ COMPARE ALL ALGORITHMS**
3. Switch to the **"Algorithm Comparison"** tab
4. See 4 comparison charts: top scores, avg scores, execution time, multi-metric

### Analytics Dashboard
- The **"Analytics Dashboard"** tab auto-updates after every search
- Shows: score histogram, metrics bar chart, score curve, relevance pie chart
- Metric cards at the top show live Accuracy / Precision / Recall / F1

### Custom Dataset
1. Go to **"Dataset Manager"** tab
2. Click **📂 Upload Dataset**
3. Select a `.txt` file (one entry per line) or `.csv` file
4. The system loads all text entries automatically
5. Click **↺ Restore Default** to go back to the built-in 50-document dataset

### Export Results
- **↓ EXPORT CSV** → saves ranked results with scores to a .csv file
- **⎙ EXPORT PDF REPORT** → generates a professional PDF with charts, metrics table, and full results

### Relevance Threshold
- The **"Relevance Threshold"** slider controls what counts as "relevant"
- Used for Precision/Recall/F1 calculation and the pie chart split

---

## 📊 Algorithm Reference

| Algorithm | Best For | Score Range |
|---|---|---|
| **N-Gram** | Short strings, typos | 0–1 |
| **Edit Distance** | Character-level similarity | 0–1 |
| **Cosine** | Word-level bag-of-words | 0–1 |
| **TF-IDF Cosine** | Document search (best for multi-word) | 0–1 |
| **Jaccard** | Word set overlap | 0–1 |
| **Jaro-Winkler** | Names, short strings | 0–1 |
| **Soundex** | Phonetic matching (sounds alike) | 0 or 1 |
| **Fuzzy** | Combined token + edit similarity | 0–1 |

---

## 🛠️ Troubleshooting

### `ModuleNotFoundError: No module named 'customtkinter'`
```bash
pip install customtkinter
```

### `ModuleNotFoundError: No module named 'tkinter'`
On Linux:
```bash
sudo apt-get install python3-tk
```
On Mac, reinstall Python from python.org (includes Tk).

### Window doesn't open / crashes immediately
- Check that Python is 3.10+
- Try: `python -m tkinter` — if a small window opens, tkinter works

### PDF export fails
```bash
pip install reportlab --upgrade
```

### Charts don't render
```bash
pip install matplotlib --upgrade
```

---

## 📂 Project File Structure

```
StringSimilarity/
├── main.py              ← Main application (run this)
├── requirements.txt     ← Package list
└── README.md           ← This guide
```

All algorithms, UI, metrics, and export logic are in `main.py` — single-file architecture for easy review.

---

## 🎓 Academic Notes

### Algorithms Implemented (from scratch, no external similarity libraries):
- **N-Gram**: Jaccard overlap of character n-grams
- **Edit Distance**: Wagner-Fischer DP algorithm O(mn)
- **Cosine**: Manual TF vectorization + dot product
- **Jaro-Winkler**: Full Jaro computation + prefix bonus
- **Soundex**: Phonetic encoding + equality check

### Sklearn used only for:
- TF-IDF vectorization (TfidfVectorizer)
- Cosine similarity on TF-IDF matrix

### Metrics computed using:
- Ground truth simulation (top 30% = relevant)
- Standard Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × P × R / (P + R)

---

*String Similarity Search System — Final Year Project*
*Built with Python 3.12, CustomTkinter, Matplotlib, scikit-learn*
