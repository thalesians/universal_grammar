#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Empirical test of cross-lingual predicate-argument invariants on Genesis.

H0 (Null): Core dependency structures per verse are independent across languages.
H1 (UG-ish): Aligned verses show higher cross-language structural similarity than chance.

Test statistic: mean cosine similarity of role vectors across languages, verse-aligned.
Significance via permutation test (randomly permute verses in one language).

Requirements:
  pip install stanza pandas requests tqdm numpy

Notes:
- Parses EACH verse separately to avoid huge memory spikes.
- Uses conservative Stanza batch sizes; GPU disabled by default on Windows.
- Start with small MAX_VERSES and N_PERM, scale up after verifying.
"""

import os
import random
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import stanza

# --------------------------- Config ----------------------------------

LANGS: List[Tuple[str, str]] = [
    ("Arabic",    "ar"),
    ("Chinese",   "zh"),    # Mandarin (Simplified, mainland)
    ("English",   "en"),
    ("French",    "fr"),
    ("German",    "de"),
    ("Greek",     "grc"),   # Ancient/Koine Greek
    ("Hebrew",    "he"),    # Biblical Hebrew
    ("Hindi",     "hi"),
    ("Hungarian", "hu"),
    ("Italian",   "it"),
    ("Latin",     "la"),
    ("Russian",   "ru"),
    ("Spanish",   "es"),
]

BOOK_ID = "b.GEN"  # Genesis
BIBLE_XML_BASE = "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/{lang}.xml"

# Keep batch sizes modest to limit RAM; disable GPU by default on Windows.
STANZA_ARGS = dict(
    processors="tokenize,pos,lemma,depparse",
    use_gpu=False,
    tokenize_batch_size=256,
    pos_batch_size=128,
    depparse_batch_size=32,
    verbose=False,
)

CORE_ROLES = ["nsubj", "csubj", "obj", "iobj", "obl", "xcomp", "ccomp"]

# Experiment knobs
N_PERM = 199        # increase to 999+ once stable
RANDOM_SEED = 13
MAX_VERSES = None   # e.g., 200 for a quick run; None = all aligned verses

# ------------------------ Data loading --------------------------------

def fetch_genesis(lang_name: str) -> List[str]:
    """Download XML for the given language from bible-corpus (CC0) and extract Genesis verses."""
    url = BIBLE_XML_BASE.format(lang=lang_name)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    verses = []
    for seg in root.findall(f'.//div[@id="{BOOK_ID}"]/*seg'):
        txt = (seg.text or "").strip()
        if txt:
            verses.append(txt)
    return verses

def ensure_stanza(lang_code: str):
    """Build a conservative Stanza pipeline; download models if missing."""
    try:
        return stanza.Pipeline(lang=lang_code, **STANZA_ARGS)
    except Exception:
        stanza.download(lang_code, verbose=False)
        return stanza.Pipeline(lang=lang_code, **STANZA_ARGS)

# ------------------------ Feature extraction --------------------------

def _role_vector_from_doc(doc) -> np.ndarray:
    """Aggregate CORE_ROLES counts over all sentences in a Stanza doc; L1-normalize."""
    c = Counter()
    for sent in doc.sentences:
        labs = [w.deprel for w in sent.words]
        for role in CORE_ROLES:
            c[role] += labs.count(role)
    v = np.array([c[r] for r in CORE_ROLES], dtype=float)
    s = v.sum()
    if s > 0:
        v = v / s
    return v

def parse_role_vectors(nlp, verses: List[str]) -> List[np.ndarray]:
    """
    One role vector per verse. We call nlp() on EACH verse (safe on memory).
    If stanza splits a verse into multiple sentences, we aggregate counts.
    """
    vecs = []
    for i, verse in enumerate(tqdm(verses, desc="Parsing verses", leave=False)):
        verse = (verse or "").strip()
        if not verse:
            vecs.append(np.zeros(len(CORE_ROLES), dtype=float))
            continue
        try:
            doc = nlp(verse)
            vecs.append(_role_vector_from_doc(doc))
        except Exception as e:
            # Robustness: if a verse crashes, log and fall back to zeros (rare)
            print(f"[WARN] Skipping verse {i} due to parse error: {e}")
            vecs.append(np.zeros(len(CORE_ROLES), dtype=float))
    assert len(vecs) == len(verses)
    return vecs

def dep_length_mean(nlp, verses: List[str]) -> float:
    """Average dependency length over all tokens in all verses."""
    total, cnt = 0, 0
    for i, verse in enumerate(tqdm(verses, desc="DL compute", leave=False)):
        verse = (verse or "").strip()
        if not verse:
            continue
        try:
            doc = nlp(verse)
        except Exception as e:
            print(f"[WARN] Skipping verse {i} in DL due to parse error: {e}")
            continue
        for sent in doc.sentences:
            for w in sent.words:
                if w.head != 0:
                    total += abs(w.id - w.head)
                    cnt += 1
    return (total / cnt) if cnt else float("nan")

# --------------------------- Similarity & test ------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def mean_pairwise_similarity(role_mats: Dict[str, List[np.ndarray]]) -> float:
    """
    role_mats: language -> list of vectors [v_0, v_1, ..., v_{V-1}]
    Assumes verse alignment by index.
    Returns mean cosine similarity across all language pairs and verses.
    """
    langs = list(role_mats.keys())
    V = len(role_mats[langs[0]])
    for L in langs[1:]:
        assert len(role_mats[L]) == V, "All languages must have same verse count."
    sims = []
    for i in range(V):
        for a in range(len(langs)):
            va = role_mats[langs[a]][i]
            for b in range(a + 1, len(langs)):
                vb = role_mats[langs[b]][i]
                sims.append(cosine_sim(va, vb))
    return float(np.mean(sims)) if sims else np.nan

def permute_and_score(role_mats: Dict[str, List[np.ndarray]], n_perm: int, rng: random.Random, permute_lang: str = None) -> List[float]:
    """
    Build null by permuting verse indices in one language (default: the first).
    """
    langs = list(role_mats.keys())
    if permute_lang is None:
        permute_lang = langs[0]
    V = len(role_mats[permute_lang])
    base = {L: np.array(role_mats[L]) for L in langs}
    scores = []
    for _ in tqdm(range(n_perm), desc="Permuting", leave=False):
        idx = list(range(V))
        rng.shuffle(idx)
        perm_mats = {}
        for L in langs:
            if L == permute_lang:
                perm_mats[L] = base[L][idx]
            else:
                perm_mats[L] = base[L]
        perm_mats = {L: list(perm_mats[L]) for L in langs}
        scores.append(mean_pairwise_similarity(perm_mats))
    return scores

# ------------------------------ Main ----------------------------------

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1) Download verses for each language
    verses_by_lang: Dict[str, List[str]] = {}
    min_len = 10**9
    print("Downloading Genesis...")
    for lang_name, code in LANGS:
        verses = fetch_genesis(lang_name)
        if MAX_VERSES:
            verses = verses[:MAX_VERSES]
        verses_by_lang[lang_name] = verses
        min_len = min(min_len, len(verses))
        print(f"  {lang_name}: {len(verses)} verses")

    # Align by truncating to common length
    for L in verses_by_lang:
        verses_by_lang[L] = verses_by_lang[L][:min_len]
    print(f"\nAligned to {min_len} verses across all languages.")

    # 2) Parse with stanza and compute role vectors + DL
    role_mats: Dict[str, List[np.ndarray]] = {}
    mean_dl: Dict[str, float] = {}
    print("\nParsing & extracting features...")
    for lang_name, code in LANGS:
        print(f"  > {lang_name} ({code})")
        nlp = ensure_stanza(code)
        role_mats[lang_name] = parse_role_vectors(nlp, verses_by_lang[lang_name])
        mean_dl[lang_name] = dep_length_mean(nlp, verses_by_lang[lang_name])
        print(f"    vectors ready; mean dependency length = {mean_dl[lang_name]:.3f}")

    # 3) Observed statistic
    T_obs = mean_pairwise_similarity(role_mats)
    print(f"\nObserved mean pairwise structural similarity: {T_obs:.4f}")

    # 4) Permutation test (permute one language)
    rng = random.Random(RANDOM_SEED)
    perm_lang = LANGS[0][0]  # permute the first language's verse order
    print(f"Building null by permuting verses in: {perm_lang} ({N_PERM} permutations)...")
    null_scores = permute_and_score(role_mats, N_PERM, rng, permute_lang=perm_lang)
    null_scores = np.array(null_scores, dtype=float)

    # Right-tailed: high similarity is evidence against H0
    p_val = float((np.sum(null_scores >= T_obs) + 1) / (len(null_scores) + 1))  # +1 smoothing
    print(f"\nPermutation p-value (right-tailed): p = {p_val:.4f}")

    # 5) Secondary check: DL variance vs null (left-tailed: unusually small spread suggests a universal pressure)
    dl_vals = np.array(list(mean_dl.values()))
    obs_spread = float(np.var(dl_vals))
    dl_null = []
    for _ in tqdm(range(N_PERM), desc="DL variance null", leave=False):
        shuffled = np.random.permutation(dl_vals)
        dl_null.append(float(np.var(shuffled)))
    dl_null = np.array(dl_null)
    p_val_dl = float((np.sum(dl_null <= obs_spread) + 1) / (len(dl_null) + 1))

    # 6) Compact report
    print("\n=== REPORT ===")
    print(f"Languages: {', '.join([L for L,_ in LANGS])}")
    print(f"Verses aligned: {min_len}")
    print(f"Mean pairwise structural similarity (obs): {T_obs:.4f}")
    print(f"Permutation p-value: {p_val:.4f}")
    print("Mean dependency length by language:")
    for L, v in mean_dl.items():
        print(f"  {L:>8}: {v:.3f}")
    print(f"DL variance p-value (left-tailed): {p_val_dl:.4f}")

if __name__ == "__main__":
    main()
