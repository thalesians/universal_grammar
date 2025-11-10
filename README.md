# universal_grammar

**universal_grammar** is a cross-linguistic NLP experiment that empirically tests for universal predicate–argument structures in human language.  
It parses multilingual versions of *Genesis* using [Stanza](https://stanfordnlp.github.io/stanza/) and [Universal Dependencies](https://universaldependencies.org/), then evaluates cross-lingual structural similarity through a permutation-based significance test.

---

## Overview

The project explores whether deep syntactic structures — specifically predicate–argument relations such as subjects, objects, and obliques — remain invariant across languages that differ in word order, morphology, and family.  
The experiment draws on the **Bible-corpus** (Genesis) for parallel verse alignment, computes normalized dependency-role vectors per verse, and measures structural similarity across languages.

A permutation test assesses whether the observed cross-lingual syntactic alignment exceeds what would be expected by chance, providing quantitative evidence consistent with the existence of structural universals (as predicted by theories of Universal Grammar).

---

## Languages Included

| Family | Languages |
|---------|------------|
| Indo-European | English, German, French, Spanish, Italian, Russian, Latin |
| Semitic | Arabic, Hebrew |
| Indo-Aryan | Hindi |
| Uralic | Hungarian |
| Sino-Tibetan | Chinese |
| Hellenic | Greek (Koine) |

---

## Requirements

- Python >= 3.9  
- [Stanza](https://stanfordnlp.github.io/stanza/)  
- `numpy`, `pandas`, `requests`, `tqdm`

Install dependencies:

```bash
pip install stanza numpy pandas requests tqdm
```

---

## Usage

Clone the repository and run the main experiment:

```bash
python universal_grammar.py
```

By default, the script:
1. Downloads the Book of *Genesis* in multiple languages from the [bible-corpus](https://github.com/christos-c/bible-corpus) (CC0).
2. Parses each verse independently using Universal Dependencies.
3. Computes normalized role vectors and dependency lengths.
4. Performs a permutation test to measure cross-lingual structural invariance.

Results are printed to the console, including:
- Observed mean pairwise structural similarity  
- Permutation test p-value  
- Mean dependency lengths per language  

---

## Example Output

```
Observed mean pairwise structural similarity: 0.7124
Permutation p-value (right-tailed): p = 0.0050
DL variance p-value (left-tailed): 0.8950
```

Interpretation: aligned verses across 13 languages display significantly higher structural similarity than random permutations, suggesting deep cross-linguistic invariants in predicate–argument architecture.

---

## Citation

If you use this code or replicate the results, please cite:

> Bilokon, P.A. (2025). *Quantitative Evidence for Universal Grammar: A Cross-Linguistic Test Using Multilingual Genesis and Universal Dependencies.*

---

## License

This project is released under the **Apache License 2.0**.  
Bible text data courtesy of the [bible-corpus project](https://github.com/christos-c/bible-corpus) (CC0).

---

## Related Resources

- [Universal Dependencies](https://universaldependencies.org/)  
- [Stanza NLP Toolkit](https://stanfordnlp.github.io/stanza/)  
- [Christodoulopoulos & Steedman (2015) — *The Bible in 100 Languages*](https://aclanthology.org/L14-1627/)
