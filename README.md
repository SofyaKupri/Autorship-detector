# Who Wrote This?

Guesses the author of a text from writing style, using **Burrows' Delta**

How it works:

1. Books of several authors are read from the NLTK Gutenberg corpus (one zip
   file, downloaded once into `books/`) and cut into chunks of 2000 words.
2. For every chunk we measure how often each of the 150 most common words
   (mostly "the", "of", "but", "which"...) is used. Such small words are used
   unconsciously, so they are a good fingerprint of an author.
3. Frequencies are converted to z-scores, and the average z-score vector of an
   author's chunks becomes the author's style profile.
4. An unknown text gets the same kind of vector. The **Delta** is the mean
   absolute difference between it and a profile. The smallest Delta wins.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --evaluate
python main.py my_text.txt
python main.py --text "It is a truth universally acknowledged, ..."
pytest
ruff check .
```

If the automatic download fails, download
`https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip`
in a browser and save it as `books/gutenberg.zip`.

Texts shorter than about 300 words give unreliable results. Books must be
English (the tokenizer only keeps the letters a-z).
