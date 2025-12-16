
import re
import html
from typing import List

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
NON_ALNUM_PATTERN = re.compile(r"[^0-9a-zA-Z\s]+")  # keep alnum and spaces
MULTI_SPACE = re.compile(r"\s+")

SLANG_MAP = {
    "gw": "saya",
    "gue": "saya",
    "gak": "tidak",
    "ga": "tidak",
    "tdk": "tidak",
    "td": "tidak",
    "nih": "",
    "klo": "kalau",
    "kalo": "kalau",
    "dg": "dengan",
    "yg": "yang",
    "bgt": "banget",
    "btw": "omong",
}

EMOTICON_MAP = {
    ":)": "senang",
    ":-)": "senang",
    ":(": "sedih",
    ":-(": "sedih",
    "😂": "tertawa",
    "😭": "sedih",
    "😊": "senang",
    "😒": "jengkel",
}

def replace_slangs(text: str, slang_map: dict) -> str:
    tokens = text.split()
    out = []
    for t in tokens:
        key = t.lower()
        out.append(slang_map.get(key, t))
    return " ".join(out)

def replace_emoticons(text: str, emoticon_map: dict) -> str:
    for k, v in emoticon_map.items():
        text = text.replace(k, " " + v + " ")
    return text

def clean_text_basic(text: str, do_lower=True, remove_punct=True, normalize_slang=True) -> str:
    if text is None:
        return ""
    s = str(text)
    s = html.unescape(s)
    s = URL_PATTERN.sub(" ", s)
    s = MENTION_PATTERN.sub(" ", s)
    s = replace_emoticons(s, EMOTICON_MAP)
    if do_lower:
        s = s.lower()
    if normalize_slang:
        s = replace_slangs(s, SLANG_MAP)
    if remove_punct:
        s = NON_ALNUM_PATTERN.sub(" ", s)
    s = MULTI_SPACE.sub(" ", s).strip()
    return s

def tokenize(text: str) -> List[str]:
    return text.split()

def preprocess_texts(texts: List[str], do_lower=True, remove_punct=True, normalize_slang=True) -> List[str]:
    return [clean_text_basic(t, do_lower=do_lower, remove_punct=remove_punct, normalize_slang=normalize_slang) for t in texts]
