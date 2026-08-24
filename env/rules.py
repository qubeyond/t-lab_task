"""Домен: маршрутизация тикетов поддержки, темы/стили/шаблоны текста, таблица верных действий."""

from __future__ import annotations

import numpy as np

TOPICS = [
    "billing",
    "technical",
    "account",
    "refund",
    "security",
    "feature_request",
]
N_TOPICS = len(TOPICS)
TOPIC_TO_IDX = {t: i for i, t in enumerate(TOPICS)}

STYLES = [
    "terse_frustrated",
    "polite_formal",
    "casual_chatty",
    "urgent_caps",
    "apologetic_confused",
]
N_STYLES = len(STYLES)

CONTENT_TEMPLATES = {
    "billing": [
        "I was charged twice for {product} this month.",
        "The invoice amount doesn't match what I agreed to pay for {product}.",
        "My card was billed {amount} but I never confirmed that purchase.",
        "I need a copy of the receipt for my last payment on {product}.",
        "There's an unexpected charge on my statement labeled {product}.",
        "Can someone explain why my monthly bill for {product} went up to {amount}?",
        "My {product} plan renewed at a higher price than I was quoted.",
    ],
    "technical": [
        "{product} keeps crashing every time I try to open it.",
        "I'm getting error code {code} whenever I use {product}.",
        "{product} won't sync across my devices anymore.",
        "The dashboard for {product} is stuck loading and never finishes.",
        "I can't upload files in {product}, it just spins forever.",
        "After the last update {product} stopped responding entirely.",
        "{product} throws error {code} on every single launch.",
    ],
    "account": [
        "I can't log into {product}, it says my password is wrong.",
        "I need to change the email address linked to my {product} account.",
        "My account for {product} got merged with someone else's by mistake.",
        "I want to update the username on my {product} profile.",
        "My account for {product} shows the wrong subscription tier.",
        "I forgot the security answers for my {product} account.",
        "Can you transfer my {product} account to a new email address?",
    ],
    "refund": [
        "I cancelled {product} last week and still haven't received my refund.",
        "I want my money back for {product}, it doesn't work as advertised.",
        "Please reverse the charge of {amount} for {product}, I never used it.",
        "How long does a refund for {product} usually take to process?",
        "I was double-charged {amount} for {product} and need one refunded.",
        "I'm requesting a full refund because {product} was cancelled by you.",
        "Refund status for {product} still shows pending after two weeks.",
    ],
    "security": [
        "I think someone else logged into my {product} account without permission.",
        "I received a suspicious email claiming to be from {product} support.",
        "Can you enable two-factor authentication for my {product} account?",
        "There was a login to {product} from a country I've never visited.",
        "I'm worried my {product} password was leaked in a data breach.",
        "Someone changed my {product} account settings without my knowledge.",
        "I got a login alert for {product} that wasn't me at all.",
    ],
    "feature_request": [
        "It would be great if {product} supported dark mode.",
        "Could you add an export-to-CSV option in {product}?",
        "I'd love to see keyboard shortcuts added to {product}.",
        "Please consider adding an offline mode for {product}.",
        "Any plans to let {product} integrate with third-party calendars?",
        "It would help a lot if {product} had a bulk-edit feature.",
        "Could {product} get a customizable notification schedule?",
    ],
}

PRODUCTS = [
    "the mobile app",
    "the web dashboard",
    "my subscription",
    "the desktop client",
    "the Pro plan",
    "the API",
    "the browser extension",
    "the analytics module",
    "the team workspace",
    "the billing portal",
]
AMOUNTS = ["$49.99", "$12.00", "$199.00", "$9.99", "$25.50", "$74.00", "$5.00"]
CODES = ["500", "403", "E-1042", "timeout-07", "auth-failed", "E-2210", "503"]

STYLE_WRAPPERS = {
    "terse_frustrated": {
        "prefixes": ["", "Seriously.", "Look,"],
        "suffixes": [" This is ridiculous.", " Fix this now.", " I'm losing patience."],
    },
    "polite_formal": {
        "prefixes": ["Dear support team,", "Hello,", "Good afternoon,"],
        "suffixes": [
            " Thank you for your time.",
            " I would appreciate your help.",
            " Kind regards.",
        ],
    },
    "casual_chatty": {
        "prefixes": ["Hey there!", "So um,", "Quick question —"],
        "suffixes": [
            " Thanks a bunch!",
            " Let me know what you think :)",
            " No rush though.",
        ],
    },
    "urgent_caps": {
        "prefixes": ["URGENT:", "PLEASE READ ASAP:", "This can't wait:"],
        "suffixes": [
            " I NEED THIS FIXED TODAY.",
            " PLEASE RESPOND IMMEDIATELY.",
            " THIS IS TIME SENSITIVE.",
        ],
    },
    "apologetic_confused": {
        "prefixes": [
            "Sorry to bother you,",
            "I might be missing something, but",
            "Not sure if this is the right place,",
        ],
        "suffixes": [
            " Sorry if this is a silly question.",
            " I'm a bit confused about all this.",
            " Apologies for the trouble.",
        ],
    },
}


def render_situation(topic: str, style: str, rng: np.random.Generator) -> str:
    clause = rng.choice(CONTENT_TEMPLATES[topic])
    clause = clause.format(
        product=rng.choice(PRODUCTS),
        amount=rng.choice(AMOUNTS),
        code=rng.choice(CODES),
    )
    wrapper = STYLE_WRAPPERS[style]
    prefix = rng.choice(wrapper["prefixes"])
    suffix = rng.choice(wrapper["suffixes"])
    parts = [p for p in (prefix, clause) if p]
    text = " ".join(parts) + suffix
    return text.strip()


def build_action_table(mode: int, rng: np.random.Generator) -> np.ndarray:
    if mode == 1:
        return np.tile(np.arange(N_TOPICS), (N_TOPICS, 1))

    elif mode == 2:
        F = np.zeros((N_TOPICS, N_TOPICS), dtype=int)

        for j in range(N_TOPICS):
            F[j, j] = j
            other_sources = [i for i in range(N_TOPICS) if i != j]
            other_actions = rng.permutation([a for a in range(N_TOPICS) if a != j])

            for i, a in zip(other_sources, other_actions):
                F[i, j] = a

        return F

    elif mode == 3:
        gaps = range(-(N_TOPICS - 1), N_TOPICS)
        g = {d: int(rng.integers(0, N_TOPICS)) for d in gaps if d != 0}
        F = np.zeros((N_TOPICS, N_TOPICS), dtype=int)

        for i in range(N_TOPICS):
            for j in range(N_TOPICS):
                F[i, j] = j if i == j else g[j - i]

        return F

    else:
        raise ValueError(f"unknown mode {mode}")
