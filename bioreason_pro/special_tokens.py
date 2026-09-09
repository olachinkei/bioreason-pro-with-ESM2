"""The two — and only two — added special tokens (verified against bioreason2/models/special_tokens.py).

Modality *boundaries* are handled as natural language in the chat template
("Protein: <|protein_pad|>", "GO graph: <|go_graph_pad|>"); there are NO <dna_start>-style
delimiter tokens. Fusion injects encoder embeddings at these pad positions via scatter-add.
"""

PROTEIN_PAD_TOKEN = "<|protein_pad|>"
GO_GRAPH_PAD_TOKEN = "<|go_graph_pad|>"

ALL_SPECIAL_TOKENS = [PROTEIN_PAD_TOKEN, GO_GRAPH_PAD_TOKEN]

SPECIAL_TOKENS = {"protein_pad": PROTEIN_PAD_TOKEN, "go_graph_pad": GO_GRAPH_PAD_TOKEN}


def get_token(token_name: str) -> str:
    """Map 'protein_pad'/'go_graph_pad' to their token strings (mirrors bioreason2)."""
    return SPECIAL_TOKENS[token_name]


def add_special_tokens(tokenizer):
    """Add the two pad tokens; caller must then `model.resize_token_embeddings(len(tokenizer))`.

    Returns dict of token_id per token for locating scatter-add positions in `input_ids`.
    """
    tokenizer.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})
    return {t: tokenizer.convert_tokens_to_ids(t) for t in ALL_SPECIAL_TOKENS}
