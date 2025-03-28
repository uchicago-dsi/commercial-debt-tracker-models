"""Code for evaluating model outputs for agreement and debt instrument extraction

Compares model outputs in with the following components:
- agreements table (csv/dataframe)
- debt instruments table (csv/dataframe)
- texts table (csv/dataframe)
- coreference map (json)

Agreements and Debt Instruments are referred to as entity types and each row of their
respective tables represents a single instance of that entity type. The columns in these
tables represent various attributes of the entities. The attributes are extracted from
input texts that have been tagged by humans or models. The tags are derived in multiple
ways:
    1. Tagged Properties. These are tags with in the text that have either either an
        'instrument' or 'agreement' attribute. All tags in a given text with the same
        value for the 'instrument' or 'agreement' attribute are will appear in the
        same row of either the agreements or debt instruments table, respectively.
        Each tagged property will have 5 or 6 columns in the table. 1 column that
        is just the tag name (e.g., 'name', 'lender', etc.), and 2 columns each
        for the indices of the text span in characters and tokens (i.e.
        'name_token_start', 'name_token_end', 'name_char_start', 'name_char_end')
        A fourth optional column is 'coreference' which maps to a unique identifier for
        that span. This allows for multiple mentions of the same property instance to
        be linked together.
    2. Attribute-Based Properties: These are properties that are derived from the
        attributes of the tags in the text. Specifically 'type' for debt instruments
        and relations (like 'amendment_of', 'split_of', etc.). In the agreements and
        debt instruments tables, these properties will be column names with values
        being the attribute values (replaced with updated ids for relations)
    3. Metadata: Each agreement or debt instrument will include the id of the text
        they appeared in (text_id)

The texts table minimally has 'id', 'text', and 'tokens' columns. The 'id' column is used to link
the texts to the agreements and debt instruments tables. Each text contains the raw
text that was tagged.

The coreference map is a dictionary that maps coreference IDs to their each mention of
the same property instance in the text. Each mention is represented as a tuple with
text, start_index, end_index.

The evaluation process involves:
"""

import pandas as pd


def calculate_iou(span1: tuple[int, int], span2: tuple[int, int]) -> float:
    """Calculate the Intersection over Union (IoU) of two spans.

    Args:
        span1 (tuple): A tuple (start, end) representing the first span.
        span2 (tuple): A tuple (start, end) representing the second span.

    Returns:
        float: The IoU value.
    """
    start1, end1 = span1
    start2, end2 = span2

    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)

    return intersection / union if union > 0 else 0.0


def compare_coreference_clusters(
    cluster1: list[dict],
    cluster2: list[dict],
    min_acceptable_iou: int = 0.5,
    method: str = "char",
) -> bool:
    """True if any pair of mentions from two coreference clusters exceed threshold

    Args:
        cluster1: A list of dictionaries with 'start' and 'end' keys for the first cluster.
        cluster2: A list of dictionaries with 'start' and 'end' keys for the second cluster.
        min_acceptable_iou (float): The minimum IoU threshold for a match.
        method: Either 'char' or 'token'

    Returns:
        bool: True if any mention pair meets the IoU threshold, False otherwise.
    """
    for mention1 in cluster1:
        for mention2 in cluster2:
            iou = calculate_iou(
                (mention1[f"{method}_start"], mention1[f"{method}_end"]),
                (mention2[f"{method}_start"], mention2[f"{method}_end"]),
            )
            if iou >= min_acceptable_iou:
                return True
    return False


def get_property_mention_spans_by_entity(
    result_row: pd.Series,
    property_name: str,
    coreferences: dict[str, tuple[str, int, int]],
) -> set[tuple[str, int, int]]:
    """Get spans for all mentions of a given property instance from entity row

    Args:
        result_row (pd.Series): A row from the DataFrame containing entity spans.
        property_name (str): The name of the property to extract spans for.
        coreferences (dict): A dictionary mapping coreference IDs to their spans.

    Returns:
        list: A set of tuples representing the spans (text, start, end) for the specified property.
    """
    spans = set()
    # Check if the property exists in the row
    if (
        f"{property_name}_char_start" in result_row
        and f"{property_name}_char_end" in result_row
        and f"{property_name}_token_start" in result_row
        and f"{property_name}_token_end" in result_row
        and property_name in result_row
    ):
        char_start = result_row[f"{property_name}_char_start"]
        char_end = result_row[f"{property_name}_char_end"]
        token_start = result_row[f"{property_name}_token_start"]
        token_end = result_row[f"{property_name}_token_end"]
        text = result_row[property_name]
        if f"{property_name}_coreferences" in result_row:
            coref_id = result_row[f"{property_name}_coreferences"]
            spans = spans.union(coreferences.get(coref_id, set()))
        spans.append((text, char_start, char_end, token_start, token_end))
    return spans


def get_entity_tags_for_text(
    text_id: str,
    entity_df: pd.DataFrame,
    coreference_map: dict,
) -> list[tuple[str, int, int, int, int]]:
    """Get all property tags for entities associated with a text.

    Args:
        text_id (str): ID of the text to get tags for
        entity_df (pd.DataFrame): DataFrame containing entity data (agreements/instruments)
        coreference_map (Dict): Dictionary mapping coreference IDs to mentions

    Returns:
        List[Tuple[str, int, int, int, int]]: List of tuples containing:
            (property_name, char_start, char_end, token_start, token_end)
    """
    tags = []
    rows = entity_df[entity_df["text_id"] == text_id]

    for _, row in rows.iterrows():
        # Get property columns (those with _char_start suffix)
        prop_cols = [col[:-11] for col in row.index if col.endswith("_char_start")]

        for prop in prop_cols:
            spans = get_property_mention_spans_by_entity(row, prop, coreference_map)
            for span in spans:
                tags.append(
                    (prop, span[0], span[1], span[3], span[4])
                )  # (property_name, text, char_start, token_start, token_end)

    return tags


def create_seqeval_tags(
    text_id: str,
    num_tokens: int,
    instruments_df: pd.DataFrame,
    agreements_df: pd.DataFrame,
    coreference_map: dict,
) -> list[str]:
    """Create seqeval format tags for a document.

    Args:
        text_id (str): ID of the text to create tags for
        num_tokens (int): Number of tokens in the text
        instruments_df (pd.DataFrame): DataFrame containing instrument data
        agreements_df (pd.DataFrame): DataFrame containing agreement data
        coreference_map (Dict): Dictionary mapping coreference IDs to mentions

    Returns:
        List[str]: List of BIO tags, one per token
    """
    # Initialize all tokens as 'O'
    tags = ["O"] * num_tokens

    # Get all entity tags
    instrument_tags = get_entity_tags_for_text(
        text_id, instruments_df, coreference_map, "instrument"
    )
    agreement_tags = get_entity_tags_for_text(
        text_id, agreements_df, coreference_map, "agreement"
    )
    all_tags = sorted(
        instrument_tags + agreement_tags, key=lambda x: x[3]
    )  # Sort by token_start

    # Apply tags
    for tag_type, _, _, tok_start, tok_end in all_tags:
        # Set beginning tag
        if tok_start < len(tags):
            tags[tok_start] = f"B-{tag_type}"

        # Set inside tags
        for i in range(tok_start + 1, min(tok_end, len(tags))):
            tags[i] = f"I-{tag_type}"

    return tags


def convert_dataset_to_seqeval(
    texts_df: pd.DataFrame,
    instruments_df: pd.DataFrame,
    agreements_df: pd.DataFrame,
    coreference_map: dict,
) -> dict[str, list[list[str]]]:
    """Convert entire dataset to seqeval format.

    Args:
        texts_df (pd.DataFrame): DataFrame containing texts with 'id' and 'tokens' columns
        instruments_df (pd.DataFrame): DataFrame containing instrument data
        agreements_df (pd.DataFrame): DataFrame containing agreement data
        coreference_map (Dict): Dictionary mapping coreference IDs to mentions

    Returns:
        Dict[str, List[List[str]]]: Dictionary with keys:
            - tags: List of tag sequences
            - tokens: List of token sequences
    """
    all_tags = []
    all_tokens = []

    for _, row in texts_df.iterrows():
        text_id = row["id"]
        tokens = row["tokens"]
        num_tokens = len(tokens)

        tags = create_seqeval_tags(
            text_id, num_tokens, instruments_df, agreements_df, coreference_map
        )

        all_tags.append(tags)
        all_tokens.append(tokens)

    return {"tags": all_tags, "tokens": all_tokens}
