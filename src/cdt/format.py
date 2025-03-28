"""Format annotation results into structured dataframes for evaluation."""

import uuid
from collections import defaultdict

import pandas as pd
import spacy
from bs4 import BeautifulSoup, NavigableString

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])


def get_token_indices(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    """Get token start and end indices for a span of text."""
    doc = nlp(text)
    token_start = None
    token_end = None

    curr_char = 0
    for i, token in enumerate(doc):
        token_chars = len(token.text_with_ws)
        if curr_char <= char_start < curr_char + token_chars and token_start is None:
            token_start = i
        if curr_char <= char_end <= curr_char + token_chars:
            token_end = i + 1
            break
        curr_char += token_chars

    return token_start or 0, token_end or len(doc)


def get_text_indices(
    tag: NavigableString, soup: BeautifulSoup
) -> tuple[int, int, int, int]:
    """Get character and token indices of tag's text in the original text."""
    # Get all text nodes before this tag
    text_before = ""
    current = tag
    while current.previous_sibling:
        current = current.previous_sibling
        if isinstance(current, NavigableString):
            text_before = current + text_before
        else:
            text_before = current.get_text() + text_before

    # Add text from parent tags before this one
    parent = tag.parent
    while parent:
        current = parent
        while current.previous_sibling:
            current = current.previous_sibling
            if isinstance(current, NavigableString):
                text_before = current + text_before
            else:
                text_before = current.get_text() + text_before
        parent = parent.parent

    # Get character indices
    start = len(text_before)
    end = start + len(tag.get_text())

    # Get token indices
    tok_start, tok_end = get_token_indices(soup.get_text(), start, end)

    return start, end, tok_start, tok_end


def get_coreference_map(soup: BeautifulSoup) -> dict:
    """Create a mapping of coreference IDs to their corresponding tag contents."""
    coreference_map = defaultdict(list)

    for tag in soup.find_all(attrs={"coreference": True}):
        char_start, char_end, tok_start, tok_end = get_text_indices(tag, soup)
        coref_ids = tag["coreference"].split()
        for coref_id in coref_ids:
            coreference_map[coref_id].append(
                {
                    "text": tag.text,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_start": tok_start,
                    "token_end": tok_end,
                }
            )
    return dict(coreference_map)


def extract_entity_properties(soup: BeautifulSoup, entity_type: str) -> dict:
    """Extract properties for entities of a given type from annotated text."""
    # First pass: collect all properties and group by entity/property/coref
    property_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    entity_properties = defaultdict(dict)

    # Initialize entities with UUID and ID
    for tag in soup.find_all(attrs={entity_type: True}):
        for entity_id in tag[entity_type].split():
            if entity_id not in entity_properties:
                entity_properties[entity_id] = {
                    "uuid": str(uuid.uuid4()),
                    f"{entity_type}_id": entity_id,
                }

    # Group properties by entity_id, property_name, and coref_id
    for tag in soup.find_all(attrs={entity_type: True}):
        entity_ids = tag[entity_type].split()
        property_name = tag.name
        property_value = tag.text
        char_start, char_end, tok_start, tok_end = get_text_indices(tag, soup)
        coref_id = tag.get(
            "coreference", "__NO_COREF__"
        )  # Special value for no coreference

        for entity_id in entity_ids:
            property_groups[entity_id][property_name][coref_id].append(
                {
                    "value": property_value,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_start": tok_start,
                    "token_end": tok_end,
                    "coref_id": coref_id if coref_id != "__NO_COREF__" else None,
                }
            )

            # Copy any additional attributes
            for attr, value in tag.attrs.items():
                if attr not in {entity_type, "coreference"}:
                    entity_properties[entity_id][f"{property_name}_{attr}"] = value

    # Second pass: consolidate properties based on coreference groups
    for entity_id, properties in property_groups.items():
        for property_name, coref_groups in properties.items():
            # If only one coreference group (or no coreferences), take first value
            if len(coref_groups) == 1:
                values = next(iter(coref_groups.values()))
                first_value = values[0]
                entity_properties[entity_id].update(
                    {
                        property_name: first_value["value"],
                        f"{property_name}_char_start": first_value["char_start"],
                        f"{property_name}_char_end": first_value["char_end"],
                        f"{property_name}_token_start": first_value["token_start"],
                        f"{property_name}_token_end": first_value["token_end"],
                    }
                )
                if first_value["coref_id"]:
                    entity_properties[entity_id][f"{property_name}_coreferences"] = (
                        first_value["coref_id"]
                    )

            # If multiple coreference groups, keep all values as lists
            else:
                all_values = []
                char_starts = []
                char_ends = []
                token_starts = []
                token_ends = []
                coref_ids = []

                for coref_id, values in coref_groups.items():
                    for v in values:
                        all_values.append(v["value"])
                        char_starts.append(v["char_start"])
                        char_ends.append(v["char_end"])
                        token_starts.append(v["token_start"])
                        token_ends.append(v["token_end"])
                        if v["coref_id"]:
                            coref_ids.append(v["coref_id"])

                entity_properties[entity_id].update(
                    {
                        property_name: all_values,
                        f"{property_name}_char_start": char_starts,
                        f"{property_name}_char_end": char_ends,
                        f"{property_name}_token_start": token_starts,
                        f"{property_name}_token_end": token_ends,
                    }
                )
                if coref_ids:
                    entity_properties[entity_id][f"{property_name}_coreferences"] = (
                        coref_ids
                    )

    return entity_properties


def extract_instrument_properties(soup: BeautifulSoup) -> dict:
    """Extract properties of debt instruments from the annotated text."""
    return extract_entity_properties(soup, "instrument")


def extract_agreements(soup: BeautifulSoup) -> dict:
    """Extract agreement properties from the annotated text."""
    return extract_entity_properties(soup, "agreement")


def convert_to_dataframes(
    instrument_properties: dict, agreement_properties: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert extracted properties into dataframes."""
    instruments_df = pd.DataFrame(list(instrument_properties.values()))
    agreements_df = pd.DataFrame(list(agreement_properties.values()))

    # Convert any list columns to strings
    for df in [instruments_df, agreements_df]:
        for col in df.columns:
            if df[col].apply(type).eq(list).any():
                df[col] = df[col].apply(
                    lambda x: ", ".join(str(y) for y in x) if isinstance(x, list) else x
                )

    return instruments_df, agreements_df


def parse_annotated_text(
    text: str, text_id: str = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse the annotated text and extract information about debt instruments and agreements.

    Args:
        text (str): The annotated text.
        text_id (str, optional): ID of the text being processed.

    Returns:
        tuple: Three items - instruments DataFrame, agreements DataFrame, and coreference map.
    """
    soup = BeautifulSoup(text, "html.parser")

    # Get coreference mappings with modified IDs if text_id provided
    coreference_map = get_coreference_map(soup)
    if text_id:
        coreference_map = {f"{text_id}-REF-{k}": v for k, v in coreference_map.items()}

    # Extract instrument properties
    instrument_properties = extract_instrument_properties(soup)
    if text_id:
        # Update coreference IDs in properties
        for props in instrument_properties.values():
            for k in list(props.keys()):
                if k.endswith("_coreferences"):
                    props[k] = f"{text_id}-REF-{props[k]}"
            props["text_id"] = text_id

    # Extract agreements
    agreement_properties = extract_agreements(soup)
    if text_id:
        for agreement in agreement_properties.values():
            agreement["text_id"] = text_id

    # Convert to dataframes
    instruments_df, agreements_df = convert_to_dataframes(
        instrument_properties, agreement_properties
    )

    return instruments_df, agreements_df, coreference_map


def format_llm_outputs(results_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Process LLM outputs and convert to structured format.

    Args:
        results_path (str): Path to CSV file containing LLM outputs.
            Must have columns: input_text_id, input_text, response

    Returns:
        tuple: Combined instruments DataFrame, agreements DataFrame, and coreference map
    """
    results_df = pd.read_csv(results_path)
    all_instruments = []
    all_agreements = []
    all_coreferences = {}

    for _, row in results_df.iterrows():
        text_id = row["input_text_id"]
        response = row["response"]

        # Parse the response
        instruments_df, agreements_df, coref_map = parse_annotated_text(
            response, text_id=text_id
        )

        all_instruments.append(instruments_df)
        all_agreements.append(agreements_df)
        all_coreferences.update(coref_map)

    # Combine results
    combined_instruments = (
        pd.concat(all_instruments, ignore_index=True)
        if all_instruments
        else pd.DataFrame()
    )
    combined_agreements = (
        pd.concat(all_agreements, ignore_index=True)
        if all_agreements
        else pd.DataFrame()
    )

    return combined_instruments, combined_agreements, all_coreferences


def main() -> None:
    """Example usage of the parse_annotated_text function."""
    annotated_text = """
    <meta confidence='high'/>On <start_date instrument="1" agreement="1">December 8th, 2023</start_date>, <borrower instrument="1 2">OpenAI</borrower> entered in a <name instrument="1" type="loan">term loan</name> for <amount instrument="1">$56 trillion</amount> with <lender instrument="1">SoftBank</lender>, <lender instrument="1">JPMorgan</lender>, <lender instrument="1">Citibank</lender>, and <lender instrument="1">other lenders</lender> thereto. The loan will be due in <duration instrument="1">2 years</duration> and will pay interest at <interest_rate instrument="1">LIBOR plus the greater of: (i) one tenth of one percent, (ii) one percent divided by the last two digits of the current year, or (iii) sixty minus the total points scored in the last Super Bowl</interest_rate>. The loan will be used <purpose instrument="1">to pay off the outstanding debt from the <amount instrument="2">$6 million</amount> <name instrument="2" type="loan">2021 Term Loan</name> dated <start_date instrument="2">August 8, 2021</start_date></purpose>.
    """
    instruments_df, agreements_df, coreference_map = parse_annotated_text(
        annotated_text
    )
    print(instruments_df)
    print(agreements_df)
    print("Coreference map:", coreference_map)

    annotated_text_2 = """
<meta confidence='medium'/>The <name instrument="1" type="bond" coreference="a">2025 Corporate Bond</name> issued by <borrower instrument="1" coreference="b">TechCorp</borrower> was underwritten by <underwriter instrument="1">Goldman Sachs</underwriter>. The bond matures on <end_date instrument="1">December 31, 2025</end_date> and has an interest rate of <interest_rate instrument="1">5.5%</interest_rate>. 

Later in the document, <borrower instrument="1" coreference="b">the Company</borrower> announced that the <name instrument="1" coreference="a">Bond</name> will be used to fund <purpose instrument="1">new research and development projects</purpose>. The <name agreement="1" coreference="c">Underwriting Agreement</name> governing this bond was signed on <start_date agreement="1">January 15, 2023</start_date>.

Additionally, <borrower instrument="1" coreference="b">TechCorp</borrower> confirmed that the <name agreement="1" coreference="c">Agreement</name> includes provisions for early repayment.
    """
    instruments_df_2, agreements_df_2, coreference_map_2 = parse_annotated_text(
        annotated_text_2, text_id="doc2"
    )
    print("--------------------------------")
    print(instruments_df_2)
    print(agreements_df_2)
    print("Coreference map:", coreference_map_2)


if __name__ == "__main__":
    main()
