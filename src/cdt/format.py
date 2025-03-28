import pandas as pd
import uuid
from bs4 import BeautifulSoup, NavigableString
from collections import defaultdict


def get_text_indices(tag, soup):
    """Get the start and end character indices of tag's text in the original text."""
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

    start = len(text_before)
    end = start + len(tag.get_text())
    return start, end


def get_coreference_map(soup):
    """
    Create a mapping of coreference IDs to their corresponding tag contents.

    Args:
        soup (BeautifulSoup): Parsed HTML of the annotated text.

    Returns:
        dict: A dictionary mapping coreference IDs to lists of tag information.
    """
    coreference_map = defaultdict(list)
    for tag in soup.find_all(attrs={"coreference": True}):
        start, end = get_text_indices(tag, soup)
        coref_ids = tag["coreference"].split()
        for coref_id in coref_ids:
            coreference_map[coref_id].append(
                {"text": tag.text, "start": start, "end": end}
            )
    return dict(coreference_map)


def extract_instrument_properties(soup):
    """
    Extract properties of debt instruments from the annotated text.

    Args:
        soup (BeautifulSoup): Parsed HTML of the annotated text.

    Returns:
        dict: A dictionary of instrument properties keyed by instrument ID.
    """
    instrument_properties = defaultdict(dict)

    for instrument_tag in soup.find_all(attrs={"instrument": True}):
        instrument_ids = instrument_tag["instrument"].split()
        property_name = instrument_tag.name
        property_value = instrument_tag.text
        start, end = get_text_indices(instrument_tag, soup)
        coref_id = instrument_tag.get("coreference")

        for instrument_id in instrument_ids:
            if instrument_id not in instrument_properties:
                instrument_properties[instrument_id] = {
                    "uuid": str(uuid.uuid4()),
                    "instrument_id": instrument_id,
                }

            if property_name == "name":
                instrument_properties[instrument_id]["name"] = property_value
                instrument_properties[instrument_id]["name_start"] = start
                instrument_properties[instrument_id]["name_end"] = end
                instrument_properties[instrument_id]["type"] = instrument_tag.get(
                    "type"
                )
            elif property_name == "lender":
                if "lender" not in instrument_properties[instrument_id]:
                    instrument_properties[instrument_id]["lender"] = []
                    instrument_properties[instrument_id]["lender_start"] = []
                    instrument_properties[instrument_id]["lender_end"] = []
                instrument_properties[instrument_id]["lender"].append(property_value)
                instrument_properties[instrument_id]["lender_start"].append(start)
                instrument_properties[instrument_id]["lender_end"].append(end)
            else:
                instrument_properties[instrument_id][property_name] = property_value
                instrument_properties[instrument_id][f"{property_name}_start"] = start
                instrument_properties[instrument_id][f"{property_name}_end"] = end

            # Store coreference ID if present
            if coref_id:
                instrument_properties[instrument_id][
                    f"{property_name}_coreferences"
                ] = coref_id

            if instrument_tag.get("agreement"):
                instrument_properties[instrument_id]["agreement"] = instrument_tag.get(
                    "agreement"
                )

    return instrument_properties


def extract_agreements(soup):
    """
    Extract agreements from the annotated text.

    Args:
        soup (BeautifulSoup): Parsed HTML of the annotated text.

    Returns:
        list: A list of dictionaries containing agreement data.
    """
    agreements = []
    for agreement_tag in soup.find_all(attrs={"agreement": True}):
        agreement_id = agreement_tag["agreement"]
        agreement_data = {
            "uuid": str(uuid.uuid4()),
            "agreement_id": agreement_id,
            "name": agreement_tag.text if agreement_tag.name == "name" else None,
            "governs": agreement_tag.get("governs", None),
        }
        agreements.append(agreement_data)
    return agreements


def convert_to_dataframes(instrument_properties, agreements):
    """
    Convert extracted instrument properties and agreements into dataframes.

    Args:
        instrument_properties (dict): Dictionary of instrument properties.
        agreements (list): List of agreement data.

    Returns:
        tuple: Two pandas DataFrames, one for instruments and one for agreements.
    """
    instruments_df = pd.DataFrame(list(instrument_properties.values()))
    agreements_df = pd.DataFrame(agreements).drop_duplicates(subset=["agreement_id"])

    # Convert lender and index lists to strings
    if "lender" in instruments_df.columns:
        instruments_df["lender"] = instruments_df["lender"].apply(
            lambda x: ", ".join(str(y) for y in x) if isinstance(x, list) else x
        )
        instruments_df["lender_start"] = instruments_df["lender_start"].apply(
            lambda x: ", ".join(str(y) for y in x) if isinstance(x, list) else x
        )
        instruments_df["lender_end"] = instruments_df["lender_end"].apply(
            lambda x: ", ".join(str(y) for y in x) if isinstance(x, list) else x
        )

    return instruments_df, agreements_df


def parse_annotated_text(text):
    """
    Parse the annotated text and extract information about debt instruments and agreements.

    Args:
        text (str): The annotated text.

    Returns:
        tuple: Three items - instruments DataFrame, agreements DataFrame, and coreference map.
    """
    soup = BeautifulSoup(text, "html.parser")

    # Get coreference mappings
    coreference_map = get_coreference_map(soup)

    # Extract instrument properties
    instrument_properties = extract_instrument_properties(soup)

    # Extract agreements
    agreements = extract_agreements(soup)

    # Convert to dataframes
    instruments_df, agreements_df = convert_to_dataframes(
        instrument_properties, agreements
    )

    return instruments_df, agreements_df, coreference_map


def main():
    """
    Example usage of the parse_annotated_text function.
    """
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
        annotated_text_2
    )
    print("--------------------------------")
    print(instruments_df_2)
    print(agreements_df_2)
    print("Coreference map:", coreference_map_2)


if __name__ == "__main__":
    main()
