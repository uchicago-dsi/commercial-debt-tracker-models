import pandas as pd
import uuid
from bs4 import BeautifulSoup
from collections import defaultdict


def parse_annotated_text(text):
    """
    Parse the annotated text and extract information about debt instruments and agreements.

    Args:
        text (str): The annotated text.

    Returns:
        tuple: Two dataframes, one for debt instruments and one for agreements.
    """
    soup = BeautifulSoup(text, "html.parser")

    # Initialize dictionaries to collect properties for each instrument
    instrument_properties = defaultdict(dict)
    agreements = []

    # Extract debt instruments
    for instrument_tag in soup.find_all(attrs={"instrument": True}):
        instrument_ids = instrument_tag["instrument"].split()

        # Determine the property value based on the tag
        property_value = None
        if instrument_tag.name == "name":
            property_name = "name"
            property_value = {
                "value": instrument_tag.text,
                "type": instrument_tag.get("type", None),
            }
        else:
            property_name = instrument_tag.name
            property_value = instrument_tag.text

        # Assign the property to each instrument ID
        for instrument_id in instrument_ids:
            if instrument_id not in instrument_properties:
                instrument_properties[instrument_id] = {
                    "uuid": str(uuid.uuid4()),
                    "instrument_id": instrument_id,
                }

            if property_name == "name":
                instrument_properties[instrument_id]["name"] = property_value["value"]
                instrument_properties[instrument_id]["type"] = property_value["type"]
            elif property_name == "lender":
                # Combine multiple lenders into a list
                if "lender" in instrument_properties[instrument_id]:
                    if isinstance(instrument_properties[instrument_id]["lender"], list):
                        instrument_properties[instrument_id]["lender"].append(
                            property_value
                        )
                    else:
                        instrument_properties[instrument_id]["lender"] = [
                            instrument_properties[instrument_id]["lender"],
                            property_value,
                        ]
                else:
                    instrument_properties[instrument_id]["lender"] = [property_value]
            else:
                instrument_properties[instrument_id][property_name] = property_value

            # Add agreement reference if present
            if instrument_tag.get("agreement"):
                instrument_properties[instrument_id]["agreement"] = instrument_tag.get(
                    "agreement"
                )

    # Extract agreements
    for agreement_tag in soup.find_all(attrs={"agreement": True}):
        agreement_id = agreement_tag["agreement"]
        agreement_data = {
            "uuid": str(uuid.uuid4()),
            "agreement_id": agreement_id,
            "name": agreement_tag.text if agreement_tag.name == "name" else None,
            "governs": agreement_tag.get("governs", None),
        }
        agreements.append(agreement_data)

    # Convert to dataframes
    instruments_df = pd.DataFrame(list(instrument_properties.values()))
    agreements_df = pd.DataFrame(agreements).drop_duplicates(subset=["agreement_id"])

    # Convert lender lists to strings
    if "lender" in instruments_df.columns:
        instruments_df["lender"] = instruments_df["lender"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    return instruments_df, agreements_df


def main():
    # Example usage
    annotated_text = """
    <meta confidence='high'/>On <start_date instrument="1" agreement="1">December 8th, 2023</start_date>, <borrower instrument="1 2">OpenAI</borrower> entered in a <name instrument="1" type="loan">term loan</name> for <amount instrument="1">$56 trillion</amount> with <lender instrument="1">SoftBank</lender>, <lender instrument="1">JPMorgan</lender>, <lender instrument="1">Citibank</lender>, and <lender instrument="1">other lenders</lender> thereto. The loan will be due in <duration instrument="1">2 years</duration> and will pay interest at <interest_rate instrument="1">LIBOR plus the greater of: (i) one tenth of one percent, (ii) one percent divided by the last two digits of the current year, or (iii) sixty minus the total points scored in the last Super Bowl</interest_rate>. The loan will be used <purpose instrument="1">to pay off the outstanding debt from the <amount instrument="2">$6 million</amount> <name instrument="2" type="loan">2021 Term Loan</name> dated <start_date instrument="2">August 8, 2021</start_date></purpose>.
    """
    instruments_df, agreements_df = parse_annotated_text(annotated_text)
    print(instruments_df)
    print(agreements_df)


if __name__ == "__main__":
    main()
