"""Post-processing module for the LLM model output."""

import re

from bs4 import BeautifulSoup

VALID_TAGS = [
    "name",
    "administrative_agent",
    "underwriter",
    "other_related_party",
    "start_date",
    "end_date",
    "duration",
    "purpose",
    "borrower",
    "lender",
    "amount",
    "interest_rate",
    "meta",
]

VALID_ATTRIBUTES = {
    "instrument": r"^\d+(\s\d+)*$",
    "coreference": r"^[a-z]+(\s[a-z]+)*$",
    "agreement": r"^\d+$",
    "amendment_of": r"^\d+$",
    "split_of": r"^\d+$",
    "governed_by": r"^\d+$",
    "governs": r"^\d+(\s\d+)*$",
    "type": r"^(loan|bond|credit line|revolving credit)$",
    "confidence": r"^(low|medium|high)$",
}


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", text).strip()


def strip_common_additions(text: str) -> str:
    r"""Remove common artifacts added by LLMs.

    Cases covered:
     - Removes all text before (and including) any "</think>" tag.
     - If there is a markdown code block (i.e. ```xml\nCode...```), it
        strips everything except the contents of the code block.

    Args:
        text: The text to be cleaned.
    """
    # Remove all text before (and including) any </think> tag
    text = re.sub(r".*</think>", "", text, flags=re.DOTALL)

    # If there is a markdown code block, strip everything except the contents
    match = re.search(r"```xml\n(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1)

    return text.strip()


def standardize_llm_output(text: str) -> str:
    """Normalize whitespaces and strip common additions from LLM output.

    Args:
        text: The text to be cleaned.
    """
    text = normalize_whitespace(strip_common_additions(text))
    return text


def get_llm_output_errors(output_html: str, original_text: str) -> list[str]:
    """Validates the HTML output from the model.

    Requirements:
      - When all tags are stripped, the text should match original_text (ignoring whitespace differences).
      - Only tags in valid_tags should be present.
      - Each tag's attributes must be a subset of allowed_attrs for that tag.

    Args:
        output_html: The HTML output from the model.
        original_text: The original text used as input to the model.

    Returns:
        A list of error messages, or an empty list if the output is valid.
    """
    # Parse the output HTML
    soup = BeautifulSoup(output_html, "html.parser")

    # Check that the text content matches the original text (ignoring whitespace)
    cleaned_output_text = standardize_llm_output(soup.get_text())
    expected_text = normalize_whitespace(original_text)
    errors = []
    if cleaned_output_text != expected_text:
        errors.append(
            "Stripped text does not match the original text. Please ensure all text is preserved and no commentary is added."
        )

    # Check all tags and their attributes
    for tag in soup.find_all():
        if tag.name not in VALID_TAGS:
            errors.append(f"Found invalid tag: <{tag.name}>.")
        # For each tag, check that its attributes are allowed
        for attr, value in tag.attrs.items():
            if attr not in VALID_ATTRIBUTES:
                errors.append(
                    f"Tag <{tag.name}> contains disallowed attribute: '{attr}'."
                )
            elif not re.match(VALID_ATTRIBUTES[attr], value):
                errors.append(
                    f"Tag <{tag.name}> has invalid value for attribute '{attr}': '{value}'."
                )
    return errors
