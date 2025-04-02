"""Post-processing module for the LLM model output."""

import re
from difflib import ndiff

from bs4 import BeautifulSoup

VALID_TAGS = [
    "body",
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
    "instrument": r"^[a-zA-Z0-9]+(\s[a-zA-Z0-9]+)?$",
    "coreference": r"^[a-zA-Z0-9]+(\s[a-zA-Z0-9]+)?$", 
    "agreement": r"^[a-zA-Z0-9]+(\s[a-zA-Z0-9]+)?$",
    "amendment_of": r"^[a-zA-Z0-9]+(\s[a-zA-Z0-9]+)?$",
    "split_of": r"^[a-zA-Z0-9]+(\s[a-zA-Z0-9]+)?$",
    "governed_by": r"^[a-zA-Z0-9]+(\s[a-zA-Z0-9]+)?$",
    "governs": r"^[a-zA-Z0-9]+(\s[a-zA-Z0-9]+)?$",
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
    errors = []
    # find the html portion of the output based on <body> tags
    start = output_html.find("<body>")
    end = output_html.find("</body>")
    if start == -1 or end == -1:
        errors.append("Output does not contain a <body> tags indicating start and end of response. It is very important to include these tags.")
    else:
        output_html = output_html[start:end+len("</body>")]
    soup = BeautifulSoup(output_html, "html.parser")
    output_text_no_spaces = re.sub(r'\s+', '', soup.get_text())
    original_text_no_spaces = re.sub(r'\s+', '', original_text)
    # Check that the text content matches the original text (ignoring whitespace)
    if output_text_no_spaces != original_text_no_spaces:
        errors.append("Output text does not match original text.")

    # Check all tags and their attributes
    for tag in soup.find_all():
        if tag.name not in VALID_TAGS:
            errors.append(f"Found invalid tag: <{tag.name}>.")
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


def normalize_html_whitespace(output_html: str, original_text: str) -> str:
    """Reformat HTML to match original text's whitespace while preserving HTML tags.
    
    Args:
        output_html: HTML output with potentially incorrect whitespace
        original_text: Original text with the desired whitespace pattern
        
    Returns:
        HTML with tags preserved but whitespace matching the original text
    """
    soup = BeautifulSoup(output_html, "html.parser")
    
    output_text_no_space = re.sub(r'\s+', '', soup.get_text())
    original_no_spaces = re.sub(r'\s+', '', original_text)
    
    if output_text_no_space != original_no_spaces:
        raise ValueError("Text content does not match original text when spaces are removed.")
    
    # Match whitespace in output text to original text
    char_positions = {}
    text_pos = 0
    
    for i, char in enumerate(original_text):
        if char.isspace():
            continue
        while text_pos < len(output_text_no_space) and output_text_no_space[text_pos] != char:
            text_pos += 1
        if text_pos < len(output_text_no_space):
            char_positions[text_pos] = i
            text_pos += 1
    
    # Add HTML tags to the whitespace-matched output text
    text_index = 0
    for node in soup.descendants:
        if isinstance(node, str):
            node_text = node.strip()
            if not node_text:
                continue
                
            adjusted_text = ""
            for char in node:
                if not char.isspace():
                    # Find the position in original text
                    if text_index in char_positions:
                        orig_index = char_positions[text_index]
                        
                        # Add any whitespace that should precede this character
                        while orig_index > 0 and original_text[orig_index-1].isspace():
                            adjusted_text += " "
                            orig_index -= 1
                            
                    adjusted_text += char
                    text_index += 1
            
            # Replace the node with adjusted text
            node.replace_with(adjusted_text)
    
    # Convert back to string
    return str(soup)

def find_html_portion(output_html: str, original_text: str) -> str:
    """Find the HTML portion of the output that matches the original text.
    
    Args:
        output_html: HTML output with potentially incorrect whitespace
            or without <body> tags
        original_text: Original text with the desired whitespace pattern

    Returns:
        The HTML portion of the output that matches the original text.
    """
    original_text_no_spaces = re.sub(r'\s+', '', original_text)
    output_text = BeautifulSoup(output_html, "html.parser").get_text()
    output_text_no_spaces = re.sub(r'\s+', '', output_text)
    # check if output text is a substring of original text. If so, return
    # the output text (with spaces and whitespace and <body> tags added)
    start_index = output_text_no_spaces.find(original_text_no_spaces)
    if start_index != -1:
        # TODO
        pass
    else:
        return None
