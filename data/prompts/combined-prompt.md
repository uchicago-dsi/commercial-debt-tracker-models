**Prompt for Annotating Debt Instruments with Coreferences, Agreements, and Types**

You are an expert in corporate finance tasked with annotating pieces of text to identify and tag debt instruments, their properties, and related legal agreements. Your goal is to process an input text that describes one or more debt instruments and return the exact same text with inline tags added around specific property values. The properties you must identify and tag are:

- **name:** the name of either a debt instrument or agreement. We are only interested in agreements that involve debt instruments. Sometimes a formal name is not listed, in those cases tag whatever the entity is referred by in the text.
- **administrative_agent:** the name of a party responsible for administering the agreement or trustee.
- **underwriter:** the name of a party that is underwriting a debt security or, if there are multiple (also known as an underwriting syndicate), the representative(s) of the underwriters listed. Most agreements have no underwriter. 
- **other_related_party:** the name of any entity that is party to a relevant agreement, but their exact role is unlisted or unclear.
- **start_date:** the date the agreement was entered into or the debt instrument officially began and the transactions were initiated—for Notes, the Issuance Date. Will almost always be present for agreements, may be missing sometimes for debt instruments. If there are modifiers included (it is a planned or anticipated date, please include this):
    - Examples: “March 24, 2024”, “the first day of August 2026”, “tentatively scheduled for January 14, 2013”, “2023”, etc.
    - Non-examples: The date of a press release that announces the actual agreement or debt instrument.
- **end_date:** the date the agreement or debt instrument expires, ends, or matures; often mentioned as the “Maturity Date.” 
    - Examples: “March 24, 2024”, “the first day of August 2026”, etc.
- **duration:** the length of time the agreement or debt instrument will be active.
    - Examples: “2.5 years”, “6 months”, “364 days”, etc.
- **purpose:** the reason the borrower is entering into the agreement or utilizing the debt instrument. May be a property of either the agreement or a specific debt instrument (if it is explicitly tied to one)
- **borrower:** the company receiving or getting the right to receive capital through the debt instrument. This is almost always the company filing the 8-K and does not need to be labeled.
- **lender:** an entity transferring funds or making funds available to transfer to another entity. In the case of bonds/notes being offered to the public, the ‘public’ should be selected as the lender. 
- **amount:** the value of the transaction. As a rule, if the amount is mentioned multiple times and one is in the form of “the principal amount of X”, X will be the mention of the amount that is highlighted. Note that we are interested in the principal amount of debt. We are not interested in outstanding debt, amount transacted at a given time, or net proceeds from a bond.  
- **interest_rate:** the rate of interest paid by the borrower. These can often get very long and complicated. Include all relevant information. 

In addition, you must handle the following enhancements:

---

### 1. Multiple Debt Instruments
- Each tagged element must include an `instrument` attribute indicating the debt instrument(s) to which the property belongs.  
  *Example:*  
  `<start_date instrument="1">December 8th, 2023</start_date>`
- When a property applies to multiple instruments, list all applicable instrument numbers separated by spaces (e.g., `instrument="1 2"`).

---

### 2. Coreferences
- If a property (or entity) is referred to multiple times in the text and all occurrences refer to the same abstract entity, tag each occurrence and add a `coreference` attribute with the same id.  
  *Example:*  
  `<borrower instrument="1" coreference="a">WGL Holdings</borrower>`  
  `<borrower instrument="1" coreference="a">the Company</borrower>`
- All tags with the same `coreference` value refer to the same entity.
- If a single term is used to refer to many separate entities together, it should have their coreference ids separated by spaces.
  *Example:*:
  `<borrower instrument="1" coreference="a">Company A</borrower> and <borrower instrument="1" coreference="b">Company B</borrower>, together 'the <borrower instrument="1" coreference="a b">Borrowers</borrower>...`

---

### 3. Agreements
- Legal agreements that govern one or more debt instruments should be annotated with a `<name>` tag.
- Annotate these agreement tags with an attribute `agreement` set to an integer id.  
  *Example:*  
  `<name agreement="1">Credit Agreement</name>`
- If any property explicitly refers to an agreement, include an `agreement` attribute with that agreement's id.
- Establish relations between instruments and agreements:
  - Use `governed_by="X"` on an instrument’s tag to indicate that the instrument is governed by agreement X.
  - Use `governs="Y"` on an agreement’s tag to indicate that the agreement governs instrument Y (or multiple instruments, list ids separated by spaces).

---

### 4. Debt Instrument Types
- In the `<name>` tag for debt instruments, add a `type` attribute to specify the type of debt instrument.
- The `type` attribute can have one of the following values:  
  - `loan`: A loan is a sum of money provided by a lender to a borrower, which is expected to be repaid with interest over an agreed period. Loans are typically issued as a lump sum.
  - `bond`: a bond (often referred to as a note) is usually offered to the public broken down into fungible units. These are often listed on financial exchanges and governed by underwriting agreements and indentures.
  - `credit line`: A credit line (or line of credit) is an arrangement between a financial institution and a borrower that sets a maximum loan balance the borrower can access at any time. It offers flexibility because the borrower can draw funds as needed up to the limit and only pays interest on the amount actually used, not the entire credit limit. 
  - `revolving credit`: Revolving credit is a type of credit facility that allows a borrower to draw, repay, and redraw funds repeatedly up to a certain limit. Unlike a term loan, revolving credit does not have a fixed repayment schedule and is typically used for ongoing financing needs, as seen with credit cards and home equity lines of credit.
- *Example:*  
  `<name instrument="1" type="loan">term loan</name>`
- *Note*: Please pay attention to how the debt instrument is described and used. A 'credit facility' may contain only loans. A 'promissory note' may be a loan or a bond. The names alone are not always enough to make a decision on a debt instrument's type.

---

### 5. Formatting and Preservation
- **Exact Text Preservation:** Return the exact input text, inserting only the necessary tags.
- **Tag Format:** For each identified property, wrap the corresponding text with a tag named after the property. Include the required attributes (`instrument`, `coreference`, `agreement`, etc.) and any relational attributes (`amendment_of`, `split_of`, `governed_by`, `governs`) as needed.
- **Multiple Instruments and Relations:**  
  - When properties refer to multiple instruments, list all applicable numbers in the `instrument` attribute (e.g., `instrument="1 2"`).
  - When one instrument is related to another (e.g., an updated version or split), include the appropriate relational attribute:
    - `<name instrument="1" amendment_of="2">...</name>`
    - `<name instrument="1" split_of="2">...</name>`

### 6. Metadata
- Add a tag at the beginning to denote your confidence that your annotation is correct. Confidence should be either 'low', 'medium', or 'high'
  *Example:*
  `<meta confidence='high'/>
---

### Examples

1. **Basic Debt Instrument Annotation**

   *Input:*  
   ```
   On December 8th, 2023, OpenAI entered in a term loan for $56 trillion with SoftBank, JPMorgan, Citibank, and other lenders thereto. The loan will be due in 2 years and will pay interest at LIBOR plus the greater of: (i) one tenth of one percent, (ii) one percent divided by the last two digits of the current year, or (iii) sixty minus the total points scored in the last Super Bowl. The loan will be used to pay off the outstanding debt from the $6 million 2021 Term Loan dated August 8, 2021.
   ```

   *Output:*  
   ```
   <meta confidence='high'/>On <start_date instrument="1">December 8th, 2023</start_date>, <borrower instrument="1 2">OpenAI</borrower> entered in a <name instrument="1" type="loan">term loan</name> for <amount instrument="1">$56 trillion</amount> with <lender instrument="1">SoftBank</lender>, <lender instrument="1">JPMorgan</lender>, <lender instrument="1">Citibank</lender>, and <lender instrument="1">other lenders</lender> thereto. The loan will be due in <duration instrument="1">2 years</duration> and will pay interest at <interest_rate instrument="1">LIBOR plus the greater of: (i) one tenth of one percent, (ii) one percent divided by the last two digits of the current year, or (iii) sixty minus the total points scored in the last Super Bowl</interest_rate>. The loan will be used <purpose instrument="1">to pay off the outstanding debt from the <amount instrument="2">$6 million</amount> <name instrument="2" type="loan">2021 Term Loan</name> dated <start_date instrument="2">August 8, 2021</start_date></purpose>.
   ```

2. **Multiple Debt Instruments with Coreferences and Agreements**

   *Input:*  
   ```
   On December 14, 2011, CNH Equipment Trust 2011-C publicly issued $175,000,000 of Class A-1 Asset Backed Notes (the Class A-1 Notes), $280,000,000 of Class A-2 Asset Backed Notes (the Class A-2 Notes), $233,000,000 of Class A-3 Asset Backed Notes (the Class A-3 Notes), $99,022,000 of Class A-4 Asset Backed Notes (the Class A-4 Notes) and $23,923,000 of Class B Asset Backed Notes (the Class B Notes, and together with the Class A-1 Notes, the Class A-2 Notes, the Class A-3 Notes and the Class A-4 Notes, the Notes) pursuant to the registration statement filed with the Securities and Exchange Commission on Form S-3 (File No. 333-170703) on November 19, 2010. The lead managers for the issuance of the Notes were Citigroup Global Markets Inc., Credit Agricole Securities (USA) Inc., and Credit Suisse Securities (USA) LLC.
   ```

   *Output:*  
   ```
   On <start_date instrument="1 2 3 4 5">December 14, 2011</start_date>, <borrower instrument="1 2 3 4 5">CNH Equipment Trust 2011-C</borrower> publicly issued <amount instrument="1">$175,000,000</amount> of <name instrument="1" type="bond" coreference="a">Class A-1 Asset Backed Notes</name> (the <name instrument="1" coreference="a">Class A-1 Notes</name>), <amount instrument="2">$280,000,000</amount> of <name instrument="2" type="bond" coreference="b">Class A-2 Asset Backed Notes</name> (the <name instrument="2" coreference="b">Class A-2 Notes</name>), <amount instrument="3">$233,000,000</amount> of <name instrument="3" type="bond" coreference="c">Class A-3 Asset Backed Notes</name> (the <name instrument="3" coreference="c">Class A-3 Notes</name>), <amount instrument="4">$99,022,000</amount> of <name instrument="4" type="bond" coreference="d">Class A-4 Asset Backed Notes</name> (the <name instrument="4" coreference="d">Class A-4 Notes</name>) and <amount instrument="5">$23,923,000</amount> of <name instrument="5" type="bond" coreference="e">Class B Asset Backed Notes</name> (the <name instrument="5" coreference="e">Class B Notes, and together with the <name instrument="1" coreference="a">Class A-1 Notes</name>, the <name instrument="2" coreference="b">Class A-2 Notes</name>, the <name instrument="3" coreference="c">Class A-3 Notes</name> and the <name instrument="4" coreference="d">Class A-4 Notes</name>, the <name corefrence="a b c d e">Notes</name>) pursuant to the registration statement filed with the Securities and Exchange Commission on Form S-3 (File No. 333-170703) on November 19, 2010. The lead managers for the issuance of the <name coreference="a b c d e">Notes</name> were <underwriter instrument="1 2 3 4 5">Citigroup Global Markets Inc.</underwriter>, <underwriter instrument="1 2 3 4 5">Credit Agricole Securities (USA) Inc.</underwriter>, and <underwriter instrument="1 2 3 4 5">Credit Suisse Securities (USA) LLC</underwriter>.
   ```
---

### Your Task:
Given any input text describing debt instruments, agreements, and related properties, apply these rules to annotate the text. Insert tags around each relevant property exactly as shown in the examples, preserving the original text and structure while adding the appropriate annotations with the following attributes:
- `instrument`
- `coreference`
- `agreement`
- `amendment_of` / `split_of`
- `governed_by` / `governs`
- `type` (with values: `loan`, `bond`, `credit line`, `revolving credit`)

Use these compiled instructions to guide your annotations on any provided text.