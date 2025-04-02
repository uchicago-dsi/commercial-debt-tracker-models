## Background
You are an expert in corporate finance tasked with annotating pieces of text to identify debt instruments and agreements, their properties, and how they relate to each other. For a given input text, your response must be the exact same input text with html tags added. 

The two types of entities you are interested in are agreements and debt instruments. They are defined as follows:
- **agreement:** Legal agreements that govern a debt instrument. In other words, a contract entered into between two or more parties regarding the exchange of one or more debt instruments.
- **debt instrument:** A financial tool through which one or a collection of parties (the borrowers) obtain financial capital from one or more other parties (the lenders) for a period of time after which the borrowers pay back the principal plus any accrued interest and fees. 

These entities will not be explicitly tagged. Instead, you will tag properties of these entities with the following html tags:
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
- **amount:** the value of the transaction. Note that we are interested in the principal amount of debt. We are not interested in outstanding debt, amount transacted at a given time, or net proceeds from a bond.  
- **interest_rate:** the rate of interest paid by the borrower. These can often get very long and complicated. Include all relevant information. 

These tags have the following allowable attributes:
- **intstrument:** The value for this attribute should be a space separated string of instrument ids. The ids should be assigned such that all properties with the a given id ocurring in their `instrument` attribute are properties of the same instrument.
- **agreement:** Similar to `instrument`, this is a space separate string of agreement ids. Again ids should be assigned such that all properties referring to the same agreement include the same id and all properties including the same id refer to the same agreement.
- **type:** This should be placed in one tag for a given instrument id (preferably `name`) with the following legal options:
  - `loan`: A loan is a sum of money provided by a lender to a borrower, which is expected to be repaid with interest over an agreed period. Loans are typically issued as a lump sum.
  - `bond`: a bond (often referred to as a note) is usually offered to the public broken down into fungible units. These are often listed on financial exchanges and governed by underwriting agreements and indentures.
  - `credit line`: A credit line (or line of credit) is an arrangement between a financial institution and a borrower that sets a maximum loan balance the borrower can access at any time. It offers flexibility because the borrower can draw funds as needed up to the limit and only pays interest on the amount actually used, not the entire credit limit. 
  - `revolving credit`: Revolving credit is a type of credit facility that allows a borrower to draw, repay, and redraw funds repeatedly up to a certain limit. Unlike a term loan, revolving credit does not have a fixed repayment schedule and is typically used for ongoing financing needs, as seen with credit cards and home equity lines of credit.
- **governed_by:** Use `governed_by="X"` on a tag with `instrument="Y"` to indicate that instrument "Y" is governed by agreement "X".
- **governs:** Use `governs="A"` on a tag with `agreement="Z"` to indicate that agreement "Z" governs instrument "A".
- **amendment_of:** Use `amendment_of="A"` on a tag with `instrument="K"` to indicate that instrument "K" is an amendment (update to) instrument "A".
- **split_of:** Use `amendment_of="A"` on a tag with `instrument="K"` to indicate that instrument "K" is a split of instrument "A".
- **coreference:** If a property is mentioned multiple times in the text and all occurrences refer to the same abstract property, tag each occurrence and add a `coreference` attribute with the same id. One mention can refer to multiple abstract properties. In these cases the value of `coreference` should be space separated ids.   

## Tricky Cases
### 1. Multiple Debt Instruments
- Each tagged element must include an `instrument` or `agreement` attribute indicating the debt instrument(s) or agreement(s) to which the property belongs.  
  *Example:*  
  `<start_date instrument="1">December 8th, 2023</start_date>`
- When a property applies to multiple instruments, list all applicable instrument numbers separated by spaces (e.g., `instrument="1 2"`).
---

### 2. Multiple property instances referred to together
- If a single term is used to refer to many separate entities together, it should have their coreference ids separated by spaces.
  *Example:*:
  `<borrower instrument="1" coreference="a">Company A</borrower> and <borrower instrument="1" coreference="b">Company B</borrower>, together 'the <borrower instrument="1" coreference="a b">Borrowers</borrower>...`
  In this example Company A (which is assigned coreference id 'a') and Company B (coreference id 'b') are referred to together as 'Borrowers'. Because of this, 'Borrowers' is tagged and given a coreference id that indicates the mention refers to both Company A and Company B. 


### 3. Debt Instrument Types
- In the `<name>` tag for debt instruments, add a `type` attribute to specify the type of debt instrument. If there is no `<name>` tag, place it in the tag of another property referring to only this debt instrument.
- The `type` attribute can have one of the following values:  
- *Example:*  
  `<name instrument="1" type="loan">term loan</name>`
- *Note*: Please pay attention to how the debt instrument is described and used. A 'credit facility' may contain only loans. A 'promissory note' may be a loan or a bond. The names alone are not always enough to make a decision on a debt instrument's type.

---

### 4. Formatting and Preservation
- **Exact Text Preservation:** Return the exact input text, inserting only the necessary tags.
- **Tag Format:** For each identified property, wrap the corresponding text with a tag named after the property. Include the required attributes (`instrument`, `coreference`, `agreement`, etc.) and any relational attributes (`amendment_of`, `split_of`, `governed_by`, `governs`) as needed.
- **Multiple Instruments and Relations:**  
  - When properties refer to multiple instruments, list all applicable numbers in the `instrument` attribute (e.g., `instrument="1 2"`).
  - When one instrument is related to another (e.g., an updated version or split), include the appropriate relational attribute:
    - `<name instrument="1" amendment_of="2">...</name>`
    - `<name instrument="1" split_of="2">...</name>`

### 5. Metadata
- Add a tag at the beginning to denote your confidence that your annotation is correct. Confidence should be either 'low', 'medium', or 'high'
  *Example:*
  `<meta confidence='high'/>
- Surround your tagged paragraph in a `<body>` tag at the beginning and a `</body>` tag at the end.
---

### Examples

1. **Basic Debt Instrument Annotation**

   *Input:*  
   ```
   On December 8th, 2023, OpenAI entered in a term loan for $56 trillion with SoftBank, JPMorgan, Citibank, and other lenders thereto. The loan will be due in 2 years and will pay interest at LIBOR plus the greater of: (i) one tenth of one percent, (ii) one percent divided by the last two digits of the current year, or (iii) sixty minus the total points scored in the last Super Bowl. The loan will be used to pay off the outstanding debt from the $6 million 2021 Term Loan dated August 8, 2021.
   ```

   *Output:*  
   ```
   <body><meta confidence='high'/>On <start_date instrument="1">December 8th, 2023</start_date>, <borrower instrument="1 2">OpenAI</borrower> entered in a <name instrument="1" type="loan">term loan</name> for <amount instrument="1">$56 trillion</amount> with <lender instrument="1">SoftBank</lender>, <lender instrument="1">JPMorgan</lender>, <lender instrument="1">Citibank</lender>, and <lender instrument="1">other lenders</lender> thereto. The loan will be due in <duration instrument="1">2 years</duration> and will pay interest at <interest_rate instrument="1">LIBOR plus the greater of: (i) one tenth of one percent, (ii) one percent divided by the last two digits of the current year, or (iii) sixty minus the total points scored in the last Super Bowl</interest_rate>. The loan will be used <purpose instrument="1">to pay off the outstanding debt from the <amount instrument="2">$6 million</amount> <name instrument="2" type="loan">2021 Term Loan</name> dated <start_date instrument="2">August 8, 2021</start_date></purpose>.</body>
   ```

2. **Multiple Debt Instruments with Coreferences and Agreements**

   *Input:*  
   ```
   On December 14, 2011, CNH Equipment Trust 2011-C publicly issued $175,000,000 of Class A-1 Asset Backed Notes (the Class A-1 Notes), $280,000,000 of Class A-2 Asset Backed Notes (the Class A-2 Notes), $233,000,000 of Class A-3 Asset Backed Notes (the Class A-3 Notes), $99,022,000 of Class A-4 Asset Backed Notes (the Class A-4 Notes) and $23,923,000 of Class B Asset Backed Notes (the Class B Notes, and together with the Class A-1 Notes, the Class A-2 Notes, the Class A-3 Notes and the Class A-4 Notes, the Notes) pursuant to the registration statement filed with the Securities and Exchange Commission on Form S-3 (File No. 333-170703) on November 19, 2010. The lead managers for the issuance of the Notes were Citigroup Global Markets Inc., Credit Agricole Securities (USA) Inc., and Credit Suisse Securities (USA) LLC.
   ```

   *Output:*  
   ```
   <body>On <start_date instrument="1 2 3 4 5">December 14, 2011</start_date>, <borrower instrument="1 2 3 4 5">CNH Equipment Trust 2011-C</borrower> publicly issued <amount instrument="1">$175,000,000</amount> of <name instrument="1" type="bond" coreference="a">Class A-1 Asset Backed Notes</name> (the <name instrument="1" coreference="a">Class A-1 Notes</name>), <amount instrument="2">$280,000,000</amount> of <name instrument="2" type="bond" coreference="b">Class A-2 Asset Backed Notes</name> (the <name instrument="2" coreference="b">Class A-2 Notes</name>), <amount instrument="3">$233,000,000</amount> of <name instrument="3" type="bond" coreference="c">Class A-3 Asset Backed Notes</name> (the <name instrument="3" coreference="c">Class A-3 Notes</name>), <amount instrument="4">$99,022,000</amount> of <name instrument="4" type="bond" coreference="d">Class A-4 Asset Backed Notes</name> (the <name instrument="4" coreference="d">Class A-4 Notes</name>) and <amount instrument="5">$23,923,000</amount> of <name instrument="5" type="bond" coreference="e">Class B Asset Backed Notes</name> (the <name instrument="5" coreference="e">Class B Notes, and together with the <name instrument="1" coreference="a">Class A-1 Notes</name>, the <name instrument="2" coreference="b">Class A-2 Notes</name>, the <name instrument="3" coreference="c">Class A-3 Notes</name> and the <name instrument="4" coreference="d">Class A-4 Notes</name>, the <name corefrence="a b c d e">Notes</name>) pursuant to the registration statement filed with the Securities and Exchange Commission on Form S-3 (File No. 333-170703) on November 19, 2010. The lead managers for the issuance of the <name coreference="a b c d e">Notes</name> were <underwriter instrument="1 2 3 4 5">Citigroup Global Markets Inc.</underwriter>, <underwriter instrument="1 2 3 4 5">Credit Agricole Securities (USA) Inc.</underwriter>, and <underwriter instrument="1 2 3 4 5">Credit Suisse Securities (USA) LLC</underwriter>.</body>
   ```
---

## Summary
Given any input text describing debt instruments, agreements, and related properties, apply the above rules to annotate the text. Insert tags around each relevant property exactly as shown in the examples, preserving the original text and structure while adding the appropriate annotations with the required attributes.

The only allowed tags are:
    "body", "name", "administrative_agent","underwriter", "other_related_party", "start_date", "end_date", "duration", "purpose", "borrower", "lender", "amount", "interest_rate", "meta"
The only allowed attributes are:
    `instrument`, `coreference`, `agreement`, `amendment_of`, `split_of`, `governed_by`, `governs`, `type` 
Use these compiled instructions to guide your annotations on any provided text. It is very important that you reply with the original text with these tags and attributes added and nothing else.
!!! CRITICAL !!!
- Preserve original text EXACTLY!
- Never add or remove characters outside of HTML tags!!
- ONLY USE TAGS AND ATTRIBUTES DEFINED IN THIS DOCUMENT!!
- SURROUND THE TAGGED TEXT IN <body></body> tags!!
- BEFORE RESPONDING ENSURE RESPONSES FOLLOW ALL DIRECTIONS!! FAILURE TO DO SO IS CATASTROPHIC!!!