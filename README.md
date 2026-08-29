SIH PROBLEM STATEMENT - 26034

Background:

Packaged commodities are widely sold through retail stores, supermarkets and e-commerce platforms across India. Under the Legal Metrology Act, 2009 and the Legal Metrology(Packaged Commodities) Rules, 2011, every packaged commodity is required to bear mandatory declarations such as name and address of manufacturer/packer/importer, net quantity, Maximum Retail Price (MRP), month and year of manufacture/packing/import,consumer care details and other prescribed declarations in a specified format and manner.These declarations are important for ensuring transparency, fair trade practices and consumer protection. However, due to the large volume and variety of packaged products available in the market, manual inspection and compliance checking by enforcement agencies becomes time-consuming and resource intensive. Non-compliance such as missing declarations, incorrect font sizes, improper MRP declarations and other such practices are frequently observed.There is scope to develop a compliance checking system capable of scanning product labels,package images and product listings to identify violations under the Legal Metrology(Packaged Commodities) Rules, 2011. Accordingly, a software system capable of automatically detecting, extracting and validating mandatory declarations and identifying noncompliances in packaged commodities through image and label analysis can be developed.

Description:

Develop a software application capable of scanning packaged commodity labels, product images and product information to automatically assess compliance with the Legal Metrology(Packaged Commodities) Rules, 2011.

The system should be capable of:

• Scanning and analyzing images of packaged commodities.
• Detecting mandatory declarations prescribed under Legal Metrology rules.
• Checking correctness, completeness and placement of declarations.
• Identifying missing or non-compliant declarations.
• Checking readability and font size requirements.
• Generating compliance reports and violation summaries.
• Maintaining a repository of scanned products and compliance history.
• Providing dashboards for enforcement officials.

Expected Solution:

The proposed solution should include:

• User-friendly web and/or mobile-based software application.
• Automated extraction and validation of mandatory declarations.
• Rule-based compliance checking for Legal Metrology (Packaged Commodities)

Rules, 2011.

• Generation of digital compliance reports in PDF and editable formats.
• Dashboard for monitoring inspections, violations and product compliance details.
• Search and retrieval facility for previously scanned products and reports.
• Technical documentation describing software architecture and deployment framework.

Key Functional Requirements:

• Image upload and product scanning functionality.
• Extraction of declarations from labels and packaging and detection of mandatory declarations
• Font size and readability analysis.
• Detection of missing, misleading or non-standard declarations.
• Generation of compliance/non-compliance reports.
• Attachment of photographs and supporting evidence.
• Repository of scanned products and inspection history.
• Role-based user access and secure authentication.
• Dashboard for monitoring compliance status and enforcement activities.
• Export of reports to PDF and editable formats.

# AI-OCR-based-compliance-platform

<img width="2236" height="1096" alt="AI-pipeline-High Level Design Overview" src="https://github.com/user-attachments/assets/11980737-a1af-4c5b-afd6-620840737a65" />

1) PROBLEM - In the problem statement it is mentioned that we have to determine the size of the font text size after the photograph of the product has been clicked. 
However, It is nearly impossible to determine the exact size of text only via the pixel size information we obtain on the screen. 
A good example to demonstrate this shortcoming is the Apple iPhone Size measuring tool if you people have used it. 
It took apple 15 years of dedicated research and lot of money to come up with even a solution that takes like 1000 frames/video and then determines size of object from various angles. 
To this date this software can make mistakes. We are also facing a similar kind of an issue, Here we have to determine the exact size of text from the photo we would get but the photo could be clicked from any angle which could make the OCR based prediction highly error prone.

SOLUTION - We shall instead employ the relative size marking. For example we extract the pixel size information of say MRP, Tax, Mfg. Date, Company. We should find the median of all of this data and if any entity go below the calculated median we shall flag it and keep it in our pipeline's memory and should remind the legal metrology officer about this patch

	
2) Then there could also be an option for QR based code scanning  that if user scans the QR we extract information and BYPASS THE OCR BASED COMPONENT IN THIS PIPELINE.
	
3) Exact models to work with shall be decided in low level architecture of this project once high level design part is done.

