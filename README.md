# Document Verification System

A production-ready document verification pipeline that extracts and validates identity information from government IDs, bank statements, and employment letters using AI-powered OCR and structured LLM extraction.

## Features

- **OCR + LLM Architecture**: Uses vision models for text extraction and language models for structured data parsing.
- **Strict KYC Validation**: Enforces 7 deterministic rules for cross-document consistency:
  1. Full name matching (case-insensitive)
  2. Date of birth alignment
  3. Address similarity (≥70%)
  4. Phone number consistency
  5. Father’s name verification
  6. PAN format: `ABCDE1234F`
  7. Aadhaar format: exactly 12 digits
- **Robust Error Handling**: Automatic retries, rate-limit protection, and detailed logging.
- **Modular Design**: Clean separation between extraction, structuring, and verification logic.

## Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Create a `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Data Layout
Organize documents as:
```
data/
├── P001/
│   ├── doc1.jpg  # e.g., Aadhaar
│   ├── doc2.jpg  # e.g., Bank Statement
│   └── doc3.jpg  # e.g., Employment Letter
└── P002/
    └── ...
```

### Run
```bash
python main.py
```

Output is saved to `output/verification_results.json`.

## Output Format
Each result includes:
- `extracted_data`: Structured fields from each document
- `verification_results`: PASS/FAIL status for all 7 rules
- `overall_status`: `VERIFIED` or `FAILED`

## Dependencies
- `google-generativeai`
- `python-dotenv`
- `loguru`
- `Pillow`
- `python-dateutil`

Designed for reliability in real-world document processing workflows.
