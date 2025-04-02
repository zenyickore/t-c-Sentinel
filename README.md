# T&C Sentinel

## Overview

T&C Sentinel is a web application that analyzes and compares Terms & Conditions documents to identify discrepancies, contradictions, and missing clauses. It uses advanced natural language processing techniques to help legal teams efficiently review contract differences.

## Features

- **Document Processing**: Extract and preprocess text from PDF documents
- **Intelligent Comparison**: Compare documents across 10 key legal categories
- **Discrepancy Detection**: Identify differences, contradictions, and missing clauses
- **Interactive UI**: User-friendly Streamlit interface with configurable settings
- **Exportable Reports**: Generate and download comparison reports

## Technical Architecture

T&C Sentinel uses a Retrieval Augmented Generation (RAG) approach:

1. **Document Processing**: Extracts text from PDFs and splits into manageable chunks
2. **Vector Embeddings**: Creates embeddings for document chunks using OpenAI's embedding model
3. **Similarity Search**: Retrieves relevant sections across documents for comparison
4. **LLM Analysis**: Uses GPT-4 to analyze differences and generate insights
5. **Result Presentation**: Displays findings in an organized, user-friendly interface

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone the repository or download the source code

2. Create a virtual environment:
   ```bash
   python -m venv contractcompare_env
   contractcompare_env\Scripts\activate  # On Windows
   ```

3. Install required packages:
   ```bash
   pip install streamlit langchain langchain_openai pypdf python-dotenv chromadb
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the Application

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Usage Guide

1. **Upload Documents**:
   - Upload your master T&C document on the left
   - Upload the client T&C document on the right

2. **Configure Settings** (optional):
   - Adjust chunk size and overlap in the sidebar
   - Modify retrieval count and temperature settings

3. **Run Comparison**:
   - Click the "Compare Documents" button
   - Wait for the processing to complete

4. **Review Results**:
   - Examine the executive summary
   - Navigate through category tabs to see detailed findings
   - Expand source sections to view the original text

5. **Export Report**:
   - Click "Download Report" to save findings as a markdown file

## Comparison Categories

The system analyzes documents across these key legal categories:

1. Liability provisions
2. Payment terms
3. Termination conditions
4. Warranty information
5. Intellectual property rights
6. Confidentiality requirements
7. Dispute resolution mechanisms
8. Force majeure clauses
9. Notice requirements
10. Amendment procedures

## Project Structure

```
contractcompare/
├── app.py                 # Main Streamlit application
├── document_processor.py  # Document processing functions
├── comparison_engine.py   # Document comparison logic
├── .env                   # Environment variables
└── README.md              # Project documentation
```

## Future Enhancements

- Support for additional document formats (DOCX, TXT)
- Custom category definition
- Confidence scoring for identified discrepancies
- Integration with document management systems
- Batch processing for multiple document pairs

## License

This project is licensed under the MIT License - see the LICENSE file for details.
