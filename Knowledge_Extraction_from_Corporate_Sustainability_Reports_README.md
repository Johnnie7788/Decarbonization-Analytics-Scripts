# Knowledge Extraction from Corporate Sustainability Reports

## Introduction
This project provides a Python script for **Knowledge Extraction from Corporate Sustainability Reports** using **Named Entity Recognition (NER)**. The script is designed to extract actionable insights from multilingual sustainability documents, enabling organizations, policymakers, and researchers to streamline compliance and make data-driven decisions.

The system leverages the state-of-the-art **BERT-based multilingual NER model** to identify key entities such as organizations, locations, and dates. The extracted insights are saved in a structured format for further analysis and reporting.

---

## Key Features

- **Multilingual Support**:
  - Uses a pre-trained multilingual BERT model for processing texts in multiple languages.

- **Entity Recognition**:
  - Identifies and extracts entities such as organizations, locations, and dates from sustainability reports.

- **Actionable Insights**:
  - Structures extracted entities into meaningful categories to support compliance and decision-making.

- **Scalable Design**:
  - Processes multiple reports efficiently with robust logging and modular functions.

- **Explainability and Transparency**:
  - Ensures entity extraction is clear and interpretable.

---

## Repository Structure

```
Knowledge-Extraction/
├── README.md                 # Project documentation (this file)
├── main_script.py            # Python script for knowledge extraction
├── requirements.txt          # List of required dependencies
└── example_reports.csv       # Example dataset (optional)
```

---

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- PyTorch
- Transformers
- Pandas, NumPy

### Installation

 Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

## How to Use

1. **Prepare Reports**:
   - Replace the example data in the `sustainability_reports` variable with your own sustainability reports.

2. **Run the Script**:
   ```bash
   python main_script.py
   ```

3. **Outputs**:
   - The extracted insights will be saved as a CSV file (`sustainability_insights.csv`) in the root directory.

---

## Example Insights

| Organizations     | Locations     | Dates |
|-------------------|---------------|-------|
| Company ABC       | Berlin        | 2023  |
| Organization XYZ  | Paris         | 2030  |

---

## Applications

- **Corporate Compliance**:
  - Automate the analysis of sustainability reports to ensure alignment with regulatory requirements.

- **Policy Development**:
  - Provide actionable insights for crafting impactful environmental policies.

- **Academic Research**:
  - Support data-driven research in sustainability and environmental studies.

---

## Contribution

Contributions are welcome! 
---

## License

This project is licensed under the MIT License. 
---

## Contact

For questions or suggestions, please reach out:
- **Email**: johnjohnsonogbidi@gmail.com
