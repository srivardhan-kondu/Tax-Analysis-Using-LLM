# Tax Evasion Detection System - Architecture

## System Overview

The Tax Evasion Detection System uses GPT-4.0 AI to analyze company financial data and identify potential tax evasion patterns. The system replaces traditional Machine Learning/Deep Learning models with intelligent prompt engineering and anti-hallucination measures.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Authentication│  │ Data Upload  │  │  Dashboard   │      │
│  │    Module     │  │   Module     │  │   Module     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Risk Prediction Engine                   │
│              (Orchestrates the analysis pipeline)            │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│Data Processor│    │ GPT-4.0 AI   │    │  Validator   │
│              │    │   Analyzer   │    │              │
│• Cleaning    │    │• Analysis    │    │• Input Check │
│• Validation  │    │• Prompts     │    │• Output Valid│
│• Features    │    │• API Calls   │    │• Rules Check │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Component Breakdown

### 1. User Interface Layer (`ui/`)

#### `authentication.py`
- User login/logout (FR1)
- Session management
- Access control (FR2)
- Default demo users

#### `data_upload.py`
- CSV file upload (FR3)
- Manual data entry (FR5)
- Data validation display
- Sample data loading

#### `dashboard.py`
- Results visualization (FR20, FR21)
- Risk distribution charts
- Company details view
- Export functionality (FR23)

### 2. Data Processing Layer (`data/`)

#### `data_processor.py`
- Data cleaning (FR6, FR7)
- Missing value handling
- Duplicate removal
- Normalization (FR8)
- Feature engineering (FR9)
- Ratio calculations

#### `data_validator.py`
- Schema validation
- Range checks
- Business logic validation
- Warning generation

### 3. AI Intelligence Layer (`ai/`)

#### `gpt4_analyzer.py`
- OpenAI GPT-4.0 integration
- API call management
- Retry logic with tenacity
- Batch processing support (FR24)
- Confidence scoring (FR19)

**Anti-Hallucination Measures:**
- Temperature = 0 for deterministic output
- JSON response format enforcement
- Response validation
- Confidence thresholds
- Rule-based cross-checking

#### `prompts.py`
- System prompt engineering
- Analysis prompt templates
- Structured output requirements
- Evidence-based reasoning prompts

#### `validation.py`
- Output consistency checks
- Logical validation
- Evidence quality assessment
- Multi-analysis comparison

### 4. Prediction Engine (`engine/`)

#### `risk_predictor.py`
- Main orchestrator
- Single/batch prediction (FR16, FR17, FR24)
- Risk classification (FR18)
- Summary generation (FR22)
- Result formatting

### 5. Configuration (`config.py`)

- Environment variables
- API key management
- Model parameters
- Risk thresholds
- Directory structure

## Data Flow

### Single Company Analysis

```
1. User Input (Manual Entry)
   │
   ▼
2. Input Validation
   │
   ▼
3. Feature Engineering
   │
   ▼
4. GPT-4.0 Analysis
   │  • Create analysis prompt
   │  • Call OpenAI API
   │  • Parse JSON response
   │
   ▼
5. Output Validation
   │  • Check consistency
   │  • Apply rule-based validation
   │  • Verify confidence score
   │
   ▼
6. Risk Classification
   │  • Assign risk level (0/1/2)
   │  • Generate findings
   │  • Create explanation
   │
   ▼
7. Display Results
```

### Batch Analysis

```
1. CSV Upload
   │
   ▼
2. Data Preprocessing
   │  • Clean data (FR6, FR7)
   │  • Validate schema
   │  • Engineer features (FR9)
   │
   ▼
3. Batch Processing
   │  • Split into manageable batches
   │  • Analyze each company
   │  • Aggregate results
   │
   ▼
4. Summary Generation
   │  • Risk distribution
   │  • Confidence statistics
   │  • High-risk identification
   │
   ▼
5. Visualization & Export
```

## Anti-Hallucination Strategy

### Layer 1: Prompt Engineering
- Clear, specific instructions
- Evidence requirement
- Numerical grounding
- Structured output format

### Layer 2: Temperature Control
- Temperature = 0.0
- Deterministic outputs
- Consistent results across runs

### Layer 3: Structured Outputs
- JSON format enforcement
- Schema validation
- Required fields check

### Layer 4: Rule-Based Validation
- Financial ratio checks
- Threshold validation
- Logical consistency

### Layer 5: Confidence Filtering
- Threshold: 0.7 (configurable)
- Manual review flagging
- Low-confidence handling

## Security Features

1. **Authentication**
   - Password hashing (SHA-256)
   - Session management
   - Role-based access

2. **Data Protection**
   - API key in environment variables
   - Git ignore for sensitive files
   - No data persistence (optional)

3. **Input Validation**
   - Range checks
   - Type validation
   - Business logic rules

## Scalability Considerations

1. **Batch Processing**: Handles multiple companies efficiently
2. **API Rate Limiting**: Implements retry logic
3. **Stateless Design**: Each analysis is independent
4. **Result Caching**: Can be implemented for repeated analyses
5. **Database Support**: SQLite included for future expansion

## Performance Metrics

- **Analysis Time**: ~2-5 seconds per company
- **Accuracy**: Depends on GPT-4.0 model quality
- **Confidence**: Average 0.85+ on well-structured data
- **API Costs**: ~$0.01-0.02 per company (GPT-4.0 pricing)

## Future Enhancements

1. Result caching to reduce API costs
2. Historical trend analysis
3. Industry benchmarking
4. Multi-language support
5. Advanced reporting (PDF generation)
6. API endpoint for integration
7. Real-time monitoring dashboard

## Technology Stack

- **Language**: Python 3.8+
- **AI**: OpenAI GPT-4.0
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Database**: SQLite (optional)
- **API Client**: OpenAI Python SDK

## Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Add secrets for API keys
4. Deploy

### Docker (Future)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

## Monitoring & Logging

- Python logging module
- Console output for debugging
- API call tracking
- Error handling with try-except
- Validation result logging

---

**Version**: 1.0.0  
**Last Updated**: February 2026
