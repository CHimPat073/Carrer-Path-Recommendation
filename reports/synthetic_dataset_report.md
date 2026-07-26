# Synthetic Dataset Generation Report

## Metadata
- **Generation Timestamp**: 2026-07-26T16:07:08.807666
- **Dataset Version**: v2.0.0
- **Random Seed**: 42
- **Total Samples**: 20000
- **Number of Features**: 38

## Career Distribution
| Career | Count |
|--------|-------|
| AI Research Engineer | 1000 |
| Backend Developer | 1000 |
| Business Analyst | 1000 |
| Cloud Engineer | 1000 |
| Cyber Security Analyst | 1000 |
| Data Engineer | 1000 |
| Data Scientist | 1000 |
| Database Administrator | 1000 |
| DevOps Engineer | 1000 |
| Ethical Hacker | 1000 |
| Frontend Developer | 1000 |
| Full Stack Developer | 1000 |
| Game Developer | 1000 |
| ML Engineer | 1000 |
| Mobile App Developer | 1000 |
| Network Engineer | 1000 |
| Product Manager | 1000 |
| QA Engineer | 1000 |
| Software Engineer | 1000 |
| UI/UX Designer | 1000 |

## Data Quality Analysis

### Missing Value Analysis
- **Total Missing Values**: 0
- **Missing by Column**: {
  "career": 0,
  "years_experience": 0,
  "education_level": 0,
  "projects_completed": 0,
  "certifications": 0,
  "python_score": 0,
  "java_score": 0,
  "javascript_score": 0,
  "sql_score": 0,
  "machine_learning_score": 0,
  "deep_learning_score": 0,
  "cloud_score": 0,
  "devops_score": 0,
  "cybersecurity_score": 0,
  "data_analysis_score": 0,
  "database_score": 0,
  "networking_score": 0,
  "mobile_score": 0,
  "game_dev_score": 0,
  "testing_score": 0,
  "business_analysis_score": 0,
  "product_management_score": 0,
  "ui_design_score": 0,
  "ux_research_score": 0,
  "communication_score": 0,
  "leadership_score": 0,
  "problem_solving_score": 0,
  "teamwork_score": 0,
  "agile_score": 0,
  "research_score": 0,
  "salary_band": 0,
  "remote_preference": 0,
  "career_growth_score": 0,
  "job_satisfaction": 0,
  "work_hours_per_week": 0,
  "country": 0,
  "industry": 0,
  "employment_type": 0
}

### Duplicate Analysis
- **Duplicate Rows**: 0

### Feature Range Verification
All features were verified to be within valid career-specific ranges during generation.

## Statistical Analysis

### Numeric Feature Summary
- **years_experience**: min=0.00, max=20.00, mean=6.45, std=5.84
- **projects_completed**: min=0.00, max=30.00, mean=9.34, std=7.52
- **certifications**: min=0.00, max=6.00, mean=2.05, std=1.67
- **python_score**: min=1.00, max=10.00, mean=5.18, std=1.22
- **java_score**: min=1.00, max=10.00, mean=5.01, std=1.17
- **javascript_score**: min=1.00, max=10.00, mean=5.04, std=1.15
- **sql_score**: min=1.00, max=9.00, mean=5.09, std=1.18
- **machine_learning_score**: min=1.00, max=10.00, mean=5.12, std=1.19
- **deep_learning_score**: min=1.00, max=10.00, mean=5.00, std=1.16
- **cloud_score**: min=1.00, max=9.00, mean=5.02, std=1.15...

### Outlier Summary
- **Total Outliers Detected**: 230
- **Outlier Percentage**: 0.04%

### Correlation Summary
- **High Correlation Pairs (|r| > 0.7)**: 1

### Top 10 Correlated Feature Pairs
| Feature 1 | Feature 2 | Correlation |
|-----------|----------|-------------|
| years_experience | projects_completed | 0.9178 |

## Generation Parameters
- **Samples per Career**: 1000
- **Total Careers**: 20
- **Shuffle**: True
- **Gaussian Sampling**: Enabled
- **Career Overlap**: Enabled (25% probability)
- **Noise Injection**: Enabled (8% probability)
- **Correlation Rules**: Applied

## Validation Checklist
- [x] No missing values
- [x] No duplicate rows
- [x] Balanced career distribution (1000 per career)
- [x] Feature values within valid ranges
- [x] Gaussian sampling applied
- [x] Career overlap applied
- [x] Noise injection applied
- [x] Correlation rules applied
- [x] Dataset shuffled before saving

## Overall Status
**PASS**

## Recommendations
- Dataset is ready for ML model training
- Consider stratified splits for train/test splitting
- High correlation between some features is expected (e.g., ML-related skills)
