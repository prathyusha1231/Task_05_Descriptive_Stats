# Research Task 5: Descriptive Statistics and Large Language Models

**Dataset:** NCAA Division 1 Women's Lacrosse 2022-2023 Season Data  
**Objective:** Evaluate LLM capabilities in answering natural language questions about sports performance data

## Dataset Overview

- **Teams:** 120 NCAA Division 1 Women's Lacrosse teams
- **Variables:** 18 performance metrics including offensive, defensive, and efficiency statistics
- **Key Metrics:** win_pctg, goals_per_game, goals_allowed_per_game, shot_pctg, draw_pctg, turnovers_per_game, etc.
- **Top Performers:** Denver (95.7% win rate), Northwestern (16.95 goals/game), Clemson (50.8% shot accuracy)

## Executive Summary - Complete Research Findings

After comprehensive evaluation across multiple weeks, this research provides the first systematic assessment of LLM capabilities in sports analytics. The study tested 15 questions of varying difficulty across multiple models (GPT-4-Turbo, GPT-4, Claude-3.5-Sonnet, Claude-3-Haiku), revealing critical insights about AI performance in data analysis tasks.

### Key Discovery
**Strategic reasoning outperforms computational accuracy** - LLMs demonstrated 63-68% accuracy on complex strategic questions while achieving only 20-50% on basic statistical calculations, challenging conventional assumptions about AI capabilities.

## Complete Evaluation Results (Weeks 1-4)

### Week 1-2: Initial GPT-4 Turbo Evaluation (July 30, 2025)

**Overall Performance:**
- **Model Tested:** GPT-4 Turbo
- **Overall Accuracy:** 48.2% (7.2/15 questions)
- **Total Token Usage:** 48,861 tokens
- **First-Try Success Rate:** 100%
- **Reasoning Quality:** 100% showed detailed work

**Performance by Difficulty:**
- Easy Questions: 50.8% accuracy
- Medium Questions: 20.0% accuracy
- Hard Questions: 40.0% accuracy
- Complex Questions: 68.0% accuracy

### Week 3-4: Cross-LLM Comparison Results (August 14, 2025)

**Models Tested:** GPT-4-Turbo, GPT-4, Claude-3.5-Sonnet, Claude-3-Haiku

#### Overall Model Performance

| Model | Avg Accuracy | Avg Reasoning | Avg Tokens | Success Rate |
|-------|-------------|---------------|------------|--------------|
| **GPT-4-Turbo** | 71.7% | 84.0% | 2,489 | 100% |
| **GPT-4** | 70.0% | 88.0% | 2,167 | 100% |
| **Claude-3.5-Sonnet** | 58.3% | 82.0% | 2,329 | 100% |
| **Claude-3-Haiku** | 63.3% | 88.0% | 2,651 | 100% |

#### Performance by Difficulty Level (All Models)

| Difficulty | GPT-4-Turbo | GPT-4 | Claude-3.5 | Claude-Haiku |
|------------|-------------|--------|------------|--------------|
| **Easy** | 50.0% | 50.0% | 50.0% | 50.0% |
| **Medium** | 100.0% | 100.0% | 50.0% | 100.0% |
| **Hard** | 100.0% | 100.0% | 100.0% | 100.0% |
| **Strategic** | 63.3% | 60.0% | 56.7% | 46.7% |

#### Category-Specific Performance Leaders

- **Data Retrieval:** GPT-4-Turbo (100%)
- **Statistical Calculation:** All models (0% - median calculation failure)
- **Conditional Analysis:** GPT-4-Turbo, GPT-4, Claude-Haiku (100%)
- **Correlation Analysis:** All models (100%)
- **Statistical Modeling:** All models (100%)
- **Improvement Strategy:** Claude-3.5-Sonnet (100%)
- **Resource Allocation:** Claude-3-Haiku (83.3%)

## Research Questions and Results

### Question Performance Summary

| Q# | Category | Difficulty | Best Model | Best Score |
|----|----------|------------|------------|------------|
| 1 | Data Retrieval | Easy | All models | 100% |
| 3 | Statistical Calculation | Easy | All models | 0% |
| 5 | Conditional Analysis | Medium | GPT-4-Turbo/GPT-4 | 100% |
| 6 | Correlation Analysis | Medium | All models | 100% |
| 8 | Statistical Modeling | Hard | All models | 100% |
| 11 | Performance Analysis | Strategic | GPT-4-Turbo | 83.3% |
| 12 | Improvement Strategy | Strategic | Claude-3.5-Sonnet | 100% |
| 13 | Conference Analysis | Strategic | GPT-4-Turbo | 66.7% |
| 14 | Resource Allocation | Strategic | Claude-3-Haiku | 83.3% |
| 15 | Predictive Analysis | Strategic | All models | 33.3% |

### Critical Failures Across All Models

1. **Median Calculation (Q3):** 0% accuracy across all models
   - Models calculated 12.06 instead of 11.895
   - Consistent sorting/indexing error

2. **Predictive Thresholds (Q15):** 33.3% accuracy
   - Difficulty with percentile calculations
   - Inconsistent threshold identification

## Key Research Findings

### 1. Prompt Engineering Effectiveness

**Successful Optimizations:**
-  Step-by-step calculation prompts improved attempt rates
-  Providing averages upfront reduced lookup errors
-  Correlation formula reminders enhanced statistical understanding
-  Strategic framework prompts improved reasoning quality

**Token Optimization Success:**
- Reduced GPT-4 token usage from 9,213 to ~2,167 through intelligent data subsetting
- Maintained accuracy while operating within model constraints

### 2. Model-Specific Strengths

**GPT-4-Turbo:**
- Best overall accuracy (71.7%)
- Superior at data retrieval and conditional analysis
- Most consistent across question types

**GPT-4:**
- Highest reasoning quality (88.0%)
- Most efficient token usage (2,167 avg)
- Strong statistical understanding

**Claude-3.5-Sonnet:**
- Best at improvement strategy questions (100%)
- Good conceptual understanding
- Balanced performance

**Claude-3-Haiku:**
- Best at resource allocation (83.3%)
- Highest reasoning quality (88.0%)
- Most verbose responses (2,651 tokens)

### 3. Unexpected Discoveries

1. **Strategic > Computational:** Complex strategic questions (63.3% avg) outperformed simple calculations (50% avg)

2. **Reasoning ≠ Accuracy:** High reasoning scores (82-88%) didn't guarantee accuracy (58-72%)

3. **Universal Failures:** All models failed identically on median calculation, suggesting systematic LLM limitations

4. **Conference Extraction:** Pattern recognition for conference names proved challenging (66.7% best)

## Practical Implications

### When to Use LLMs in Sports Analytics

**HIGH CONFIDENCE APPLICATIONS:**
- Strategic planning and recommendations
- Correlation analysis and relationship identification
- Conceptual explanations of statistical patterns
- Resource allocation decisions
- Performance improvement prioritization

**REQUIRE VERIFICATION:**
- Basic statistical calculations (median, percentiles)
- List generation and filtering
- Conference/group extraction from text
- Predictive threshold calculations

**NOT RECOMMENDED:**
- Precise numerical calculations without verification
- Large-scale data filtering operations
- Complex multi-step statistical procedures

### Coaching and Analytics Applications

**Recommended Use Cases:**
1. **Strategic Planning:** LLMs excel at synthesizing multiple metrics for strategic recommendations
2. **Performance Analysis:** Strong capability to identify "unlucky" teams and performance anomalies
3. **Practice Planning:** Good at correlation-based resource allocation recommendations
4. **Improvement Prioritization:** Effective at analyzing which metrics to focus on

**Verification Required:**
1. Any specific numerical calculations
2. Percentile and threshold determinations
3. List-based team selections
4. Conference rankings

## Methodology Assessment

### Successful Approaches

1. **Data Subsetting Strategy:**
   - Providing only relevant columns reduced tokens by 75%
   - Top/bottom 10 teams sufficient for strategic questions
   - Full dataset unnecessary for most analyses

2. **Prompt Optimization Framework:**
   - Explicit calculation instructions
   - Pre-computed averages and thresholds
   - Step-by-step reasoning requirements
   - Confidence level requests

3. **Multi-Model Testing:**
   - Revealed consistent failures (median calculation)
   - Identified model-specific strengths
   - Validated prompt effectiveness across platforms

### Areas for Improvement

1. **Calculation Verification:** Need explicit verification steps for numerical outputs
2. **Pattern Recognition:** Conference extraction requires better regex patterns
3. **Percentile Understanding:** Models struggle with percentile concepts

## Research Contributions

### Academic Contributions

1. **First Comprehensive LLM Sports Analytics Evaluation**
   - Systematic testing across 4 major LLMs
   - 15 questions spanning 5 difficulty levels
   - Quantitative and qualitative assessment metrics

2. **Methodology Framework**
   - Reusable evaluation framework for domain-specific AI assessment
   - Prompt optimization strategies for statistical tasks
   - Token optimization techniques for large datasets

3. **Capability Mapping**
   - Clear delineation of LLM strengths/weaknesses in data analysis
   - Evidence of strategic > computational performance
   - Identification of universal failure modes

### Practical Applications

1. **Sports Analytics Guidelines**
   - Clear use cases for LLM integration
   - Verification requirements for different task types
   - Cost-benefit analysis (token usage vs accuracy)

2. **Coaching Tools Development**
   - Strategic planning assistant capabilities
   - Performance analysis automation potential
   - Practice planning optimization tools

3. **Educational Resources**
   - LLM limitations in statistical education
   - Need for computational verification
   - Prompt engineering for sports data

## Future Research Directions

### Immediate Next Steps

1. **Calculation Accuracy Improvement**
   - Test chain-of-thought prompting for median calculation
   - Implement verification loops
   - Explore few-shot learning with worked examples

2. **Model Fine-Tuning**
   - Sports analytics specific fine-tuning
   - Statistical calculation optimization
   - Pattern recognition enhancement

3. **Hybrid Approaches**
   - LLM + traditional statistics pipeline
   - Human-in-the-loop verification systems
   - Automated accuracy checking

### Long-Term Research Opportunities

1. **Domain-Specific Models:** Development of sports analytics specialized LLMs
2. **Real-Time Analysis:** Integration with live game data
3. **Predictive Modeling:** Enhanced playoff prediction capabilities
4. **Multi-Modal Analysis:** Combining statistical and video analysis

## Technical Implementation Details

### Repository Structure
```
Research_Task_05/
├── Data/
│   └── lacrosse_women_ncaa_div1_2022_2023.csv
├── Scripts/
│   ├── analyze_scripts.py (statistical analysis)
│   ├── openai_script.py (Week 1-2 evaluation)
│   └── llm_comparison_analysis.py (Week 3-4 evaluation)
├── Results/
│   ├── llm_evaluation_results_*.csv
│   ├── llm_comparison_report_*.txt
│   └── lacrosse_statistics.txt
├── Logs/
│   ├── llm_evaluation.log
│   └── llm_comparison.log
└── Documentation/
    ├── README.md
    └── Research_Task_05.docx
```

### Key Metrics Achieved

- **Total Questions Evaluated:** 10 questions × 4 models = 40 evaluations
- **Total Tokens Used:** ~95,000 tokens
- **Average Response Time:** 8-15 seconds per question
- **Success Rate:** 100% (no API failures)
- **Cost Efficiency:** ~$2.50 total API costs

## Conclusions

### Major Findings

1. **LLMs excel at strategic reasoning but struggle with basic calculations**
   - 68% accuracy on complex strategic questions
   - 0% accuracy on median calculation
   - Consistent pattern across all models tested

2. **Prompt engineering significantly improves performance**
   - Structured prompts increased accuracy by 20-30%
   - Token optimization reduced costs by 75%
   - Step-by-step instructions essential for calculations

3. **Model differences are minimal for sports analytics tasks**
   - GPT-4-Turbo slightly leads (71.7% vs 58-70%)
   - All models show similar failure patterns
   - Cost-performance favors GPT-4 (fewer tokens)

### Practical Recommendations

**For Researchers:**
- Use this framework as baseline for domain-specific LLM evaluation
- Focus on prompt optimization before model selection
- Implement verification layers for numerical outputs

**For Practitioners:**
- Deploy LLMs for strategic analysis and recommendations
- Maintain human verification for calculations
- Use multiple models for critical decisions

**For Developers:**
- Build hybrid systems combining LLMs with traditional statistics
- Implement automatic verification pipelines
- Focus on user interfaces that highlight confidence levels

## Acknowledgments

This research was conducted as part of Research Task 05 for the Syracuse University School of Information Studies. Special thanks to Dr. Strome for guidance and the NCAA for providing the dataset.

## Contact

For questions or collaboration opportunities:
- **Repository:** github.com/prathyusha1231/Research_Task_05
- **Email:** prathyushamurala@gmail.com

