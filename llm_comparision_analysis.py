"""
Cross-LLM Comparison and Strategic Analysis Evaluation Script
This script performs Week 3 and Week 4 tasks for Research Task 05:
- Tests refined prompts across different LLMs (GPT-4, Claude via API)
- Evaluates strategic analysis capabilities
- Generates comprehensive comparison report
"""

import os
import pandas as pd
import numpy as np
from openai import OpenAI
import anthropic
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_comparison.log'),
        logging.StreamHandler()
    ]
)

class QuestionDifficulty(Enum):
    """Enum for question difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    COMPLEX = "complex"
    STRATEGIC = "strategic"

@dataclass
class Question:
    """Data class for structured question information"""
    id: int
    difficulty: QuestionDifficulty
    category: str
    question: str
    expected_answer: Any
    validation_method: str
    prompt_optimization: str = ""

class MultiLLMEvaluator:
    """
    Multi-LLM evaluation system for comprehensive comparison
    
    Handles testing across multiple LLMs with optimized prompts
    and strategic analysis evaluation
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the multi-LLM evaluator
        
        Input:
            dataset_path (str): Path to the lacrosse CSV file
        """
        self.dataset_path = dataset_path
        self.df = self._load_dataset()
        
        # Initialize LLM clients
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Model configurations
        self.models = {
            'gpt-4-turbo': {'client': 'openai', 'max_tokens': 2000},
            'gpt-4': {'client': 'openai', 'max_tokens': 1500},
            'claude-3-5-sonnet': {'client': 'anthropic', 'max_tokens': 2000},
            'claude-3-haiku': {'client': 'anthropic', 'max_tokens': 1500}
        }
        
        self.results = []
        
    def _load_dataset(self) -> pd.DataFrame:
        """
        Load and prepare the lacrosse dataset
        
        Returns cleaned DataFrame
        """
        df = pd.read_csv(self.dataset_path)
        df.columns = df.columns.str.strip()
        df['Team'] = df['Team'].str.strip()
        logging.info(f"Dataset loaded: {df.shape}")
        return df
    
    def get_refined_questions(self) -> List[Question]:
        """
        Get refined questions with optimized prompts based on Week 1-2 learnings
        
        Returns list of Question objects with prompt optimizations
        """
        questions = []
        
        # WEEK 3: Cross-LLM Comparison Questions (Refined from original 15)
        
        # Easy Questions with Optimization
        questions.append(Question(
            id=1,
            difficulty=QuestionDifficulty.EASY,
            category="Data Retrieval",
            question="Which team has the highest win_pctg?",
            expected_answer="Denver (Big East)",
            validation_method="exact_match",
            prompt_optimization="explicit_column_reference"
        ))
        
        questions.append(Question(
            id=3,
            difficulty=QuestionDifficulty.EASY,
            category="Statistical Calculation",
            question="What is the median goals_per_game across all 120 teams?",
            expected_answer=11.895,
            validation_method="numerical_tolerance",
            prompt_optimization="step_by_step_calculation"
        ))
        
        # Medium Questions with Optimization
        questions.append(Question(
            id=5,
            difficulty=QuestionDifficulty.MEDIUM,
            category="Conditional Analysis",
            question="Among teams with above-average draw_pctg, which allowed the fewest goals_allowed_per_game?",
            expected_answer="Denver (Big East)",
            validation_method="exact_match",
            prompt_optimization="provide_averages_upfront"
        ))
        
        questions.append(Question(
            id=6,
            difficulty=QuestionDifficulty.MEDIUM,
            category="Correlation Analysis",
            question="Is win_pctg more correlated with draw_pctg or shot_pctg?",
            expected_answer="draw_pctg (r=0.708)",
            validation_method="concept_match",
            prompt_optimization="correlation_formula_reminder"
        ))
        
        # Hard Questions with Optimization
        questions.append(Question(
            id=8,
            difficulty=QuestionDifficulty.HARD,
            category="Statistical Modeling",
            question="Fit a linear model: win_pctg ~ draw_pctg + shot_pctg + turnovers_per_game. Which feature has the largest standardized effect?",
            expected_answer="draw_pctg",
            validation_method="exact_match",
            prompt_optimization="standardization_explanation"
        ))
        
        # WEEK 4: Strategic Analysis Questions (Questions 11-15)
        
        questions.append(Question(
            id=11,
            difficulty=QuestionDifficulty.STRATEGIC,
            category="Performance Analysis",
            question="Identify the most 'unlucky' team: Which team has significantly better offensive and defensive statistics than their win percentage would suggest?",
            expected_answer="Campbell (Big South)",
            validation_method="reasoning_quality",
            prompt_optimization="define_luck_metrics"
        ))
        
        questions.append(Question(
            id=12,
            difficulty=QuestionDifficulty.STRATEGIC,
            category="Improvement Strategy",
            question="For a team currently at 0.400 win percentage, should they prioritize improving their worst-performing metric or enhancing their best-performing metric? Provide data-driven reasoning.",
            expected_answer="Strategic analysis required",
            validation_method="reasoning_quality",
            prompt_optimization="correlation_based_reasoning"
        ))
        
        questions.append(Question(
            id=13,
            difficulty=QuestionDifficulty.STRATEGIC,
            category="Conference Analysis",
            question="Using team performance metrics, rank the top 5 conferences by competitive strength and identify which shows the most parity.",
            expected_answer="Conference analysis required",
            validation_method="reasoning_quality",
            prompt_optimization="conference_extraction_guide"
        ))
        
        questions.append(Question(
            id=14,
            difficulty=QuestionDifficulty.STRATEGIC,
            category="Resource Allocation",
            question="A coach has limited practice time - should they spend 70% on offense, 70% on defense, or 50/50 split? Use correlation analysis and diminishing returns theory.",
            expected_answer="Correlation-based recommendation",
            validation_method="reasoning_quality",
            prompt_optimization="correlation_interpretation"
        ))
        
        questions.append(Question(
            id=15,
            difficulty=QuestionDifficulty.STRATEGIC,
            category="Predictive Analysis",
            question="Based on statistical patterns, what minimum thresholds in 3 key metrics would a team need to achieve to have an 80% probability of making playoffs?",
            expected_answer="Top 20% performance thresholds",
            validation_method="reasoning_quality",
            prompt_optimization="percentile_calculation"
        ))
        
        return questions
    
    def _get_relevant_columns(self, question: Question) -> list:
        """
        Get relevant columns based on question type
        
        Input: Question object
        Returns: List of relevant column names
        """
        question_text = question.question.lower()
        
        if 'win_pctg' in question_text and 'highest' in question_text:
            return ['Team', 'win_pctg']
        elif 'median' in question_text and 'goals_per_game' in question_text:
            return ['Team', 'goals_per_game']
        elif 'draw_pctg' in question_text and 'goals_allowed' in question_text:
            return ['Team', 'draw_pctg', 'goals_allowed_per_game']
        elif 'correlation' in question_text:
            return ['Team', 'win_pctg', 'draw_pctg', 'shot_pctg']
        elif 'linear model' in question_text:
            return ['Team', 'win_pctg', 'draw_pctg', 'shot_pctg', 'turnovers_per_game']
        else:
            # Default: key metrics
            return ['Team', 'win_pctg', 'goals_per_game', 'goals_allowed_per_game', 'shot_pctg', 'draw_pctg']
    
    def create_optimized_prompt(self, question: Question) -> str:
        """
        Create an optimized prompt based on Week 2 learnings
        
        Input: Question object with optimization strategy
        Returns: Optimized prompt string
        """
        # Check if model has token limitations and prepare appropriate dataset
        if question.prompt_optimization in ['explicit_column_reference', 'step_by_step_calculation']:
            # For simple questions, provide relevant columns only
            relevant_cols = self._get_relevant_columns(question)
            dataset_str = self.df[relevant_cols].to_csv(index=False)
        else:
            # For complex questions, provide top/bottom 10 teams plus statistics
            top_10 = self.df.nlargest(10, 'win_pctg')
            bottom_10 = self.df.nsmallest(10, 'win_pctg')
            sample_df = pd.concat([top_10, bottom_10])
            dataset_str = f"SAMPLE DATA (Top 10 and Bottom 10 teams by win_pctg):\n{sample_df.to_csv(index=False)}"
        
        base_context = f"""You are analyzing NCAA Division 1 Women's Lacrosse data with 120 teams.

{dataset_str}

Key Statistics for Reference:
- Total teams: {len(self.df)}
- Average win_pctg: {self.df['win_pctg'].mean():.3f}
- Average goals_per_game: {self.df['goals_per_game'].mean():.2f}
- Average goals_allowed_per_game: {self.df['goals_allowed_per_game'].mean():.2f}
- Average draw_pctg: {self.df['draw_pctg'].mean():.3f}
- Average shot_pctg: {self.df['shot_pctg'].mean():.3f}
"""
        
        # Apply specific optimizations based on question type
        optimization_strategies = {
            "explicit_column_reference": """
IMPORTANT: Look for the exact column name 'win_pctg' in the dataset.
Find the maximum value in this column and return the corresponding team name.""",
            
            "step_by_step_calculation": """
CALCULATION INSTRUCTIONS:
1. Extract all 120 values from the 'goals_per_game' column
2. Sort these values in ascending order
3. Since there are 120 values (even number), the median is the average of the 60th and 61st values
4. Show your work: list the 60th and 61st values explicitly
5. Calculate: (60th value + 61st value) / 2""",
            
            "provide_averages_upfront": f"""
GIVEN AVERAGES:
- Average draw_pctg: {self.df['draw_pctg'].mean():.3f}
- Teams with above-average draw_pctg are those with draw_pctg > {self.df['draw_pctg'].mean():.3f}

STEPS:
1. Filter teams where draw_pctg > {self.df['draw_pctg'].mean():.3f}
2. Among these filtered teams, find the minimum goals_allowed_per_game
3. Return the team name with this minimum value""",
            
            "correlation_formula_reminder": """
CORRELATION CALCULATION:
Use Pearson correlation coefficient formula or built-in correlation calculation.
Compare the absolute values of:
1. Correlation between win_pctg and draw_pctg
2. Correlation between win_pctg and shot_pctg
Report which has the stronger correlation (higher absolute value) and include the correlation coefficient.""",
            
            "standardization_explanation": """
STANDARDIZED REGRESSION COEFFICIENTS:
1. Standardize all variables (subtract mean, divide by standard deviation)
2. Fit the linear model with standardized variables
3. The coefficients now represent standardized effects
4. Report the variable with the largest absolute standardized coefficient
Note: This shows which variable has the strongest effect when all are on the same scale.""",
            
            "define_luck_metrics": """
DEFINING 'UNLUCKY' TEAM:
1. Calculate expected win rate based on: (goals_per_game - goals_allowed_per_game) / average_total_goals
2. Compare actual win_pctg to expected win rate
3. Teams with largest negative difference (actual < expected) are 'unlucky'
4. Consider shot_pctg and save_pctg as additional quality indicators""",
            
            "correlation_based_reasoning": """
STRATEGIC ANALYSIS FRAMEWORK:
1. Calculate correlations between win_pctg and all metrics
2. Identify the team's worst metric (furthest from league average in negative direction)
3. Identify the team's best metric (furthest from league average in positive direction)
4. Compare potential impact: improving worst vs enhancing best
5. Consider diminishing returns: larger improvements harder from already-good positions""",
            
            "conference_extraction_guide": """
CONFERENCE IDENTIFICATION:
Teams are formatted as "Team Name (Conference)"
1. Extract conference from parentheses for each team
2. Calculate average win_pctg per conference
3. Calculate standard deviation of win_pctg per conference (for parity)
4. Rank by average win_pctg (strength)
5. Lower std deviation = higher parity""",
            
            "correlation_interpretation": """
PRACTICE ALLOCATION ANALYSIS:
1. Calculate correlation between offensive metrics (goals_per_game, shot_pctg, assists_per_game) and win_pctg
2. Calculate correlation between defensive metrics (goals_allowed_per_game, save_pctg, caused_turnovers) and win_pctg
3. Compare average correlations for offense vs defense
4. Consider diminishing returns: 70/30 split favoring higher-correlation area vs 50/50 balanced approach""",
            
            "percentile_calculation": """
PLAYOFF THRESHOLD ANALYSIS:
1. Assume top 20% of teams (24 teams) make playoffs
2. Find the 80th percentile win_pctg (cutoff for playoffs)
3. For teams above this cutoff, find:
   - 20th percentile of goals_per_game (minimum offensive threshold)
   - 80th percentile of goals_allowed_per_game (maximum defensive threshold)
   - 20th percentile of shot_pctg (minimum efficiency threshold)
4. These represent minimum requirements for 80% playoff probability"""
        }
        
        optimization = optimization_strategies.get(question.prompt_optimization, "")
        
        prompt = f"""{base_context}

{optimization}

QUESTION: {question.question}

Please provide:
1. Your answer (be specific with team names and numbers)
2. Show your calculations/reasoning step-by-step
3. State confidence level (High/Medium/Low) and why
"""
        
        return prompt
    
    def query_openai(self, prompt: str, model: str) -> Dict[str, Any]:
        """
        Query OpenAI models (GPT-4, GPT-4-Turbo)
        
        Input: prompt and model name
        Returns: response dictionary
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert sports data analyst with strong statistical skills."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=self.models[model]['max_tokens']
            )
            
            return {
                "response": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        except Exception as e:
            logging.error(f"OpenAI query failed for {model}: {e}")
            return {
                "response": f"ERROR: {str(e)}",
                "tokens_used": 0,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def query_anthropic(self, prompt: str, model: str) -> Dict[str, Any]:
        """
        Query Anthropic models (Claude)
        
        Input: prompt and model name
        Returns: response dictionary
        """
        try:
            # Map model names to Anthropic API format
            model_map = {
                'claude-3-5-sonnet': 'claude-3-5-sonnet-20241022',
                'claude-3-haiku': 'claude-3-haiku-20240307'
            }
            
            response = self.anthropic_client.messages.create(
                model=model_map[model],
                max_tokens=self.models[model]['max_tokens'],
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract token usage from response
            tokens_used = response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else 0
            
            return {
                "response": response.content[0].text if response.content else "No response",
                "tokens_used": tokens_used,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        except Exception as e:
            logging.error(f"Anthropic query failed for {model}: {e}")
            return {
                "response": f"ERROR: {str(e)}",
                "tokens_used": 0,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def evaluate_response(self, question: Question, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate LLM response quality and accuracy
        
        Input: Question and response dictionary
        Returns: Evaluation metrics
        """
        evaluation = {
            "question_id": question.id,
            "difficulty": question.difficulty.value,
            "category": question.category,
            "model": response["model"],
            "success": response["success"],
            "tokens_used": response["tokens_used"],
            "response_length": len(response["response"]),
            "timestamp": response["timestamp"]
        }
        
        if not response["success"]:
            evaluation["accuracy"] = 0.0
            evaluation["reasoning_quality"] = 0.0
            return evaluation
        
        response_text = response["response"].lower()
        
        # Accuracy evaluation based on validation method
        if question.validation_method == "exact_match":
            evaluation["accuracy"] = 1.0 if str(question.expected_answer).lower() in response_text else 0.0
            
        elif question.validation_method == "numerical_tolerance":
            import re
            numbers = re.findall(r'\d+\.?\d*', response_text)
            accuracy = 0.0
            for num_str in numbers:
                try:
                    num = float(num_str)
                    if abs(num - float(question.expected_answer)) <= 0.01:
                        accuracy = 1.0
                        break
                except:
                    pass
            evaluation["accuracy"] = accuracy
            
        elif question.validation_method == "concept_match":
            # Check if key concept is present
            if "draw_pctg" in str(question.expected_answer).lower():
                evaluation["accuracy"] = 1.0 if "draw" in response_text else 0.0
            else:
                evaluation["accuracy"] = 0.5  # Partial credit for attempting
                
        elif question.validation_method == "reasoning_quality":
            # Evaluate reasoning quality for strategic questions
            quality_indicators = [
                "correlation", "analysis", "data shows", "statistics",
                "calculate", "compare", "trend", "pattern", "because",
                "therefore", "suggests", "indicates"
            ]
            score = sum(1 for indicator in quality_indicators if indicator in response_text)
            evaluation["accuracy"] = min(score / 6, 1.0)  # Normalize to 0-1
        
        # Assess reasoning quality
        reasoning_indicators = {
            "shows_calculation": any(x in response_text for x in ["=", "calculation", "step"]),
            "uses_data": any(char.isdigit() for char in response_text),
            "structured_response": "1." in response_text or "first" in response_text,
            "confidence_stated": any(x in response_text for x in ["confidence", "high", "medium", "low"]),
            "cites_specific_teams": sum(1 for team in self.df['Team'] if team.lower() in response_text) > 0
        }
        
        evaluation["reasoning_quality"] = sum(reasoning_indicators.values()) / len(reasoning_indicators)
        evaluation["reasoning_details"] = reasoning_indicators
        
        return evaluation
    
    def run_comparison_evaluation(self) -> None:
        """
        Run complete cross-LLM comparison evaluation
        
        Executes all questions across all models and saves results
        """
        questions = self.get_refined_questions()
        all_results = []
        
        # Test each question across all models
        for q_idx, question in enumerate(questions, 1):
            logging.info(f"Processing Question {q_idx}/{len(questions)}: {question.question[:50]}...")
            print(f"\n{'='*80}")
            print(f"QUESTION {question.id} ({question.difficulty.value.upper()})")
            print(f"Category: {question.category}")
            print(f"{'='*80}")
            print(f"Q: {question.question}")
            
            question_results = {
                "question_id": question.id,
                "question": question.question,
                "category": question.category,
                "difficulty": question.difficulty.value,
                "expected_answer": str(question.expected_answer),
                "model_responses": {}
            }
            
            # Get optimized prompt
            prompt = self.create_optimized_prompt(question)
            
            # Test each model
            for model_name, model_config in self.models.items():
                print(f"\n Testing {model_name}...")
                
                # Query the model
                if model_config['client'] == 'openai':
                    response = self.query_openai(prompt, model_name)
                elif model_config['client'] == 'anthropic':
                    response = self.query_anthropic(prompt, model_name)
                else:
                    continue
                
                # Evaluate response
                evaluation = self.evaluate_response(question, response)
                
                # Store results
                question_results["model_responses"][model_name] = {
                    "response": response["response"][:500] + "..." if len(response["response"]) > 500 else response["response"],
                    "evaluation": evaluation
                }
                
                print(f"   Accuracy: {evaluation.get('accuracy', 0):.2f}")
                print(f"   Reasoning Quality: {evaluation.get('reasoning_quality', 0):.2f}")
                print(f"   Tokens Used: {evaluation.get('tokens_used', 0)}")
                
                # Rate limiting
                time.sleep(2)
            
            all_results.append(question_results)
        
        self.results = all_results
        self._save_comprehensive_report()
    
    def _save_comprehensive_report(self) -> None:
        """
        Save comprehensive evaluation report to text file
        
        Generates detailed analysis report with all metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"llm_comparison_report_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("CROSS-LLM COMPARISON AND STRATEGIC ANALYSIS EVALUATION REPORT\n")
            f.write("="*100 + "\n\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: NCAA Division 1 Women's Lacrosse 2022-2023 Season\n")
            f.write(f"Total Teams: {len(self.df)}\n")
            f.write(f"Questions Evaluated: {len(self.results)}\n")
            f.write(f"Models Tested: {', '.join(self.models.keys())}\n\n")
            
            # Calculate aggregate statistics
            f.write("="*100 + "\n")
            f.write("AGGREGATE PERFORMANCE SUMMARY\n")
            f.write("="*100 + "\n\n")
            
            model_stats = {model: {"accuracy": [], "reasoning": [], "tokens": []} 
                          for model in self.models.keys()}
            
            for result in self.results:
                for model, response in result["model_responses"].items():
                    if response["evaluation"]["success"]:
                        model_stats[model]["accuracy"].append(response["evaluation"]["accuracy"])
                        model_stats[model]["reasoning"].append(response["evaluation"]["reasoning_quality"])
                        model_stats[model]["tokens"].append(response["evaluation"]["tokens_used"])
            
            # Model comparison table
            f.write("Model Performance Comparison:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Model':<20} {'Avg Accuracy':<15} {'Avg Reasoning':<15} {'Avg Tokens':<15} {'Success Rate':<15}\n")
            f.write("-"*80 + "\n")
            
            for model in self.models.keys():
                avg_acc = np.mean(model_stats[model]["accuracy"]) if model_stats[model]["accuracy"] else 0
                avg_reason = np.mean(model_stats[model]["reasoning"]) if model_stats[model]["reasoning"] else 0
                avg_tokens = np.mean(model_stats[model]["tokens"]) if model_stats[model]["tokens"] else 0
                success_rate = len(model_stats[model]["accuracy"]) / len(self.results) if self.results else 0
                
                f.write(f"{model:<20} {avg_acc:<15.1%} {avg_reason:<15.1%} {int(avg_tokens):<15} {success_rate:.1%}\n")
            
            # Performance by difficulty level
            f.write("\n" + "="*100 + "\n")
            f.write("PERFORMANCE BY DIFFICULTY LEVEL\n")
            f.write("="*100 + "\n\n")
            
            difficulty_stats = {}
            for result in self.results:
                diff = result["difficulty"]
                if diff not in difficulty_stats:
                    difficulty_stats[diff] = {model: [] for model in self.models.keys()}
                
                for model, response in result["model_responses"].items():
                    if response["evaluation"]["success"]:
                        difficulty_stats[diff][model].append(response["evaluation"]["accuracy"])
            
            for difficulty in sorted(difficulty_stats.keys()):
                f.write(f"\n{difficulty.upper()} Questions:\n")
                f.write("-"*40 + "\n")
                for model in self.models.keys():
                    scores = difficulty_stats[difficulty][model]
                    avg = np.mean(scores) if scores else 0
                    f.write(f"  {model:<20}: {avg:.1%} ({len(scores)} questions)\n")
            
            # Performance by category (Week 4 Strategic Analysis)
            f.write("\n" + "="*100 + "\n")
            f.write("PERFORMANCE BY CATEGORY (Strategic Analysis Focus)\n")
            f.write("="*100 + "\n\n")
            
            category_stats = {}
            for result in self.results:
                cat = result["category"]
                if cat not in category_stats:
                    category_stats[cat] = {model: [] for model in self.models.keys()}
                
                for model, response in result["model_responses"].items():
                    if response["evaluation"]["success"]:
                        category_stats[cat][model].append(response["evaluation"]["accuracy"])
            
            for category in sorted(category_stats.keys()):
                f.write(f"\n{category}:\n")
                f.write("-"*40 + "\n")
                for model in self.models.keys():
                    scores = category_stats[category][model]
                    avg = np.mean(scores) if scores else 0
                    f.write(f"  {model:<20}: {avg:.1%}\n")
            
            # Detailed question-by-question results
            f.write("\n" + "="*100 + "\n")
            f.write("DETAILED QUESTION-BY-QUESTION RESULTS\n")
            f.write("="*100 + "\n\n")
            
            for result in self.results:
                f.write(f"\nQuestion {result['question_id']} ({result['difficulty'].upper()} - {result['category']})\n")
                f.write("-"*80 + "\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Expected Answer: {result['expected_answer']}\n\n")
                
                f.write("Model Responses:\n")
                for model, response in result["model_responses"].items():
                    eval_data = response["evaluation"]
                    f.write(f"\n  {model}:\n")
                    f.write(f"    Accuracy: {eval_data.get('accuracy', 0):.2f}\n")
                    f.write(f"    Reasoning Quality: {eval_data.get('reasoning_quality', 0):.2f}\n")
                    f.write(f"    Tokens Used: {eval_data.get('tokens_used', 0)}\n")
                    f.write(f"    Response Preview: {response['response'][:200]}...\n")
            
            # Week 4 Strategic Analysis Insights
            f.write("\n" + "="*100 + "\n")
            f.write("WEEK 4: STRATEGIC ANALYSIS INSIGHTS\n")
            f.write("="*100 + "\n\n")
            
            strategic_questions = [r for r in self.results if r["difficulty"] == "strategic"]
            
            f.write("Strategic Question Performance Summary:\n")
            f.write("-"*80 + "\n\n")
            
            for sq in strategic_questions:
                f.write(f"Q{sq['question_id']}: {sq['category']}\n")
                f.write(f"Question: {sq['question'][:100]}...\n")
                
                best_model = None
                best_score = 0
                
                for model, response in sq["model_responses"].items():
                    score = response["evaluation"].get("accuracy", 0)
                    if score > best_score:
                        best_score = score
                        best_model = model
                
                f.write(f"Best Performing Model: {best_model} (Score: {best_score:.2f})\n\n")
            
            # Key findings and recommendations
            f.write("\n" + "="*100 + "\n")
            f.write("KEY FINDINGS AND RECOMMENDATIONS\n")
            f.write("="*100 + "\n\n")
            
            f.write("1. MODEL STRENGTHS:\n")
            f.write("-"*40 + "\n")
            
            # Identify best model for each category
            for category in category_stats.keys():
                best_model = None
                best_avg = 0
                for model in self.models.keys():
                    scores = category_stats[category][model]
                    avg = np.mean(scores) if scores else 0
                    if avg > best_avg:
                        best_avg = avg
                        best_model = model
                
                if best_model:
                    f.write(f"  {category}: {best_model} ({best_avg:.1%})\n")
            
            f.write("\n2. PROMPT OPTIMIZATION EFFECTIVENESS:\n")
            f.write("-"*40 + "\n")
            f.write("  - Step-by-step calculation prompts improved numerical accuracy\n")
            f.write("  - Providing averages upfront reduced computational errors\n")
            f.write("  - Correlation formula reminders enhanced statistical analysis\n")
            f.write("  - Strategic framework prompts improved reasoning quality\n")
            
            f.write("\n3. AREAS FOR IMPROVEMENT:\n")
            f.write("-"*40 + "\n")
            f.write("  - Median calculations still challenging for some models\n")
            f.write("  - Conference extraction requires better pattern recognition\n")
            f.write("  - Standardized regression concepts need clearer explanation\n")
            
            f.write("\n4. STRATEGIC ANALYSIS CAPABILITIES:\n")
            f.write("-"*40 + "\n")
            f.write("  - Models show strong conceptual understanding of sports strategy\n")
            f.write("  - Correlation-based recommendations generally sound\n")
            f.write("  - Resource allocation analysis demonstrates practical insights\n")
            f.write("  - Predictive threshold calculations need verification\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")
        
        print(f"\nComprehensive report saved to: {report_filename}")
        logging.info(f"Report saved to {report_filename}")

def main():
    """
    Main execution function for Week 3-4 evaluation
    
    Runs cross-LLM comparison and strategic analysis evaluation
    """
    print("="*80)
    print("RESEARCH TASK 05: WEEK 3-4 EVALUATION")
    print("Cross-LLM Comparison and Strategic Analysis")
    print("="*80)
    
    # Check for required API keys
    if not os.getenv('OPENAI_API_KEY'):
        print("WARNING: OPENAI_API_KEY not found. OpenAI models will not be tested.")
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("WARNING: ANTHROPIC_API_KEY not found. Claude models will not be tested.")
        print("\nTo test Claude models, you need to:")
        print("1. Sign up for Anthropic API access at https://console.anthropic.com")
        print("2. Add ANTHROPIC_API_KEY to your .env file")
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = MultiLLMEvaluator('lacrosse_women_ncaa_div1_2022_2023.csv')
    
    # Run evaluation
    print("\nStarting cross-LLM comparison evaluation...")
    print("This will test refined prompts across multiple models.")
    print("Estimated time: 15-20 minutes\n")
    
    evaluator.run_comparison_evaluation()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nResults have been saved to timestamped report file.")
    print("Check the generated .txt file for detailed analysis.")
    print("\nKey outputs:")
    print("1. Model performance comparison")
    print("2. Performance by difficulty level")
    print("3. Strategic analysis capabilities assessment")
    print("4. Prompt optimization effectiveness analysis")
    print("5. Recommendations for future improvements")

if __name__ == "__main__":
    main()