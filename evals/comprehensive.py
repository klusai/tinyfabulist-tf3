"""
Comprehensive evaluator that combines all evaluation metrics
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .base import BaseEvaluator, EvaluationResult, EvaluationConfig
from .perplexity import PerplexityEvaluator
from .text_quality import TextQualityEvaluator
from .fluency import FluencyEvaluator
from .fable_structure import FableStructureEvaluator
from .semantic_coherence import SemanticCoherenceEvaluator


class ComprehensiveEvaluator(BaseEvaluator):
    """Comprehensive evaluator that runs all evaluation metrics"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(config)
        
        # Initialize all evaluators
        self.perplexity_evaluator = PerplexityEvaluator(config)
        self.text_quality_evaluator = TextQualityEvaluator(config)
        self.fluency_evaluator = FluencyEvaluator(config)
        self.fable_structure_evaluator = FableStructureEvaluator(config)
        self.semantic_coherence_evaluator = SemanticCoherenceEvaluator(config)
        
        # Define which evaluators to run (can be configured)
        self.evaluators = {
            'perplexity': self.perplexity_evaluator,
            'text_quality': self.text_quality_evaluator,
            'fluency': self.fluency_evaluator,
            'fable_structure': self.fable_structure_evaluator,
            'semantic_coherence': self.semantic_coherence_evaluator
        }
        
        self.enabled_evaluators = list(self.evaluators.keys())
    
    def set_enabled_evaluators(self, evaluator_names: List[str]):
        """Set which evaluators to run"""
        valid_names = set(self.evaluators.keys())
        self.enabled_evaluators = [name for name in evaluator_names if name in valid_names]
        
        if not self.enabled_evaluators:
            self._log("Warning: No valid evaluators specified, using all evaluators")
            self.enabled_evaluators = list(self.evaluators.keys())
    
    def calculate_overall_score(self, individual_results: Dict[str, EvaluationResult]) -> Dict[str, float]:
        """
        Calculate interpretable individual metrics without arbitrary composite scoring.
        Reports metrics separately for transparency and research validity.
        """
        
        # Extract individual metrics without arbitrary weighting
        metrics = {}
        
        # Language modeling capability (perplexity-based)
        if 'perplexity' in individual_results:
            perplexity_metrics = individual_results['perplexity'].metrics
            raw_perplexity = perplexity_metrics.get('average_perplexity', float('inf'))
            metrics['raw_perplexity'] = raw_perplexity
            metrics['bits_per_char'] = perplexity_metrics.get('average_bits_per_char', 0.0)
            
            # Log-scale normalization for interpretability (not arbitrary thresholds)
            if raw_perplexity > 0:
                metrics['log_perplexity'] = np.log(raw_perplexity)
            else:
                metrics['log_perplexity'] = float('inf')
        
        # Semantic similarity (BERTScore only - no BLEU/ROUGE)
        if 'text_quality' in individual_results:
            quality_metrics = individual_results['text_quality'].metrics
            metrics['bert_f1'] = quality_metrics.get('bert_f1', 0.0)
            metrics['bert_precision'] = quality_metrics.get('bert_precision', 0.0)
            metrics['bert_recall'] = quality_metrics.get('bert_recall', 0.0)
            metrics['vocab_diversity'] = quality_metrics.get('vocab_diversity', 0.0)
            metrics['length_ratio'] = quality_metrics.get('length_ratio', 0.0)
        
        # Fluency metrics (interpretable)
        if 'fluency' in individual_results:
            fluency_metrics = individual_results['fluency'].metrics
            metrics['repetition_ratio'] = fluency_metrics.get('avg_repetition_ratio', 0.0)
            metrics['type_token_ratio'] = fluency_metrics.get('avg_ttr', 0.0)
            metrics['coherence_score'] = fluency_metrics.get('avg_coherence_score', 0.0)
            
            # Flag high repetition as quality issue
            if metrics['repetition_ratio'] > 0.3:
                metrics['high_repetition_flag'] = True
            else:
                metrics['high_repetition_flag'] = False
        
        # Fable-specific structure
        if 'fable_structure' in individual_results:
            structure_metrics = individual_results['fable_structure'].metrics
            metrics['narrative_flow'] = structure_metrics.get('avg_narrative_flow_score', 0.0)
            metrics['moral_clarity'] = structure_metrics.get('avg_moral_clarity_score', 0.0)
            metrics['has_conflict_rate'] = structure_metrics.get('has_conflict_percentage', 0.0) / 100.0
            metrics['has_resolution_rate'] = structure_metrics.get('has_resolution_percentage', 0.0) / 100.0
        
        # Semantic coherence
        if 'semantic_coherence' in individual_results:
            semantic_metrics = individual_results['semantic_coherence'].metrics
            metrics['semantic_coherence'] = semantic_metrics.get('overall_semantic_coherence', 0.5)
            metrics['topic_consistency'] = semantic_metrics.get('avg_topic_consistency', 0.5)
            metrics['content_appropriateness'] = semantic_metrics.get('avg_appropriateness', 0.5)
            metrics['fable_relevance'] = semantic_metrics.get('avg_fable_relevance', 0.5)
            metrics['appropriate_sample_rate'] = semantic_metrics.get('appropriate_sample_rate', 1.0)
        
        # Quality assessment flags (interpretable, not arbitrary scores)
        quality_flags = []
        
        if metrics.get('repetition_ratio', 0.0) > 0.4:
            quality_flags.append("high_repetition")
        
        if metrics.get('semantic_coherence', 0.5) < 0.3:
            quality_flags.append("low_coherence")
        
        if metrics.get('appropriate_sample_rate', 1.0) < 0.7:
            quality_flags.append("inappropriate_content")
        
        if metrics.get('topic_consistency', 0.5) < 0.3:
            quality_flags.append("off_topic")
        
        metrics['quality_flags'] = quality_flags
        metrics['num_quality_issues'] = len(quality_flags)
        
        # Simple quality categorization based on interpretable thresholds
        if len(quality_flags) >= 3:
            metrics['quality_category'] = "poor"
        elif len(quality_flags) >= 2:
            metrics['quality_category'] = "needs_improvement"
        elif len(quality_flags) == 1:
            metrics['quality_category'] = "fair"
        elif (metrics.get('semantic_coherence', 0.5) > 0.7 and 
              metrics.get('repetition_ratio', 0.0) < 0.2):
            metrics['quality_category'] = "good"
        else:
            metrics['quality_category'] = "acceptable"
        
        return metrics
    
    def create_summary_report(self, individual_results: Dict[str, EvaluationResult], 
                            overall_metrics: Dict[str, float]) -> str:
        """Create a human-readable summary report with interpretable metrics"""
        
        report_lines = [
            "=" * 80,
            "COMPREHENSIVE EVALUATION REPORT",
            "=" * 80,
            f"Model: {self.config.model_name}",
            f"Dataset: {self.config.dataset_name}",
            f"Samples: {self.config.num_samples}",
            f"Temperature: {self.config.temperature}",
            "",
            "INTERPRETABLE METRICS (No Arbitrary Composite Scores)",
            "-" * 60,
        ]
        
        # Language modeling metrics
        if 'perplexity' in individual_results:
            perp_metrics = individual_results['perplexity'].metrics
            report_lines.extend([
                "LANGUAGE MODELING",
                "-" * 20,
                f"Raw Perplexity: {overall_metrics.get('raw_perplexity', 'N/A')}",
                f"Log Perplexity: {overall_metrics.get('log_perplexity', 'N/A'):.3f}",
                f"Bits per Character: {overall_metrics.get('bits_per_char', 'N/A'):.3f}",
                ""
            ])
        
        # Semantic similarity (BERTScore only)
        if 'text_quality' in individual_results:
            report_lines.extend([
                "SEMANTIC SIMILARITY",
                "-" * 20,
                f"BERTScore F1: {overall_metrics.get('bert_f1', 'N/A'):.4f}",
                f"BERTScore Precision: {overall_metrics.get('bert_precision', 'N/A'):.4f}",
                f"BERTScore Recall: {overall_metrics.get('bert_recall', 'N/A'):.4f}",
                f"Vocabulary Diversity: {overall_metrics.get('vocab_diversity', 'N/A'):.4f}",
                f"Length Ratio (gen/ref): {overall_metrics.get('length_ratio', 'N/A'):.2f}",
                ""
            ])
        
        # Fluency and repetition
        if 'fluency' in individual_results:
            report_lines.extend([
                "FLUENCY & REPETITION",
                "-" * 20,
                f"Repetition Ratio: {overall_metrics.get('repetition_ratio', 'N/A'):.4f}",
                f"Type-Token Ratio: {overall_metrics.get('type_token_ratio', 'N/A'):.4f}",
                f"Coherence Score: {overall_metrics.get('coherence_score', 'N/A'):.4f}",
                f"High Repetition Flag: {overall_metrics.get('high_repetition_flag', False)}",
                ""
            ])
        
        # Fable structure
        if 'fable_structure' in individual_results:
            report_lines.extend([
                "FABLE STRUCTURE",
                "-" * 20,
                f"Narrative Flow: {overall_metrics.get('narrative_flow', 'N/A'):.4f}",
                f"Moral Clarity: {overall_metrics.get('moral_clarity', 'N/A'):.4f}",
                f"Has Conflict Rate: {overall_metrics.get('has_conflict_rate', 'N/A'):.2%}",
                f"Has Resolution Rate: {overall_metrics.get('has_resolution_rate', 'N/A'):.2%}",
                ""
            ])
        
        # Semantic coherence
        if 'semantic_coherence' in individual_results:
            report_lines.extend([
                "SEMANTIC COHERENCE",
                "-" * 20,
                f"Semantic Coherence: {overall_metrics.get('semantic_coherence', 'N/A'):.4f}",
                f"Topic Consistency: {overall_metrics.get('topic_consistency', 'N/A'):.4f}",
                f"Content Appropriateness: {overall_metrics.get('content_appropriateness', 'N/A'):.4f}",
                f"Fable Relevance: {overall_metrics.get('fable_relevance', 'N/A'):.4f}",
                f"Appropriate Sample Rate: {overall_metrics.get('appropriate_sample_rate', 'N/A'):.2%}",
                ""
            ])
        
        # Quality assessment
        quality_flags = overall_metrics.get('quality_flags', [])
        quality_category = overall_metrics.get('quality_category', 'unknown')
        
        report_lines.extend([
            "QUALITY ASSESSMENT",
            "-" * 20,
            f"Quality Category: {quality_category.upper()}",
            f"Number of Quality Issues: {overall_metrics.get('num_quality_issues', 0)}",
        ])
        
        if quality_flags:
            report_lines.append("Quality Flags:")
            for flag in quality_flags:
                report_lines.append(f"  • {flag.replace('_', ' ').title()}")
        else:
            report_lines.append("No major quality issues detected")
        
        report_lines.extend([
            "",
            "INTERPRETATION NOTES",
            "-" * 20,
            "• Metrics are reported individually for transparency",
            "• No arbitrary composite scores or magic number weights",
            "• BERTScore measures semantic similarity to reference",
            "• Quality flags indicate specific issues, not overall scores",
            "• Perplexity indicates language modeling capability",
            "• All thresholds are interpretable and domain-motivated",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def evaluate(self, model, tokenizer, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """Run comprehensive evaluation using all enabled evaluators"""
        
        self._log(f"Running comprehensive evaluation with {len(self.enabled_evaluators)} evaluators...")
        
        # --- Centralized Text Generation ---
        # Generate completions once for all evaluators that need it.
        # Perplexity is a special case as it evaluates prompts directly.
        
        evaluators_that_need_completions = [
            name for name in self.enabled_evaluators if name != 'perplexity'
        ]
        
        completions = []
        if evaluators_that_need_completions:
            self._log("Generating text completions for all relevant evaluators...")
            completions = self._generate_completions(model, tokenizer, test_data)
            self._log("✓ Completions generated.")
        
        # --- Run Individual Evaluators ---
        individual_results = {}
        
        for evaluator_name in self.enabled_evaluators:
            self._log(f"Running {evaluator_name} evaluation...")
            evaluator = self.evaluators[evaluator_name]
            
            # Use completions if the evaluator needs them, otherwise use original test_data
            if evaluator_name == 'perplexity':
                result = evaluator.evaluate(model, tokenizer, test_data)
            else:
                result = evaluator.evaluate(model, tokenizer, completions)
                
            individual_results[evaluator_name] = result
        
        # --- Aggregation and Reporting ---
        
        # Collect sample results for detailed reporting
        all_sample_results = []
        if completions:
             # Use the generated completions as the base for sample-level results
            all_sample_results = [
                {
                    "sample_id": i,
                    "prompt": sample.get('fable', ''),
                    "generated_text": sample.get('generated_text', '')
                } for i, sample in enumerate(completions)
            ]
        
        # Merge metrics from each evaluator's samples
        for evaluator_name, result in individual_results.items():
            for i, sample_metrics in enumerate(result.samples):
                if i < len(all_sample_results):
                    # Prefix keys to avoid collisions and add to the main sample result
                    prefixed_metrics = {f"{evaluator_name}_{k}": v for k, v in sample_metrics.items()}
                    all_sample_results[i].update(prefixed_metrics)

        # Calculate overall metrics
        overall_metrics = self.calculate_overall_score(individual_results)
        
        # Create summary report
        summary_report = self.create_summary_report(individual_results, overall_metrics)
        
        # Prepare metadata
        metadata = {
            "model_name": self.config.model_name,
            "evaluators_used": self.enabled_evaluators,
            "individual_results": {name: result.to_dict() for name, result in individual_results.items()},
            "summary_report": summary_report
        }
        
        return EvaluationResult(
            evaluator_name="Comprehensive",
            metrics=overall_metrics,
            samples=all_sample_results[:10],  # Limit samples for output size
            metadata=metadata
        )
    
    def run_statistical_evaluation(self, model, tokenizer, dataset=None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with statistical analysis
        Returns both individual results and statistical summary
        """
        if self.config.num_evaluation_runs <= 1:
            # Single run
            result = self.run(model, tokenizer, dataset)
            return {
                'single_result': result,
                'statistical_summary': None,
                'methodology': self._get_methodology_info()
            }
        
        # Multiple runs for statistical significance
        self._log(f"Running {self.config.num_evaluation_runs} evaluations for statistical analysis...")
        
        all_results = self.run_multiple_evaluations(model, tokenizer, dataset)
        
        # Aggregate statistical analysis
        statistical_summary = self._aggregate_statistical_results(all_results)
        
        # Generate comprehensive report
        report = self.generate_statistical_report(all_results)
        
        return {
            'all_results': all_results,
            'statistical_summary': statistical_summary,
            'report': report,
            'methodology': self._get_methodology_info()
        }
    
    def _get_methodology_info(self) -> Dict[str, Any]:
        """Get methodology information for reporting"""
        return {
            'model_name': self.config.model_name,
            'dataset_name': self.config.dataset_name,
            'num_samples': self.config.num_samples,
            'max_prompt_tokens': self.config.max_prompt_tokens,
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': self.config.temperature,
            'prompt_split_ratio': self.config.prompt_split_ratio,
            'prompt_truncation_strategy': self.config.prompt_truncation_strategy,
            'num_evaluation_runs': self.config.num_evaluation_runs,
            'confidence_level': self.config.confidence_level,
            'length_normalize_metrics': self.config.length_normalize_metrics,
            'enabled_evaluators': self.enabled_evaluators
        }
    
    def _aggregate_statistical_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Aggregate results from multiple runs for statistical analysis
        """
        if len(results) <= 1:
            return {}
        
        # Extract overall scores from each run
        overall_scores = []
        individual_metrics = {}
        
        for result in results:
            if 'overall_score' in result.metrics:
                overall_scores.append(result.metrics['overall_score'])
            
            # Collect all metrics for statistical analysis
            for metric_name, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    if metric_name not in individual_metrics:
                        individual_metrics[metric_name] = []
                    individual_metrics[metric_name].append(float(value))
        
        statistical_summary = {}
        
        # Overall score statistics
        if overall_scores:
            statistical_summary['overall_score'] = {
                'mean': np.mean(overall_scores),
                'std': np.std(overall_scores),
                'min': min(overall_scores),
                'max': max(overall_scores),
                'values': overall_scores
            }
            
            if self.config.report_confidence_intervals:
                try:
                    ci_lower, ci_upper = self.calculate_confidence_interval(overall_scores)
                    statistical_summary['overall_score']['confidence_interval'] = {
                        'lower': ci_lower,
                        'upper': ci_upper,
                        'level': self.config.confidence_level
                    }
                except:
                    pass
        
        # Individual metric statistics
        for metric_name, values in individual_metrics.items():
            if len(values) >= 2:
                statistical_summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                }
                
                if self.config.report_confidence_intervals:
                    try:
                        ci_lower, ci_upper = self.calculate_confidence_interval(values)
                        statistical_summary[metric_name]['confidence_interval'] = {
                            'lower': ci_lower,
                            'upper': ci_upper,
                            'level': self.config.confidence_level
                        }
                    except:
                        pass
        
        return statistical_summary 