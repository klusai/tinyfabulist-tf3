"""
Comprehensive evaluator that combines all evaluation metrics
"""

from typing import Dict, List, Any, Optional
from .base import BaseEvaluator, EvaluationResult, EvaluationConfig, MetricCalculator
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
        """Calculate realistic overall performance scores with proper weighting"""
        
        # Extract key metrics with defaults
        perplexity_score = 0.0
        if 'perplexity' in individual_results:
            perp = individual_results['perplexity'].metrics.get('weighted_perplexity', float('inf'))
            # Normalize perplexity: good = ~10-20, bad = >50
            if perp != float('inf') and perp > 0:
                perplexity_score = max(0.0, min(1.0, (50.0 - perp) / 40.0))
        
        text_quality_score = 0.0
        if 'text_quality' in individual_results:
            quality_metrics = individual_results['text_quality'].metrics
            bleu_score = quality_metrics.get('avg_bleu_4', 0.0)
            rouge_score = quality_metrics.get('avg_rougeL', 0.0)
            bert_score = quality_metrics.get('bert_f1', 0.0)
            text_quality_score = MetricCalculator.mean([bleu_score, rouge_score, bert_score])
        
        # Extract fluency with heavy repetition penalty
        fluency_base_score = 0.0
        repetition_ratio = 0.0
        if 'fluency' in individual_results:
            fluency_metrics = individual_results['fluency'].metrics
            fluency_base_score = fluency_metrics.get('overall_fluency_score', 0.0)
            repetition_ratio = fluency_metrics.get('avg_repetition_ratio', 0.0)
        
        # Apply heavy penalty for repetition
        repetition_penalty = min(0.7, repetition_ratio * 2.0)  # Severe penalty
        fluency_score = max(0.0, fluency_base_score - repetition_penalty)
        
        fable_structure_score = 0.0
        if 'fable_structure' in individual_results:
            structure_metrics = individual_results['fable_structure'].metrics
            fable_structure_score = structure_metrics.get('overall_fable_score', 0.0)
        
        # Semantic coherence (most important for quality assessment)
        semantic_score = 0.5  # Default neutral
        appropriateness_rate = 1.0
        topic_consistency = 0.5
        if 'semantic_coherence' in individual_results:
            semantic_metrics = individual_results['semantic_coherence'].metrics
            semantic_score = semantic_metrics.get('overall_semantic_coherence', 0.5)
            appropriateness_rate = semantic_metrics.get('appropriate_sample_rate', 1.0)
            topic_consistency = semantic_metrics.get('avg_topic_consistency', 0.5)
        
        # Calculate component scores with realistic weighting
        component_scores = {
            'semantic_coherence': semantic_score * 0.35,      # Most important
            'content_appropriateness': appropriateness_rate * 0.20,  # Safety critical
            'fluency_adjusted': fluency_score * 0.20,         # Penalized for repetition
            'fable_structure': fable_structure_score * 0.15,  # Domain-specific
            'text_quality': text_quality_score * 0.10        # Less important than coherence
        }
        
        # Base score from weighted components
        base_score = sum(component_scores.values())
        
        # Additional quality gates
        quality_issues = []
        
        if repetition_ratio > 0.4:
            quality_issues.append(f"Highly repetitive ({repetition_ratio:.1%})")
        
        if semantic_score < 0.3:
            quality_issues.append(f"Incoherent content")
        
        if appropriateness_rate < 0.7:
            quality_issues.append(f"Inappropriate content")
        
        if topic_consistency < 0.3:
            quality_issues.append(f"Off-topic generation")
        
        # Determine realistic overall score and rating
        if len(quality_issues) >= 3 or repetition_ratio > 0.6:
            # Multiple critical issues
            overall_score = min(0.25, base_score * 0.5)
            rating = "Poor"
        elif len(quality_issues) >= 2 or repetition_ratio > 0.5:
            # Some critical issues
            overall_score = min(0.45, base_score * 0.7)
            rating = "Needs Improvement"
        elif quality_issues or semantic_score < 0.5:
            # Minor issues
            overall_score = min(0.65, base_score * 0.85)
            rating = "Fair"
        elif base_score > 0.8 and semantic_score > 0.7:
            # Actually good quality
            overall_score = base_score
            rating = "Good"
        else:
            # Acceptable quality
            overall_score = base_score
            rating = "Acceptable"
        
        # Add quality details to rating
        if quality_issues:
            rating += f" - {'; '.join(quality_issues)}"
        
        return {
            'overall_score': overall_score,
            'quality_rating': rating,
            'semantic_coherence': semantic_score,
            'content_appropriateness': appropriateness_rate,
            'fluency_adjusted': fluency_score,
            'repetition_ratio': repetition_ratio,
            'text_quality': text_quality_score,
            'fable_structure': fable_structure_score,
            'language_modeling': perplexity_score,
            'quality_issues': quality_issues,
            'num_metrics': len(individual_results)
        }
    
    def create_summary_report(self, individual_results: Dict[str, EvaluationResult], 
                            overall_metrics: Dict[str, float]) -> str:
        """Create a human-readable summary report"""
        
        report_lines = [
            "=" * 80,
            "COMPREHENSIVE EVALUATION REPORT",
            "=" * 80,
            f"Model: {self.config.model_name}",
            f"Dataset: {self.config.dataset_name}",
            f"Samples: {self.config.num_samples}",
            f"Temperature: {self.config.temperature}",
            "",
            "OVERALL PERFORMANCE",
            "-" * 40,
            f"Overall Score: {overall_metrics.get('overall_score', 0.0):.4f}",
            ""
        ]
        
        # Individual evaluator summaries
        for evaluator_name, result in individual_results.items():
            report_lines.append(f"{evaluator_name.upper()} METRICS")
            report_lines.append("-" * 40)
            
            # Extract top metrics for each evaluator
            metrics = result.metrics
            
            if evaluator_name == 'perplexity':
                report_lines.extend([
                    f"Weighted Perplexity: {metrics.get('weighted_perplexity', 0.0):.2f}",
                    f"Average Perplexity: {metrics.get('average_perplexity', 0.0):.2f}",
                    f"Bits per Character: {metrics.get('average_bits_per_char', 0.0):.3f}"
                ])
            elif evaluator_name == 'text_quality':
                report_lines.extend([
                    f"BLEU-4: {metrics.get('avg_bleu_4', 0.0):.4f}",
                    f"ROUGE-L: {metrics.get('avg_rougeL', 0.0):.4f}",
                    f"BERTScore F1: {metrics.get('bert_f1', 0.0):.4f}",
                    f"Completion Length: {metrics.get('avg_completion_length', 0.0):.1f} words"
                ])
            elif evaluator_name == 'fluency':
                report_lines.extend([
                    f"Overall Fluency: {metrics.get('overall_fluency_score', 0.0):.4f}",
                    f"Type-Token Ratio: {metrics.get('avg_ttr', 0.0):.4f}",
                    f"Repetition Ratio: {metrics.get('avg_repetition_ratio', 0.0):.4f}",
                    f"Coherence Score: {metrics.get('avg_coherence_score', 0.0):.4f}"
                ])
            elif evaluator_name == 'fable_structure':
                report_lines.extend([
                    f"Overall Fable Score: {metrics.get('overall_fable_score', 0.0):.4f}",
                    f"Narrative Flow: {metrics.get('avg_narrative_flow_score', 0.0):.4f}",
                    f"Moral Clarity: {metrics.get('avg_moral_clarity_score', 0.0):.4f}",
                    f"Has Conflict: {metrics.get('has_conflict_percentage', 0.0):.1f}%",
                    f"Has Resolution: {metrics.get('has_resolution_percentage', 0.0):.1f}%"
                ])
            
            report_lines.append("")
        
        # Add semantic coherence metrics if available
        if 'semantic_coherence' in individual_results:
            semantic_metrics = individual_results['semantic_coherence'].metrics
            report_lines.extend([
                "SEMANTIC COHERENCE METRICS",
                "-" * 40,
                f"Overall Semantic Coherence: {semantic_metrics.get('overall_semantic_coherence', 0.0):.4f}",
                f"Topic Consistency: {semantic_metrics.get('avg_topic_consistency', 0.0):.4f}",
                f"Content Appropriateness: {semantic_metrics.get('avg_appropriateness', 0.0):.4f}",
                f"Fable Relevance: {semantic_metrics.get('avg_fable_relevance', 0.0):.4f}",
                f"Appropriate Sample Rate: {semantic_metrics.get('appropriate_sample_rate', 0.0):.1%}",
                ""
            ])
        
        # Performance interpretation with realistic assessment
        report_lines.extend([
            "PERFORMANCE INTERPRETATION",
            "-" * 40
        ])
        
        quality_rating = overall_metrics.get('quality_rating', 'Unknown')
        repetition_ratio = overall_metrics.get('repetition_ratio', 0.0)
        semantic_score = overall_metrics.get('semantic_coherence', 0.0)
        
        report_lines.append(f"Overall Rating: {quality_rating}")
        
        # Add detailed quality breakdown
        if repetition_ratio > 0.4:
            report_lines.append(f"⚠️  High repetition detected: {repetition_ratio:.1%} of words are repeated")
        
        if semantic_score < 0.5:
            report_lines.append(f"⚠️  Low semantic coherence: {semantic_score:.2f}/1.0")
        
        quality_issues = overall_metrics.get('quality_issues', [])
        if quality_issues:
            report_lines.append("🔍 Identified Quality Issues:")
            for issue in quality_issues:
                report_lines.append(f"   • {issue}")
        
        # Recommendations based on actual performance
        report_lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 40
        ])
        
        recommendations = []
        
        # Priority recommendations based on critical issues
        if repetition_ratio > 0.5:
            recommendations.append("• CRITICAL: Address severe repetition issues - consider adjusting temperature or top-p sampling")
        elif repetition_ratio > 0.3:
            recommendations.append("• HIGH: Reduce repetition through better decoding strategies")
        
        if semantic_score < 0.3:
            recommendations.append("• CRITICAL: Improve semantic coherence - model generates incoherent content")
        elif semantic_score < 0.5:
            recommendations.append("• HIGH: Focus on semantic coherence and topic consistency")
        
        if overall_metrics.get('content_appropriateness', 1.0) < 0.7:
            recommendations.append("• HIGH: Address inappropriate content generation")
        
        # Secondary recommendations
        if overall_metrics.get('language_modeling', 0.0) < 0.4:
            recommendations.append("• Consider fine-tuning to improve language modeling (high perplexity)")
        
        if overall_metrics.get('fable_structure', 0.0) < 0.5:
            recommendations.append("• Improve fable-specific structure and narrative elements")
        
        if overall_metrics.get('text_quality', 0.0) < 0.3:
            recommendations.append("• Focus on improving reference-based quality metrics")
        
        # If no critical issues, add positive feedback
        if not recommendations and overall_metrics.get('overall_score', 0.0) > 0.6:
            recommendations.append("✅ Model shows good performance - consider evaluation on more diverse prompts")
        elif not recommendations:
            recommendations.append("• Model needs comprehensive improvement across multiple dimensions")
        
        report_lines.extend(recommendations)
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def evaluate(self, model, tokenizer, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """Run comprehensive evaluation using all enabled evaluators"""
        
        self._log(f"Running comprehensive evaluation with {len(self.enabled_evaluators)} evaluators...")
        
        # Run individual evaluators
        individual_results = {}
        all_sample_results = []
        
        for evaluator_name in self.enabled_evaluators:
            self._log(f"Running {evaluator_name} evaluation...")
            
            evaluator = self.evaluators[evaluator_name]
            result = evaluator.evaluate(model, tokenizer, test_data)
            individual_results[evaluator_name] = result
            
            # Collect sample results
            if not all_sample_results:  # First evaluator
                all_sample_results = [{"sample_id": i} for i in range(len(result.samples))]
            
            # Merge sample results
            for i, sample in enumerate(result.samples):
                if i < len(all_sample_results):
                    # Add metrics from this evaluator to the sample
                    evaluator_metrics = {f"{evaluator_name}_{k}": v for k, v in sample.items() 
                                       if isinstance(v, (int, float, bool))}
                    all_sample_results[i].update(evaluator_metrics)
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_score(individual_results)
        
        # Combine all metrics
        combined_metrics = overall_metrics.copy()
        
        # Add summary statistics from individual evaluators
        for evaluator_name, result in individual_results.items():
            for metric_name, value in result.metrics.items():
                combined_key = f"{evaluator_name}_{metric_name}"
                combined_metrics[combined_key] = value
        
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
            metrics=combined_metrics,
            samples=all_sample_results[:10],  # Limit samples for output size
            metadata=metadata
        ) 