"""
Report generation using existing libraries
"""
import numpy as np
from datetime import datetime
import json


class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self):
        self.weights = {
            'deep_learning': 0.50,
            'perceptual_hash': 0.15,
            'cv_methods': 0.25,
            'probabilistic': 0.10
        }
        # Configurable thresholds instead of hardcoded values
        self.thresholds = {
            'high_similarity': 0.8,
            'moderate_similarity': 0.6,
            'low_similarity': 0.4,
            'very_high_method_score': 0.9,
            'low_method_score': 0.3,
            'low_confidence': 0.5,
            'high_confidence': 0.8
        }
    
    def generate_analysis_report(self, results, timestamp=None):
        """Generate a comprehensive analysis report"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        report = {
            'timestamp': timestamp,
            'summary': self._generate_summary(results),
            'method_analysis': self._analyze_methods(results),
            'technical_details': self._get_technical_details(results),
            'recommendations': self._generate_recommendations(results),
            'statistical_analysis': self._statistical_analysis(results)
        }
        
        return report

    @staticmethod
    def _generate_summary(results):
        """Generate a summary section"""
        ensemble = results.get('ensemble', {})
        return {
            'ensemble_score': ensemble.get('score', 0),
            'confidence': ensemble.get('confidence', 0),
            'statistical_significance': ensemble.get('statistical_significance', 0),
            'analysis_timestamp': results.get('timestamp', '')
        }
    
    def _analyze_methods(self, results):
        """Analyze individual methods"""
        method_analysis = {}
        
        for method, weight in self.weights.items():
            if method in results:
                method_data = results[method]
                method_analysis[method] = {
                    'score': method_data.get('score', 0),
                    'confidence': method_data.get('confidence', 0),
                    'method_description': method_data.get('method', ''),
                    'proof': method_data.get('proof', ''),
                    'weight': weight,
                    'contribution': method_data.get('score', 0) * weight,
                    'details': method_data.get('details', {})
                }
        
        return method_analysis
    
    def _get_technical_details(self, results):
        """Get technical details"""
        scores = [results[m]['score'] for m in self.weights.keys() if m in results]
        
        return {
            'total_methods': len(scores),
            'ensemble_weights': self.weights,
            'score_variance': float(np.var(scores)) if scores else 0,
            'method_consistency': self._calculate_consistency(scores),
            'score_range': {
                'min': float(np.min(scores)) if scores else 0,
                'max': float(np.max(scores)) if scores else 0,
                'mean': float(np.mean(scores)) if scores else 0,
                'std': float(np.std(scores)) if scores else 0
            }
        }
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on analysis"""
        recommendations = []
        ensemble_score = results.get('ensemble', {}).get('score', 0)
        
        # Overall similarity recommendations using configurable thresholds
        if ensemble_score > self.thresholds['high_similarity']:
            recommendations.append("High similarity detected - images are very similar")
        elif ensemble_score > self.thresholds['moderate_similarity']:
            recommendations.append("Moderate similarity - images share some common features")
        elif ensemble_score > self.thresholds['low_similarity']:
            recommendations.append("Low similarity - images have limited common features")
        else:
            recommendations.append("Very low similarity - images are quite different")
        
        # Method-specific recommendations
        for method, weight in self.weights.items():
            if method in results:
                score = results[method].get('score', 0)
                method_name = method.replace('_', ' ').title()
                
                if score > self.thresholds['very_high_method_score']:
                    recommendations.append(f"{method_name} method shows very high similarity")
                elif score < self.thresholds['low_method_score']:
                    recommendations.append(f"{method_name} method shows low similarity")
        
        # Confidence-based recommendations using configurable thresholds
        confidence = results.get('ensemble', {}).get('confidence', 0)
        if confidence < self.thresholds['low_confidence']:
            recommendations.append("Low confidence in results - consider using different images")
        elif confidence > self.thresholds['high_confidence']:
            recommendations.append("High confidence in similarity assessment")
        
        return recommendations
    
    def _statistical_analysis(self, results):
        """Perform statistical analysis"""
        scores = [results[m]['score'] for m in self.weights.keys() if m in results]
        
        if len(scores) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Calculate statistical measures
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        cv = std_score / (mean_score + 1e-10)  # Coefficient of variation
        
        return {
            'coefficient_of_variation': float(cv),
            'consistency_score': float(1 - cv),
            'score_distribution': {
                'skewness': float(self._calculate_skewness(scores)),
                'kurtosis': float(self._calculate_kurtosis(scores))
            },
            'outlier_analysis': self._detect_outliers(scores),
            'correlation_analysis': self._correlation_analysis(results)
        }

    @staticmethod
    def _calculate_consistency(scores):
        """Calculate consistency across methods"""
        if len(scores) < 2:
            return 0.0
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        cv = std_score / (mean_score + 1e-10)
        return max(0, min(1, 1 - cv))

    @staticmethod
    def _calculate_skewness(scores):
        """Calculate skewness of scores"""
        if len(scores) < 3:
            return 0.0
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        skewness = np.mean([(x - mean_score) ** 3 for x in scores]) / (std_score ** 3)
        return skewness

    @staticmethod
    def _calculate_kurtosis(scores):
        """Calculate kurtosis of scores"""
        if len(scores) < 4:
            return 0.0
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        kurtosis = np.mean([(x - mean_score) ** 4 for x in scores]) / (std_score ** 4) - 3
        return kurtosis

    @staticmethod
    def _detect_outliers(scores):
        """Detect outliers in scores"""
        if len(scores) < 3:
            return {'outliers': [], 'outlier_count': 0}
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = 2 * std_score
        
        outliers = []
        for i, score in enumerate(scores):
            if abs(score - mean_score) > threshold:
                outliers.append({'index': i, 'score': score, 'deviation': abs(score - mean_score)})
        
        return {
            'outliers': outliers,
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(scores) * 100
        }
    
    def _correlation_analysis(self, results):
        """Analyze correlations between methods"""
        method_scores = {}
        for method in self.weights.keys():
            if method in results:
                method_scores[method] = results[method].get('score', 0)
        
        if len(method_scores) < 2:
            return {'error': 'Insufficient methods for correlation analysis'}
        
        # Calculate pairwise correlations
        methods = list(method_scores.keys())
        correlations = {}
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                # Simple correlation based on score similarity
                score1 = method_scores[method1]
                score2 = method_scores[method2]
                correlation = 1 - abs(score1 - score2)  # Simple similarity measure
                correlations[f"{method1}_vs_{method2}"] = float(correlation)
        
        return {
            'pairwise_correlations': correlations,
            'average_correlation': float(np.mean(list(correlations.values()))) if correlations else 0
        }
    
    def export_report(self, report, format='json'):
        """Export report in specified format"""
        if format.lower() == 'json':
            return json.dumps(report, indent=2, default=str)
        elif format.lower() == 'text':
            return self._format_text_report(report)
        else:
            return str(report)

    @staticmethod
    def _format_text_report(report):
        """Format report as text"""
        text = f"""
IMAGE SIMILARITY ANALYSIS REPORT
Generated: {report['timestamp']}

SUMMARY:
- Overall Similarity: {report['summary']['ensemble_score']:.3f}
- Confidence: {report['summary']['confidence']:.3f}
- Statistical Significance: {report['summary']['statistical_significance']:.3f}

METHOD ANALYSIS:
"""
        
        for method, analysis in report['method_analysis'].items():
            text += f"""
{method.replace('_', ' ').title()}:
  Score: {analysis['score']:.3f}
  Confidence: {analysis['confidence']:.3f}
  Weight: {analysis['weight']:.1%}
  Contribution: {analysis['contribution']:.3f}
  Method: {analysis['method_description']}
"""
        
        text += f"""
TECHNICAL DETAILS:
- Total Methods: {report['technical_details']['total_methods']}
- Score Variance: {report['technical_details']['score_variance']:.3f}
- Method Consistency: {report['technical_details']['method_consistency']:.3f}
- Score Range: {report['technical_details']['score_range']['min']:.3f} - {report['technical_details']['score_range']['max']:.3f}

RECOMMENDATIONS:
"""
        
        for rec in report['recommendations']:
            text += f"- {rec}\n"
        
        return text
