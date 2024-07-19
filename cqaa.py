#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code Quality Aromatherapy Assistant (CQAA)

This script analyzes a GitHub repository's code quality and generates a
"fragrance profile" based on various metrics. It provides detailed insights
and visualizations to help developers improve their code.

Author: Your Name
Date: 2024-07-19
Version: 1.0.0
"""

import os
import json
import random
import re
from collections import defaultdict
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from github import Github
from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.raw import analyze


class CodeMetricsAnalyzer:
    """Analyzes code metrics for a given piece of code."""

    @staticmethod
    def calculate_complexity(content: str) -> int:
        """Calculate cyclomatic complexity of the code."""
        return max(cc_visit(content), default=0)

    @staticmethod
    def calculate_maintainability(content: str) -> float:
        """Calculate maintainability index of the code."""
        return h_visit(content)

    @staticmethod
    def calculate_comment_ratio(content: str) -> float:
        """Calculate the ratio of comments to code."""
        analysis = analyze(content)
        return analysis.comments / analysis.loc if analysis.loc > 0 else 0

    @staticmethod
    def estimate_bug_frequency(content: str) -> int:
        """Estimate potential bug frequency based on certain patterns."""
        risky_patterns = [r'TODO', r'FIXME', r'hack', r'workaround']
        return sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in risky_patterns)


class FragranceProfileGenerator:
    """Generates a fragrance profile based on code metrics."""

    def __init__(self):
        self.scents = {
            "lavender": {"meaning": "calm, clean code", "threshold": 5},
            "peppermint": {"meaning": "refreshing, efficient code", "threshold": 10},
            "lemon": {"meaning": "clean, well-documented code", "threshold": 0.2},
            "eucalyptus": {"meaning": "complex but powerful code", "threshold": 15},
            "rosemary": {"meaning": "memory-intensive code", "threshold": 1000},
            "tea tree": {"meaning": "bug-free code", "threshold": 0.05},
            "frankincense": {"meaning": "legacy code", "threshold": 5},
            "sandalwood": {"meaning": "stable, reliable code", "threshold": 0.7}
        }

    def generate_fragrance(self, metrics: Dict[str, List[float]]) -> Dict[str, int]:
        """Generate a fragrance profile based on code metrics."""
        fragrance = {}
        for scent, info in self.scents.items():
            score = self._calculate_scent_score(scent, metrics)
            if score > info['threshold']:
                fragrance[scent] = min(round(score), 20)  # Cap at 20 drops

        # Normalize to total 20 drops
        total_drops = sum(fragrance.values())
        if total_drops > 20:
            factor = 20 / total_drops
            fragrance = {k: round(v * factor) for k, v in fragrance.items()}

        return fragrance

    def _calculate_scent_score(self, scent: str, metrics: Dict[str, List[float]]) -> float:
        """Calculate the score for a specific scent based on metrics."""
        if scent == "lavender":
            return np.mean(metrics['maintainability'])
        elif scent == "peppermint":
            return 100 / np.mean(metrics['complexity'])
        elif scent == "lemon":
            return np.mean(metrics['comment_ratio']) * 100
        elif scent == "eucalyptus":
            return np.mean(metrics['complexity'])
        elif scent == "rosemary":
            return np.mean(metrics['loc'])
        elif scent == "tea tree":
            return 1 / (np.mean(metrics['bug_frequency']) + 1) * 100
        elif scent == "frankincense":
            return np.percentile(metrics['loc'], 90)
        elif scent == "sandalwood":
            return np.mean(metrics['maintainability']) / 100
        else:
            return 0


class CodeQualityAromatherapyAssistant:
    """Main class for the Code Quality Aromatherapy Assistant."""

    def __init__(self, repo_name: str, token: str):
        self.repo_name = repo_name
        self.g = Github(token)
        self.repo = self.g.get_repo(repo_name)
        self.metrics_analyzer = CodeMetricsAnalyzer()
        self.fragrance_generator = FragranceProfileGenerator()

    def analyze_code(self) -> Dict[str, List[float]]:
        """Analyze the code in the repository."""
        files = self.repo.get_contents("")
        metrics = defaultdict(list)
        for file in files:
            if file.path.endswith('.py'):
                content = file.decoded_content.decode('utf-8')
                metrics['loc'].append(len(content.splitlines()))
                metrics['complexity'].append(self.metrics_analyzer.calculate_complexity(content))
                metrics['maintainability'].append(self.metrics_analyzer.calculate_maintainability(content))
                metrics['comment_ratio'].append(self.metrics_analyzer.calculate_comment_ratio(content))
                metrics['bug_frequency'].append(self.metrics_analyzer.estimate_bug_frequency(content))
        return metrics

    def generate_report(self, metrics: Dict[str, List[float]], fragrance: Dict[str, int]) -> str:
        """Generate a detailed report based on the analysis."""
        report = {
            "repository": self.repo_name,
            "metrics_summary": self._generate_metrics_summary(metrics),
            "fragrance_profile": fragrance,
            "scent_meanings": {scent: self.fragrance_generator.scents[scent]['meaning'] for scent in fragrance},
            "code_quality_summary": self._generate_quality_summary(metrics),
            "improvement_suggestions": self._generate_suggestions(metrics)
        }
        return json.dumps(report, indent=2)

    def _generate_metrics_summary(self, metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Generate a summary of the code metrics."""
        return {
            "avg_complexity": np.mean(metrics['complexity']),
            "avg_maintainability": np.mean(metrics['maintainability']),
            "avg_comment_ratio": np.mean(metrics['comment_ratio']),
            "total_loc": sum(metrics['loc']),
            "estimated_bugs": sum(metrics['bug_frequency'])
        }

    def _generate_quality_summary(self, metrics: Dict[str, List[float]]) -> str:
        """Generate a summary of the overall code quality."""
        avg_complexity = np.mean(metrics['complexity'])
        avg_maintainability = np.mean(metrics['maintainability'])
        
        if avg_complexity < 10 and avg_maintainability > 20:
            return "Your code is clean and well-maintained. Keep up the good work!"
        elif avg_complexity < 20 and avg_maintainability > 10:
            return "Your code quality is good, but there's room for improvement."
        else:
            return "Your code might benefit from some refactoring to improve maintainability."

    def _generate_suggestions(self, metrics: Dict[str, List[float]]) -> List[str]:
        """Generate improvement suggestions based on the metrics."""
        suggestions = []
        if np.mean(metrics['complexity']) > 10:
            suggestions.append("Consider breaking down complex functions to improve readability.")
        if np.mean(metrics['comment_ratio']) < 0.1:
            suggestions.append("Adding more comments could help improve code understanding.")
        if sum(metrics['bug_frequency']) > 10:
            suggestions.append("Address TODOs and FIXMEs to reduce potential bugs.")
        return suggestions

    def generate_visualizations(self, metrics: Dict[str, List[float]]) -> None:
        """Generate visualizations for the code metrics."""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.histplot(metrics['complexity'], kde=True)
        plt.title('Distribution of Code Complexity')
        
        plt.subplot(2, 2, 2)
        sns.histplot(metrics['maintainability'], kde=True)
        plt.title('Distribution of Maintainability Index')
        
        plt.subplot(2, 2, 3)
        sns.scatterplot(x=metrics['loc'], y=metrics['complexity'])
        plt.title('Lines of Code vs Complexity')
        
        plt.subplot(2, 2, 4)
        sns.boxplot(data=[metrics['complexity'], metrics['maintainability'], metrics['comment_ratio']])
        plt.title('Overview of Key Metrics')
        plt.xticks([0, 1, 2], ['Complexity', 'Maintainability', 'Comment Ratio'])
        
        plt.tight_layout()
        plt.savefig('code_quality_visualizations.png')

    def run(self) -> str:
        """Run the full analysis and report generation."""
        metrics = self.analyze_code()
        fragrance = self.fragrance_generator.generate_fragrance(metrics)
        report = self.generate_report(metrics, fragrance)
        self.generate_visualizations(metrics)
        return report


def main():
    """Main function to run the Code Quality Aromatherapy Assistant."""
    repo_name = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    
    if not repo_name or not token:
        print("Error: GITHUB_REPOSITORY and GITHUB_TOKEN environment variables are required.")
        return

    assistant = CodeQualityAromatherapyAssistant(repo_name, token)
    result = assistant.run()
    
    print(result)
    
    with open('code_quality_report.json', 'w') as f:
        f.write(result)


if __name__ == "__main__":
    main()
