#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code Quality Aromatherapy Assistant (CQAA)

This script analyzes the code quality of a GitHub repository and generates
a custom "Fragrance profile" based on various metrics. It provides olfactory
feedback on your code's health!

Usage:
    python cqaa.py <repo_owner>/<repo_name> <pr_number>

Environment variables:
    GITHUB_TOKEN: Your GitHub personal access token

Dependencies:
    - numpy
    - matplotlib
    - PyGithub
    - radon
    - pylint
    - coverage
"""

import os
import sys
import json
import logging
import tempfile
import subprocess
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from github import Github, GithubException
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from pylint import epylint as lint
import ast

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """
    Analyzes code quality metrics for a given GitHub repository.
    """

    def __init__(self, repo_name: str):
        """
        Initialize the CodeAnalyzer.

        Args:
            repo_name (str): The full name of the repository (owner/repo).
        """
        self.repo_name = repo_name
        self.g = Github(os.getenv('GITHUB_TOKEN'))
        self.repo = self.g.get_repo(repo_name)

    def analyze_code(self) -> Dict[str, float]:
        """
        Perform a comprehensive analysis of the repository's code.

        Returns:
            Dict[str, float]: A dictionary of various code quality metrics.
        """
        metrics = {
            'complexity': [],
            'maintainability': [],
            'comment_ratio': [],
            'test_coverage': [],
            'code_to_comment_ratio': [],
            'function_length': [],
            'class_length': [],
            'lines_of_code': 0,
            'num_of_files': 0,
            'average_file_size': 0,
        }

        total_size = 0
        for file in self.repo.get_contents(""):
            if file.name.endswith('.py'):
                metrics['num_of_files'] += 1
                total_size += file.size
                content = file.decoded_content.decode('utf-8')
                metrics['lines_of_code'] += len(content.split('\n'))
                
                complexity = self._analyze_complexity(content)
                maintainability = self._analyze_maintainability(content)
                comment_ratio = self._analyze_comment_ratio(content)
                code_to_comment_ratio = self._analyze_code_to_comment_ratio(content)
                function_length = self._analyze_function_length(content)
                class_length = self._analyze_class_length(content)
                
                metrics['complexity'].append(complexity)
                metrics['maintainability'].append(maintainability)
                metrics['comment_ratio'].append(comment_ratio)
                metrics['code_to_comment_ratio'].append(code_to_comment_ratio)
                metrics['function_length'].extend(function_length)
                metrics['class_length'].extend(class_length)

        # Aggregate metrics
        for key in ['complexity', 'maintainability', 'comment_ratio', 'code_to_comment_ratio']:
            metrics[key] = np.mean(metrics[key]) if metrics[key] else 0

        metrics['average_function_length'] = np.mean(metrics['function_length']) if metrics['function_length'] else 0
        metrics['average_class_length'] = np.mean(metrics['class_length']) if metrics['class_length'] else 0
        metrics['average_file_size'] = total_size / metrics['num_of_files'] if metrics['num_of_files'] > 0 else 0

        # Run tests and calculate coverage
        metrics['test_coverage'] = self._run_tests_and_coverage()

        return metrics

    def _analyze_complexity(self, content: str) -> float:
        """
        Analyze the cyclomatic complexity of the code.

        Args:
            content (str): The content of the file.

        Returns:
            float: The average cyclomatic complexity.
        """
        try:
            complexity = radon_cc.cc_visit(content)
            return np.mean([c.complexity for c in complexity]) if complexity else 0
        except Exception as e:
            logger.error(f"Error analyzing complexity: {str(e)}")
            return 0

    def _analyze_maintainability(self, content: str) -> float:
        """
        Analyze the maintainability index of the code.

        Args:
            content (str): The content of the file.

        Returns:
            float: The maintainability index.
        """
        try:
            return radon_metrics.mi_visit(content, True)
        except Exception as e:
            logger.error(f"Error analyzing maintainability: {str(e)}")
            return 0

    def _analyze_comment_ratio(self, content: str) -> float:
        """
        Analyze the comment ratio of the code.

        Args:
            content (str): The content of the file.

        Returns:
            float: The ratio of comments to total lines.
        """
        try:
            tree = ast.parse(content)
            comments = [node for node in ast.walk(tree) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)]
            return len(comments) / len(content.split('\n'))
        except Exception as e:
            logger.error(f"Error analyzing comment ratio: {str(e)}")
            return 0

    def _analyze_code_to_comment_ratio(self, content: str) -> float:
        """
        Analyze the ratio of code lines to comment lines.

        Args:
            content (str): The content of the file.

        Returns:
            float: The ratio of code lines to comment lines.
        """
        try:
            lines = content.split('\n')
            code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            return code_lines / comment_lines if comment_lines > 0 else float('inf')
        except Exception as e:
            logger.error(f"Error analyzing code to comment ratio: {str(e)}")
            return 0

    def _analyze_function_length(self, content: str) -> List[int]:
        """
        Analyze the length of functions in the code.

        Args:
            content (str): The content of the file.

        Returns:
            List[int]: A list of function lengths.
        """
        try:
            tree = ast.parse(content)
            return [node.end_lineno - node.lineno + 1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        except Exception as e:
            logger.error(f"Error analyzing function length: {str(e)}")
            return []

    def _analyze_class_length(self, content: str) -> List[int]:
        """
        Analyze the length of classes in the code.

        Args:
            content (str): The content of the file.

        Returns:
            List[int]: A list of class lengths.
        """
        try:
            tree = ast.parse(content)
            return [node.end_lineno - node.lineno + 1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        except Exception as e:
            logger.error(f"Error analyzing class length: {str(e)}")
            return []

    def _run_tests_and_coverage(self) -> float:
        """
        Run tests and calculate code coverage.

        Returns:
            float: The code coverage percentage.
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                repo_url = f"https://github.com/{self.repo_name}.git"
                subprocess.run(['git', 'clone', repo_url, tmpdir], check=True)
                
                # Run tests with coverage
                result = subprocess.run(['coverage', 'run', '-m', 'pytest'], 
                                        cwd=tmpdir, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"Tests failed with output: {result.stderr}")
                
                # Get coverage report
                coverage_result = subprocess.run(['coverage', 'report', '-m'], 
                                                 cwd=tmpdir, capture_output=True, text=True)
                
                # Parse coverage percentage
                coverage_lines = coverage_result.stdout.split('\n')
                total_line = coverage_lines[-2]  # The second to last line contains the total
                coverage_percentage = float(total_line.split()[-1].rstrip('%'))
                
                return coverage_percentage
        except Exception as e:
            logger.error(f"Error running tests and coverage: {str(e)}")
            return 0


class FragranceGenerator:
    """
    Generates a fragrance profile based on code quality metrics.
    """

    def __init__(self):
        """
        Initialize the FragranceGenerator with predefined scents.
        """
        self.scents = {
            'lavender': 'Calm, clean code',
            'peppermint': 'Refreshing, efficient code',
            'lemon': 'Clean, well-documented code',
            'eucalyptus': 'Complex but powerful code',
            'rosemary': 'Memory-intensive code',
            'tea tree': 'Bug-free code',
            'frankincense': 'Legacy code',
            'sandalwood': 'Stable, reliable code',
            'jasmine': 'Elegant, well-structured code',
            'bergamot': 'Balanced, harmonious code'
        }

    def generate_fragrance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate a fragrance profile based on code metrics.

        Args:
            metrics (Dict[str, float]): The code quality metrics.

        Returns:
            Dict[str, Any]: A dictionary containing the fragrance profile and description.
        """
        scores = []
        for scent in self.scents:
            score = self._calculate_scent_score(scent, metrics)
            scores.append((scent, score))
        
        top_scents = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        
        fragrance = {
            'profile': [{'scent': scent, 'intensity': score} for scent, score in top_scents],
            'description': self._generate_description(top_scents, metrics)
        }
        return fragrance

    def _calculate_scent_score(self, scent: str, metrics: Dict[str, float]) -> float:
        """
        Calculate the score for a specific scent based on metrics.

        Args:
            scent (str): The name of the scent.
            metrics (Dict[str, float]): The code quality metrics.

        Returns:
            float: The calculated score for the scent.
        """
        weights = {
            'lavender': {'maintainability': 0.5, 'complexity': -0.3, 'comment_ratio': 0.2},
            'peppermint': {'maintainability': 0.3, 'complexity': -0.2, 'test_coverage': 0.5},
            'lemon': {'comment_ratio': 0.7, 'maintainability': 0.3},
            'eucalyptus': {'complexity': 0.6, 'maintainability': -0.4},
            'rosemary': {'complexity': 0.5, 'maintainability': -0.5},
            'tea tree': {'test_coverage': 0.8, 'maintainability': 0.2},
            'frankincense': {'maintainability': -0.7, 'complexity': 0.3},
            'sandalwood': {'maintainability': 0.6, 'test_coverage': 0.4},
            'jasmine': {'maintainability': 0.7, 'comment_ratio': 0.3},
            'bergamot': {'maintainability': 0.4, 'complexity': -0.2, 'test_coverage': 0.4}
        }
        
        return sum(metrics[metric] * weight for metric, weight in weights[scent].items())

    def _generate_description(self, top_scents: List[Tuple[str, float]], metrics: Dict[str, float]) -> str:
        """
        Generate a description of the code based on the top scents and metrics.

        Args:
            top_scents (List[Tuple[str, float]]): The top scents and their scores.
            metrics (Dict[str, float]): The code quality metrics.

        Returns:
            str: A description of the code's fragrance profile.
        """
        description = f"Your code has notes of {', '.join([scent for scent, _ in top_scents])}. "
        
        if metrics['maintainability'] > 75:
            description += "It's highly maintainable and a pleasure to work with. "
        elif metrics['maintainability'] < 50:
            description += "It might benefit from some refactoring to improve maintainability. "
        
        if metrics['complexity'] > 30:
            description += "The code is quite complex, which might make it powerful but challenging to understand. "
        elif metrics['complexity'] < 10:
            description += "The code is simple and straightforward, which is great for readability. "
        
        if metrics['test_coverage'] > 80:
            description += "Excellent test coverage provides a safety net for future changes. "
        elif metrics['test_coverage'] < 50:
            description += "Increasing test coverage could help catch potential issues earlier. "
        
        return description.strip()


class Visualizer:
    """
    Generates visualizations of code quality metrics.
    """

    def generate_visualizations(self, metrics: Dict[str, float]) -> None:
        """
        Generate visualizations for code quality metrics.

        Args:
            metrics (Dict[str, float]): The code quality metrics.
        """
        self._generate_bar_chart(metrics)
        self._generate_radar_chart(metrics)
        self._generate_heatmap(metrics)

    def _generate_bar_chart(self, metrics: Dict[str, float]) -> None:
        """
        Generate a bar chart of code quality metrics.

        Args:
            metrics (Dict[str, float]): The code quality metrics.
        """
        plt.figure(figsize=(12, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Code Quality Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('code_quality_bar_chart.png')
        plt.close()

    def _generate_radar_chart(self, metrics: Dict[str, float]) -> None:
        """
        Generate a radar chart of code quality metrics.

        Args:
            metrics (Dict[str, float]): The code quality metrics.
        """
        categories = list(metrics.keys())
        values = list(metrics.values())

        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        values += values[:1]
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories)
        ax.plot(angles, values)
        ax.fill(angles, values, 'teal', alpha=0.1)
        plt.title('Code Quality Metrics Radar Chart')
        plt.savefig('code_quality_radar_chart.png')
        plt.close()

    def _generate_heatmap(self, metrics: Dict[str, float]) -> None:
        """
        Generate a heatmap of code quality metrics.

        Args:
            metrics (Dict[str, float]): The code quality metrics.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow([list(metrics.values())], cmap='YlOrRd', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45, ha='right')
        plt.yticks([])
        plt.title('Code Quality Metrics Heatmap')
        for i, v in enumerate(metrics.values()):
            plt.text(i, 0, f'{v:.2f}', ha='center', va='center')
        plt.tight_layout()
        plt.savefig('code_quality_heatmap.png')
        plt.close()


class CodeQualityAromatherapyAssistant:
    """
    Main class for the Code Quality Aromatherapy Assistant.
    """

    def __init__(self, repo_name: str):
        """
        Initialize the Code Quality Aromatherapy Assistant.

        Args:
            repo_name (str): The full name of the repository (owner/repo).
        """
        self.repo_name = repo_name
        self.analyzer = CodeAnalyzer(repo_name)
        self.fragrance_generator = FragranceGenerator()
        self.visualizer = Visualizer()

    def analyze_and_generate_fragrance(self) -> Dict[str, Any]:
        """
        Analyze the code and generate a fragrance profile.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results and fragrance profile.
        """
        metrics = self.analyzer.analyze_code()
        fragrance = self.fragrance_generator.generate_fragrance(metrics)
        self.visualizer.generate_visualizations(metrics)

        return {
            'metrics': metrics,
            'fragrance': fragrance
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a detailed report of the analysis results.

        Args:
            results (Dict[str, Any]): The analysis results and fragrance profile.

        Returns:
            str: A formatted report string.
        """
        metrics = results['metrics']
        fragrance = results['fragrance']

        report = f"Code Quality Aromatherapy Report for {self.repo_name}\n"
        report += "=" * 50 + "\n\n"

        report += "Metrics:\n"
        report += "-" * 20 + "\n"
        for metric, value in metrics.items():
            report += f"{metric.replace('_', ' ').title()}: {value:.2f}\n"
        report += "\n"

        report += "Fragrance Profile:\n"
        report += "-" * 20 + "\n"
        for scent in fragrance['profile']:
            report += f"{scent['scent'].title()}: {scent['intensity']:.2f}\n"
        report += "\n"

        report += "Fragrance Description:\n"
        report += "-" * 20 + "\n"
        report += f"{fragrance['description']}\n\n"

        report += "Recommendations:\n"
        report += "-" * 20 + "\n"
        report += self._generate_recommendations(metrics)

        return report

    def _generate_recommendations(self, metrics: Dict[str, float]) -> str:
        """
        Generate recommendations based on the code quality metrics.

        Args:
            metrics (Dict[str, float]): The code quality metrics.

        Returns:
            str: A string containing recommendations.
        """
        recommendations = []

        if metrics['maintainability'] < 65:
            recommendations.append("- Consider refactoring to improve code maintainability.")
        if metrics['complexity'] > 25:
            recommendations.append("- Look for opportunities to simplify complex functions or classes.")
        if metrics['comment_ratio'] < 0.1:
            recommendations.append("- Increase code documentation and comments to improve readability.")
        if metrics['test_coverage'] < 70:
            recommendations.append("- Improve test coverage to catch potential issues earlier.")
        if metrics['average_function_length'] > 50:
            recommendations.append("- Consider breaking down long functions into smaller, more manageable units.")
        if metrics['average_class_length'] > 200:
            recommendations.append("- Review large classes and consider splitting them into smaller, focused classes.")

        if not recommendations:
            recommendations.append("- Great job! Your code quality metrics are looking good. Keep up the good work!")

        return "\n".join(recommendations)


def main(repo_name: str) -> None:
    """
    Main function to run the Code Quality Aromatherapy Assistant.

    Args:
        repo_name (str): The full name of the repository (owner/repo).
    """
    try:
        assistant = CodeQualityAromatherapyAssistant(repo_name)
        results = assistant.analyze_and_generate_fragrance()
        report = assistant.generate_report(results)

        print(report)

        # Save report to file
        with open('code_quality_aromatherapy_report.txt', 'w') as f:
            f.write(report)

        print("\nReport saved to 'code_quality_aromatherapy_report.txt'")
        print("Visualizations saved as PNG files in the current directory.")

    except GithubException as e:
        logger.error(f"GitHub API error: {str(e)}")
        print(f"Error: Unable to access the repository. Please check the repository name and your GitHub token.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"An unexpected error occurred. Please check the logs for more information.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cqaa.py <repo_owner>/<repo_name>")
        sys.exit(1)

    repo_name = sys.argv[1]
    main(repo_name)


# Additional utility functions

def validate_github_token() -> bool:
    """
    Validate the GitHub token by making a test API call.

    Returns:
        bool: True if the token is valid, False otherwise.
    """
    try:
        g = Github(os.getenv('GITHUB_TOKEN'))
        g.get_user().login
        return True
    except GithubException:
        return False

def setup_environment() -> None:
    """
    Set up the environment for the script.
    """
    if 'GITHUB_TOKEN' not in os.environ:
        token = input("Please enter your GitHub personal access token: ")
        os.environ['GITHUB_TOKEN'] = token

    if not validate_github_token():
        print("Error: Invalid GitHub token. Please check your token and try again.")
        sys.exit(1)

def parse_arguments() -> str:
    """
    Parse command-line arguments.

    Returns:
        str: The repository name.
    """
    parser = argparse.ArgumentParser(description="Code Quality Aromatherapy Assistant")
    parser.add_argument("repo", help="The full name of the repository (owner/repo)")
    args = parser.parse_args()
    return args.repo

def check_dependencies() -> None:
    """
    Check if all required dependencies are installed.
    """
    required_packages = ['numpy', 'matplotlib', 'PyGithub', 'radon', 'pylint', 'coverage']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Error: The following required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("Please install them using 'pip install <package_name>' and try again.")
        sys.exit(1)

def display_welcome_message() -> None:
    """
    Display a welcome message and brief instructions.
    """
    print("Welcome to the Code Quality Aromatherapy Assistant!")
    print("=" * 50)
    print("This tool will analyze your GitHub repository and generate a")
    print("unique fragrance profile based on various code quality metrics.")
    print("\nPlease ensure you have set the GITHUB_TOKEN environment variable")
    print("with your GitHub personal access token before proceeding.")
    print("\nLet's begin the analysis...\n")

def cleanup_temporary_files() -> None:
    """
    Clean up any temporary files created during the analysis.
    """
    temp_files = [
        'code_quality_bar_chart.png',
        'code_quality_radar_chart.png',
        'code_quality_heatmap.png'
    ]

    for file in temp_files:
        try:
            os.remove(file)
        except OSError:
            pass

if __name__ == "__main__":
    setup_environment()
    check_dependencies()
    display_welcome_message()
    repo_name = parse_arguments()
    
    try:
        main(repo_name)
    finally:
        cleanup_temporary_files()

# End of script
