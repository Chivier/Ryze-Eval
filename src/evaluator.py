import json
import re
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from PIL import Image
import base64
from io import BytesIO

@dataclass
class EvaluationResult:
    question_id: str
    subset: str
    subtask: str
    predicted_answer: str
    correct_answer: str
    is_correct: bool
    response_time: float
    raw_response: str
    choices_presented: List[str]

@dataclass
class SubsetMetrics:
    subset_name: str
    total_questions: int
    correct_answers: int
    attempted_questions: int
    accuracy: float
    precision: float
    coverage: float
    avg_response_time: float
    subtask_performance: Dict[str, float]

class LabBenchEvaluator:
    def __init__(self, model_interface, verbose=True):
        self.model = model_interface
        self.verbose = verbose
        self.results = []
        self.subset_results = defaultdict(list)
        self.subtask_results = defaultdict(list)
    
    def format_multiple_choice_question(self, question: str, ideal: str, distractors: List[str]) -> Tuple[str, Dict[str, str], str]:
        choices = [ideal] + distractors
        import random
        random.shuffle(choices)
        
        choice_map = {}
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:len(choices)]
        
        formatted_question = question + "\n\nChoices:\n"
        correct_label = None
        
        for label, choice in zip(choice_labels, choices):
            choice_map[label] = choice
            formatted_question += f"{label}. {choice}\n"
            if choice == ideal:
                correct_label = label
        
        formatted_question += "\nPlease respond with only the letter of your answer (A, B, C, etc.)."
        
        return formatted_question, choice_map, correct_label
    
    def extract_answer_from_response(self, response: str) -> Optional[str]:
        response = response.strip().upper()
        
        direct_match = re.match(r'^([A-H])(?:\.|,|\s|$)', response)
        if direct_match:
            return direct_match.group(1)
        
        pattern_match = re.search(r'\b([A-H])\b(?:\.|,|\s|$)', response)
        if pattern_match:
            return pattern_match.group(1)
        
        answer_pattern = re.search(r'(?:answer|choice|select|pick)(?:\s+is)?[::\s]+([A-H])\b', response, re.IGNORECASE)
        if answer_pattern:
            return answer_pattern.group(1)
        
        return None
    
    def prepare_image_for_model(self, image_data) -> Optional[str]:
        try:
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image_bytes = image_data['bytes']
            elif isinstance(image_data, bytes):
                image_bytes = image_data
            else:
                return None
            
            image = Image.open(BytesIO(image_bytes))
            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
        except Exception as e:
            if self.verbose:
                print(f"Error processing image: {e}")
            return None
    
    def evaluate_question(self, item: Dict[str, Any], subset_name: str) -> EvaluationResult:
        question = item.get('question', '')
        ideal = item.get('ideal', '')
        distractors = item.get('distractors', [])
        question_id = item.get('id', 'unknown')
        subtask = item.get('subtask', 'unknown')
        
        formatted_q, choice_map, correct_label = self.format_multiple_choice_question(
            question, ideal, distractors
        )
        
        if subset_name in ['FigQA', 'TableQA']:
            image_field = 'image' if subset_name == 'FigQA' else 'table_image'
            if image_field in item and item[image_field]:
                image_b64 = self.prepare_image_for_model(item[image_field])
                if image_b64:
                    formatted_q = f"[Image attached]\n\n{formatted_q}"
        
        start_time = time.time()
        
        try:
            response = self.model.generate(formatted_q, temperature=0.1)
            response_time = time.time() - start_time
            
            predicted_answer = self.extract_answer_from_response(response)
            
            if predicted_answer is None:
                predicted_answer = "INVALID"
                is_correct = False
            else:
                is_correct = (predicted_answer == correct_label)
            
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating question {question_id}: {e}")
            response = f"ERROR: {str(e)}"
            response_time = time.time() - start_time
            predicted_answer = "ERROR"
            is_correct = False
        
        return EvaluationResult(
            question_id=question_id,
            subset=subset_name,
            subtask=subtask,
            predicted_answer=predicted_answer,
            correct_answer=correct_label,
            is_correct=is_correct,
            response_time=response_time,
            raw_response=response,
            choices_presented=list(choice_map.values())
        )
    
    def evaluate_subset(self, dataset_subset, subset_name: str, limit: Optional[int] = None) -> List[EvaluationResult]:
        subset_results = []
        
        total = min(len(dataset_subset), limit) if limit else len(dataset_subset)
        
        if self.verbose:
            print(f"\nEvaluating {subset_name} ({total} questions)...")
        
        for idx, item in enumerate(tqdm(dataset_subset, total=total, desc=subset_name)):
            if limit and idx >= limit:
                break
            
            result = self.evaluate_question(item, subset_name)
            subset_results.append(result)
            
            self.results.append(result)
            self.subset_results[subset_name].append(result)
            self.subtask_results[result.subtask].append(result)
            
            if self.verbose and (idx + 1) % 10 == 0:
                current_acc = sum(1 for r in subset_results if r.is_correct) / len(subset_results)
                print(f"  Progress: {idx+1}/{total}, Current Accuracy: {current_acc:.2%}")
        
        return subset_results
    
    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        if not results:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'coverage': 0.0,
                'avg_response_time': 0.0
            }
        
        correct = sum(1 for r in results if r.is_correct)
        attempted = sum(1 for r in results if r.predicted_answer not in ['INVALID', 'ERROR'])
        total = len(results)
        
        accuracy = correct / total if total > 0 else 0.0
        precision = correct / attempted if attempted > 0 else 0.0
        coverage = attempted / total if total > 0 else 0.0
        avg_response_time = np.mean([r.response_time for r in results])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'coverage': coverage,
            'avg_response_time': avg_response_time,
            'total_questions': total,
            'correct_answers': correct,
            'attempted_questions': attempted
        }
    
    def generate_report(self) -> Dict[str, Any]:
        overall_metrics = self.calculate_metrics(self.results)
        
        subset_metrics = {}
        for subset_name, results in self.subset_results.items():
            metrics = self.calculate_metrics(results)
            
            subtask_performance = defaultdict(list)
            for result in results:
                subtask_performance[result.subtask].append(result.is_correct)
            
            subtask_acc = {
                task: np.mean(correct_list) 
                for task, correct_list in subtask_performance.items()
            }
            
            subset_metrics[subset_name] = SubsetMetrics(
                subset_name=subset_name,
                total_questions=metrics['total_questions'],
                correct_answers=metrics['correct_answers'],
                attempted_questions=metrics['attempted_questions'],
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                coverage=metrics['coverage'],
                avg_response_time=metrics['avg_response_time'],
                subtask_performance=subtask_acc
            )
        
        subtask_metrics = {}
        for subtask_name, results in self.subtask_results.items():
            metrics = self.calculate_metrics(results)
            subtask_metrics[subtask_name] = metrics
        
        report = {
            'overall_metrics': overall_metrics,
            'subset_metrics': {k: asdict(v) for k, v in subset_metrics.items()},
            'subtask_metrics': subtask_metrics,
            'model_info': {
                'provider': self.model.__class__.__name__,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return report
    
    def save_results(self, output_dir: str = "./results"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        detailed_results = [asdict(r) for r in self.results]
        with open(f"{output_dir}/detailed_results_{timestamp}.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        report = self.generate_report()
        with open(f"{output_dir}/evaluation_report_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        self.print_report(report)
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)
        
        overall = report['overall_metrics']
        print("\nOVERALL PERFORMANCE:")
        print(f"  Accuracy: {overall['accuracy']:.2%}")
        print(f"  Precision: {overall['precision']:.2%}")
        print(f"  Coverage: {overall['coverage']:.2%}")
        print(f"  Total Questions: {overall['total_questions']}")
        print(f"  Correct Answers: {overall['correct_answers']}")
        print(f"  Avg Response Time: {overall['avg_response_time']:.2f}s")
        
        print("\n" + "-"*40)
        print("PERFORMANCE BY SUBSET:")
        print("-"*40)
        
        for subset_name, metrics in report['subset_metrics'].items():
            print(f"\n{subset_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct_answers']}/{metrics['total_questions']})")
            print(f"  Coverage: {metrics['coverage']:.2%}")
            print(f"  Avg Time: {metrics['avg_response_time']:.2f}s")
            
            if metrics['subtask_performance']:
                print(f"  Subtasks:")
                for task, acc in sorted(metrics['subtask_performance'].items()):
                    print(f"    - {task}: {acc:.2%}")
        
        print("\n" + "="*80)