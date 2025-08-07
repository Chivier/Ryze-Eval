import json
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

class ResultsVisualizer:
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = results_dir
        self.report = None
        self.detailed_results = None
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def load_latest_results(self):
        json_files = [f for f in os.listdir(self.results_dir) if f.startswith('evaluation_report_') and f.endswith('.json')]
        
        if not json_files:
            print("No evaluation reports found in results directory")
            return False
        
        latest_report = sorted(json_files)[-1]
        report_path = os.path.join(self.results_dir, latest_report)
        
        with open(report_path, 'r') as f:
            self.report = json.load(f)
        
        detailed_file = latest_report.replace('evaluation_report_', 'detailed_results_')
        detailed_path = os.path.join(self.results_dir, detailed_file)
        
        if os.path.exists(detailed_path):
            with open(detailed_path, 'r') as f:
                self.detailed_results = json.load(f)
        
        print(f"✓ Loaded results from: {latest_report}")
        return True
    
    def plot_subset_performance(self, save_path: str = None):
        if not self.report:
            print("No report loaded. Please load results first.")
            return
        
        subset_metrics = self.report['subset_metrics']
        
        subsets = list(subset_metrics.keys())
        accuracies = [subset_metrics[s]['accuracy'] * 100 for s in subsets]
        coverages = [subset_metrics[s]['coverage'] * 100 for s in subsets]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = plt.cm.viridis(range(len(subsets)))
        bars1 = ax1.bar(range(len(subsets)), accuracies, color=colors)
        ax1.set_xlabel('Subset', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Accuracy by Subset', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(subsets)))
        ax1.set_xticklabels(subsets, rotation=45, ha='right')
        ax1.set_ylim(0, 100)
        
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        
        bars2 = ax2.bar(range(len(subsets)), coverages, color=colors)
        ax2.set_xlabel('Subset', fontsize=12)
        ax2.set_ylabel('Coverage (%)', fontsize=12)
        ax2.set_title('Coverage by Subset', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(subsets)))
        ax2.set_xticklabels(subsets, rotation=45, ha='right')
        ax2.set_ylim(0, 105)
        
        for i, (bar, cov) in enumerate(zip(bars2, coverages)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{cov:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Lab-Bench Evaluation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Subset performance plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_subtask_heatmap(self, save_path: str = None):
        if not self.report:
            print("No report loaded. Please load results first.")
            return
        
        subset_metrics = self.report['subset_metrics']
        
        all_subtasks = set()
        for subset_data in subset_metrics.values():
            all_subtasks.update(subset_data.get('subtask_performance', {}).keys())
        
        all_subtasks = sorted(list(all_subtasks))
        subsets = sorted(list(subset_metrics.keys()))
        
        heatmap_data = []
        for subset in subsets:
            row = []
            subtask_perf = subset_metrics[subset].get('subtask_performance', {})
            for subtask in all_subtasks:
                if subtask in subtask_perf:
                    row.append(subtask_perf[subtask] * 100)
                else:
                    row.append(None)
            heatmap_data.append(row)
        
        df = pd.DataFrame(heatmap_data, index=subsets, columns=all_subtasks)
        
        plt.figure(figsize=(16, 8))
        sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', 
                   vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'})
        plt.title('Subtask Performance Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Subtask', fontsize=12)
        plt.ylabel('Subset', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Subtask heatmap saved to: {save_path}")
        else:
            plt.show()
    
    def plot_response_time_distribution(self, save_path: str = None):
        if not self.detailed_results:
            print("No detailed results loaded.")
            return
        
        response_times = [r['response_time'] for r in self.detailed_results]
        subsets = [r['subset'] for r in self.detailed_results]
        
        df = pd.DataFrame({'Response Time (s)': response_times, 'Subset': subsets})
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(response_times, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Response Time (s)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Response Time Distribution', fontsize=14, fontweight='bold')
        plt.axvline(x=sum(response_times)/len(response_times), 
                   color='red', linestyle='--', label=f'Mean: {sum(response_times)/len(response_times):.2f}s')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        df.boxplot(by='Subset', column='Response Time (s)', ax=plt.gca())
        plt.xlabel('Subset', fontsize=12)
        plt.ylabel('Response Time (s)', fontsize=12)
        plt.title('Response Time by Subset', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.suptitle('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Response time plot saved to: {save_path}")
        else:
            plt.show()
    
    def generate_summary_report(self, save_path: str = None):
        if not self.report:
            print("No report loaded. Please load results first.")
            return
        
        overall = self.report['overall_metrics']
        model_info = self.report['model_info']
        
        summary = []
        summary.append("="*80)
        summary.append("LAB-BENCH EVALUATION SUMMARY REPORT")
        summary.append("="*80)
        summary.append(f"\nModel: {model_info.get('provider', 'Unknown')}")
        summary.append(f"Evaluation Date: {model_info.get('timestamp', 'Unknown')}")
        summary.append("\n" + "-"*40)
        summary.append("OVERALL PERFORMANCE")
        summary.append("-"*40)
        summary.append(f"Total Questions: {overall['total_questions']}")
        summary.append(f"Correct Answers: {overall['correct_answers']}")
        summary.append(f"Accuracy: {overall['accuracy']:.2%}")
        summary.append(f"Precision: {overall['precision']:.2%}")
        summary.append(f"Coverage: {overall['coverage']:.2%}")
        summary.append(f"Avg Response Time: {overall['avg_response_time']:.2f}s")
        
        summary.append("\n" + "-"*40)
        summary.append("SUBSET PERFORMANCE RANKING")
        summary.append("-"*40)
        
        subset_metrics = self.report['subset_metrics']
        ranked_subsets = sorted(subset_metrics.items(), 
                               key=lambda x: x[1]['accuracy'], 
                               reverse=True)
        
        for rank, (subset_name, metrics) in enumerate(ranked_subsets, 1):
            summary.append(f"{rank}. {subset_name}: {metrics['accuracy']:.2%} "
                         f"({metrics['correct_answers']}/{metrics['total_questions']})")
        
        summary.append("\n" + "-"*40)
        summary.append("RECOMMENDATIONS")
        summary.append("-"*40)
        
        weakest_subset = ranked_subsets[-1]
        summary.append(f"• Weakest performance: {weakest_subset[0]} ({weakest_subset[1]['accuracy']:.2%})")
        summary.append(f"• Consider fine-tuning or prompt engineering for this area")
        
        if overall['coverage'] < 0.95:
            summary.append(f"• Coverage is {overall['coverage']:.2%} - some questions were not attempted")
            summary.append(f"• Check for parsing errors or timeout issues")
        
        summary.append("\n" + "="*80)
        
        summary_text = "\n".join(summary)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(summary_text)
            print(f"✓ Summary report saved to: {save_path}")
        
        print(summary_text)
        return summary_text
    
    def create_all_visualizations(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, 'visualizations')
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("\n📊 Generating visualizations...")
        
        self.plot_subset_performance(
            save_path=os.path.join(output_dir, f'subset_performance_{timestamp}.png')
        )
        
        self.plot_subtask_heatmap(
            save_path=os.path.join(output_dir, f'subtask_heatmap_{timestamp}.png')
        )
        
        self.plot_response_time_distribution(
            save_path=os.path.join(output_dir, f'response_times_{timestamp}.png')
        )
        
        self.generate_summary_report(
            save_path=os.path.join(output_dir, f'summary_report_{timestamp}.txt')
        )
        
        print(f"\n✓ All visualizations saved to: {output_dir}/")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Lab-Bench evaluation results')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing evaluation results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.results_dir)
    
    if visualizer.load_latest_results():
        visualizer.create_all_visualizations(args.output_dir)
    else:
        print("Failed to load results. Please run evaluation first.")

if __name__ == "__main__":
    main()