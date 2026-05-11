#!/usr/bin/env python3
"""
Parse summary.md and generate comprehensive thesis report
with three sections: Validation Agent Comparison, Per-Class Metrics, and Correction Agent
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def parse_summary_md(file_path):
    """Parse the summary.md file to extract all model metrics"""
    content = Path(file_path).read_text()
    
    # Pattern to extract each run's data
    runs = defaultdict(lambda: defaultdict(dict))
    
    # Split by the --- separator to get individual runs
    run_blocks = content.split('---\n\n')
    
    for block in run_blocks[1:]:  # Skip header
        lines = block.strip().split('\n')
        
        # Extract strategy, model, and set from header
        # Looking for patterns like "### claude-sonnet-4.6 - stefan2"
        header_match = None
        strategy = None
        model = None
        set_num = None
        
        for line in lines:
            if '### ' in line and 'stefan' in line:
                header_match = line
                # Extract model and set
                m = re.search(r'###\s+(.+?)\s+-\s+stefan(\d+)', line)
                if m:
                    model = m.group(1)
                    set_num = int(m.group(2))
                break
        
        if not header_match or not model or not set_num:
            continue
        
        # Determine strategy from context
        if 'zero-shot' in block.lower():
            strategy = 'zero-shot'
        elif 'rag' in block.lower() and 'cot' not in block.lower():
            strategy = 'rag'
        elif 'cot' in block.lower():
            strategy = 'cot'
        
        if not strategy:
            continue
        
        # Extract Validation Agent metrics
        val_metrics = {}
        corr_metrics = {}
        val_per_class = {}
        corr_field_metrics = {}
        
        # Parse Validation Agent (Classification) section
        val_start = block.find('## Validation Agent (Classification)')
        if val_start != -1:
            val_end = block.find('### Validation Per-Class Metrics')
            if val_end == -1:
                val_end = block.find('## Correction Agent')
            
            val_section = block[val_start:val_end]
            
            # Extract table from validation agent section
            tables = re.findall(r'\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|', val_section)
            for key, value in tables:
                key = key.strip()
                value = value.strip()
                if key in ['Accuracy', 'Precision', 'Recall', 'F1']:
                    try:
                        val_metrics[key] = float(value)
                    except:
                        pass
        
        # Parse Per-Class Metrics
        per_class_start = block.find('### Validation Per-Class Metrics')
        if per_class_start != -1:
            per_class_end = block.find('###', per_class_start + 1)
            if per_class_end == -1:
                per_class_end = block.find('##', per_class_start + 1)
            
            per_class_section = block[per_class_start:per_class_end]
            
            # Parse per-class table
            # Looking for rows like: | valid | 26 | 0.812 | 1.000 | 0.897 | 26 | 6 | 0 |
            per_class_rows = re.findall(
                r'\|\s*(valid|partially_valid|invalid)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|',
                per_class_section
            )
            
            for row in per_class_rows:
                class_name, precision, recall, f1, tp, fp, fn = row
                val_per_class[class_name] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'tp': int(tp),
                    'fp': int(fp),
                    'fn': int(fn)
                }
        
        # Parse Correction Agent
        corr_start = block.find('## Correction Agent')
        if corr_start != -1:
            corr_end = block.find('###', corr_start)
            if corr_end == -1:
                corr_end = len(block)
            
            corr_section = block[corr_start:corr_end]
            
            # Extract metrics from correction agent section
            tables = re.findall(r'\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|', corr_section)
            for key, value in tables:
                key = key.strip()
                value = value.strip()
                if key in ['Precision', 'Recall', 'F1']:
                    try:
                        corr_metrics[key] = float(value)
                    except:
                        pass
        
        # Store all metrics
        runs[strategy][f"{model}-set{set_num}"] = {
            'model': model,
            'set': set_num,
            'validation': val_metrics,
            'validation_per_class': val_per_class,
            'correction': corr_metrics
        }
    
    return runs


def format_table(headers, rows):
    """Format a markdown table"""
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(lines)


def generate_section_a(runs):
    """Generate Section A: Validation Agent Comparison (Stefan2+3 combined)"""
    report = []
    report.append('# Section A: Validation Agent Model & Strategy Comparison')
    report.append('')
    report.append('This section compares the performance of different models across three prompt strategies')
    report.append('(zero-shot, RAG, and chain-of-thought) using datasets Stefan2 and Stefan3 combined.')
    report.append('')
    
    for strategy in ['zero-shot', 'rag', 'cot']:
        if strategy not in runs:
            continue
        
        report.append(f'## {strategy.upper()} Strategy')
        report.append('')
        
        # Collect data for Stefan 2 and 3
        models_data = {}
        for run_key, run_data in runs[strategy].items():
            if run_data['set'] in [2, 3]:
                model = run_data['model']
                if model not in models_data:
                    models_data[model] = []
                models_data[model].append(run_data)
        
        # Build table rows (average of set 2 and 3)
        table_rows = []
        for model in sorted(models_data.keys()):
            runs_list = models_data[model]
            
            # Calculate average metrics
            avg_precision = sum(r['validation'].get('Precision', 0) for r in runs_list) / len(runs_list)
            avg_recall = sum(r['validation'].get('Recall', 0) for r in runs_list) / len(runs_list)
            avg_f1 = sum(r['validation'].get('F1', 0) for r in runs_list) / len(runs_list)
            avg_accuracy = sum(r['validation'].get('Accuracy', 0) for r in runs_list) / len(runs_list)
            
            table_rows.append([
                model,
                f"{avg_precision:.3f}",
                f"{avg_recall:.3f}",
                f"{avg_f1:.3f}",
                f"{avg_accuracy:.3f}"
            ])
        
        # Sort by accuracy (descending)
        table_rows.sort(key=lambda x: float(x[4]), reverse=True)
        
        headers = ['Model', 'Precision', 'Recall', 'F1', 'Accuracy']
        report.append(format_table(headers, table_rows))
        report.append('')
        
        # Add commentary
        if table_rows:
            best_model = table_rows[0][0]
            best_accuracy = float(table_rows[0][4])
            report.append(f'**Commentary:** {best_model} achieved the highest accuracy ({best_accuracy:.1%}) using the {strategy} strategy on combined Stefan2+3 datasets.')
            report.append('')
    
    return '\n'.join(report)


def generate_section_b(runs):
    """Generate Section B: Per-Class Metrics (RAG strategy, all datasets)"""
    report = []
    report.append('# Section B: Validation Per-Class Metrics (RAG Strategy)')
    report.append('')
    report.append('This section focuses exclusively on the RAG (Retrieval-Augmented Generation) strategy,')
    report.append('analyzing per-class classification performance across all available datasets (Stefan1-4).')
    report.append('Models are ranked by F1 score for each class.')
    report.append('')
    
    if 'rag' not in runs:
        report.append('No RAG strategy data available.')
        return '\n'.join(report)
    
    classes = ['valid', 'partially_valid', 'invalid']
    
    for class_name in classes:
        report.append(f'## {class_name.upper()} Class Metrics')
        report.append('')
        
        # Collect per-class data
        class_rows = []
        for run_key, run_data in runs['rag'].items():
            model = run_data['model']
            set_num = run_data['set']
            
            if class_name in run_data['validation_per_class']:
                metrics = run_data['validation_per_class'][class_name]
                class_rows.append([
                    model,
                    str(set_num),
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1']:.3f}",
                    str(metrics['tp']),
                    str(metrics['fp']),
                    str(metrics['fn']),
                    metrics['f1']  # For sorting
                ])
        
        # Sort by F1 score (descending)
        class_rows.sort(key=lambda x: float(x[8]), reverse=True)
        
        # Remove the sorting key
        class_rows = [row[:-1] for row in class_rows]
        
        headers = ['Model', 'Set', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN']
        report.append(format_table(headers, class_rows))
        report.append('')
    
    # Add average table across datasets per model
    report.append('## Average Metrics Across Datasets (RAG Strategy)')
    report.append('')
    
    avg_rows = []
    models_all = {}
    
    for run_key, run_data in runs['rag'].items():
        model = run_data['model']
        if model not in models_all:
            models_all[model] = {'valid': [], 'partially_valid': [], 'invalid': []}
        
        for class_name in classes:
            if class_name in run_data['validation_per_class']:
                metrics = run_data['validation_per_class'][class_name]
                models_all[model][class_name].append(metrics)
    
    for model in sorted(models_all.keys()):
        avg_precision = sum(sum(m['precision'] for m in metrics) / len(metrics) 
                           for metrics in models_all[model].values()) / 3
        avg_recall = sum(sum(m['recall'] for m in metrics) / len(metrics) 
                        for metrics in models_all[model].values()) / 3
        avg_f1 = sum(sum(m['f1'] for m in metrics) / len(metrics) 
                    for metrics in models_all[model].values()) / 3
        
        avg_rows.append([
            model,
            f"{avg_precision:.3f}",
            f"{avg_recall:.3f}",
            f"{avg_f1:.3f}",
            avg_f1  # For sorting
        ])
    
    avg_rows.sort(key=lambda x: float(x[4]), reverse=True)
    avg_rows = [row[:-1] for row in avg_rows]
    
    headers = ['Model', 'Avg Precision', 'Avg Recall', 'Avg F1']
    report.append(format_table(headers, avg_rows))
    report.append('')
    
    # Add overall confusion matrix
    report.append('## Overall Confusion Matrix (RAG Strategy, All Datasets Combined)')
    report.append('')
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for run_key, run_data in runs['rag'].items():
        for class_name in classes:
            if class_name in run_data['validation_per_class']:
                metrics = run_data['validation_per_class'][class_name]
                total_tp += metrics['tp']
                total_fp += metrics['fp']
                total_fn += metrics['fn']
    
    cm_rows = [
        ['True Positives (TP)', str(total_tp)],
        ['False Positives (FP)', str(total_fp)],
        ['False Negatives (FN)', str(total_fn)]
    ]
    
    headers = ['Metric', 'Value']
    report.append(format_table(headers, cm_rows))
    report.append('')
    
    return '\n'.join(report)


def generate_section_c(runs):
    """Generate Section C: Correction Agent Analysis"""
    report = []
    report.append('# Section C: Correction Agent Performance Analysis')
    report.append('')
    report.append('This section analyzes the correction agent performance across different strategies,')
    report.append('evaluating the effectiveness of field-level corrections on partially-valid entries.')
    report.append('')
    
    # Collect correction metrics by strategy and model
    strategy_data = {}
    for strategy in runs:
        strategy_data[strategy] = {}
        for run_key, run_data in runs[strategy].items():
            model = run_data['model']
            if model not in strategy_data[strategy]:
                strategy_data[strategy][model] = []
            strategy_data[strategy][model].append(run_data)
    
    # Create comparison table by strategy
    report.append('## Correction Agent Performance by Strategy')
    report.append('')
    
    for strategy in sorted(strategy_data.keys()):
        report.append(f'### {strategy.upper()} Strategy')
        report.append('')
        
        rows = []
        for model in sorted(strategy_data[strategy].keys()):
            runs_list = strategy_data[strategy][model]
            
            # Average correction metrics
            avg_precision = sum(r['correction'].get('Precision', 0) for r in runs_list) / len(runs_list)
            avg_recall = sum(r['correction'].get('Recall', 0) for r in runs_list) / len(runs_list)
            avg_f1 = sum(r['correction'].get('F1', 0) for r in runs_list) / len(runs_list)
            
            rows.append([
                model,
                f"{avg_precision:.3f}",
                f"{avg_recall:.3f}",
                f"{avg_f1:.3f}",
                avg_f1  # For sorting
            ])
        
        # Sort by F1 (descending)
        rows.sort(key=lambda x: float(x[4]), reverse=True)
        rows = [row[:-1] for row in rows]
        
        headers = ['Model', 'Precision', 'Recall', 'F1']
        report.append(format_table(headers, rows))
        report.append('')
    
    # Add insights about correction challenges
    report.append('## Key Observations')
    report.append('')
    report.append('- **Low Recall Challenge:** Correction agents across all strategies show consistently low recall,')
    report.append('  indicating difficulty in identifying all partially-valid entries that require correction.')
    report.append('- **High Precision:** When corrections are attempted, most are accurate (high precision),')
    report.append('  suggesting careful prediction when corrections are made.')
    report.append('- **Strategy Effectiveness:** RAG strategy generally outperforms other strategies in correction tasks,')
    report.append('  likely due to access to reference data for validation and correction decisions.')
    report.append('')
    
    return '\n'.join(report)


def main():
    summary_path = Path('/Users/passum/Documents/SWEDEN/DSI/THESIS/AI_reference_agent_detection_correction/results/summary.md')
    
    # Parse data
    print("Parsing summary.md...")
    runs = parse_summary_md(summary_path)
    
    print(f"Found {sum(len(v) for v in runs.values())} runs across {len(runs)} strategies")
    
    # Generate sections
    print("Generating Section A...")
    section_a = generate_section_a(runs)
    
    print("Generating Section B...")
    section_b = generate_section_b(runs)
    
    print("Generating Section C...")
    section_c = generate_section_c(runs)
    
    # Combine sections
    full_report = '\n\n'.join([section_a, section_b, section_c])
    
    # Write to thesis_results.md
    thesis_path = Path('/Users/passum/Documents/SWEDEN/DSI/THESIS/AI_reference_agent_detection_correction/results/thesis_result.md')
    thesis_path.write_text(full_report + '\n')
    
    print(f"\nThesis report written to {thesis_path}")


if __name__ == '__main__':
    main()
