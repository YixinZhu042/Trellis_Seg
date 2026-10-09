"""Select the continuation checkpoint using only the fixed full validation split."""
import argparse
import json
from pathlib import Path


NAMES = ['pretrained', 'regular_500', 'regular_1000', 'regular_1095',
         'ema_500', 'ema_1000', 'ema_1095']


def score(summary):
    # The checkpoint decision is fixed before reading test results.
    return (summary['projection']['miou'], summary['surface']['miou'],
            summary['boundary_f1'] if summary['boundary_f1'] is not None else -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation_dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    results = []
    for name in NAMES:
        path = args.validation_dir / name / 'metrics.json'
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        if data['split'] != 'validation' or data['seeds'] != [42] or len(data['samples']) != 512:
            raise ValueError(f'Invalid full validation result: {path}')
        results.append({'name': name, 'checkpoint': data['checkpoint'],
                        'summary': data['summary'], 'oracle_summary': data['oracle_summary']})
    continuation = [row for row in results if row['name'] != 'pretrained']
    selected = max(continuation, key=lambda row: score(row['summary']))
    best_overall = max(results, key=lambda row: score(row['summary']))
    report = {
        'selection_split': 'validation', 'selection_seed': 42,
        'selection_rule': 'projection mIoU, then surface mIoU, then boundary F1',
        'results': results, 'selected_continuation': selected,
        'best_overall_on_validation': best_overall,
        'test_results_used_for_selection': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'selected_continuation': selected['name'],
                      'best_overall_on_validation': best_overall['name'],
                      'selected_summary': selected['summary']}, indent=2))


if __name__ == '__main__':
    main()
