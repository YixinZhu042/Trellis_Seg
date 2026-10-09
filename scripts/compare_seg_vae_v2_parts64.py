"""Compare two frozen-checkpoint reconstruction runs on exactly the same cohort."""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vae_v2.config import atomic_json


def paired_records(old, new):
    if old['manifest_digest'] != new['manifest_digest']:
        raise ValueError('Data manifest changed')
    if (len(old['ids']) != len(set(old['ids'])) or len(new['ids']) != len(set(new['ids']))
            or set(old['ids']) != set(new['ids'])):
        raise ValueError('Test cohort changed or duplicated')
    before = old['samples'] + old['diagnostics']
    after = new['samples'] + new['diagnostics']
    a, b = ({r['instance']: r for r in rows} for rows in (before, after))
    if len(a) != len(before) or len(b) != len(after) or a.keys() != b.keys():
        raise ValueError('Missing or duplicated object metrics')
    tests = set(old['ids'])
    for report in (old, new):
        if {r['instance'] for r in report['samples']} != tests:
            raise ValueError('Summary IDs differ from sample IDs')
        if tests & {r['instance'] for r in report['diagnostics']}:
            raise ValueError('Diagnostic objects leaked into test summary')
    pairs = []
    for uid in sorted(a):
        x,y=a[uid],b[uid]
        if (x['role'],x['split'],x['type_id']) != (y['role'],y['split'],y['type_id']):
            raise ValueError(f'Object identity changed: {uid}')
        if [(p['label'],p['area_fraction']) for p in x['parts']] != [(p['label'],p['area_fraction']) for p in y['parts']]:
            raise ValueError(f'Part labels or geometry changed: {uid}')
        pairs.append({'instance':uid,'role':x['role'],'split':x['split'],'parts':len(x['parts']),
                      'before':x,'after':y,
                      'surface_miou_delta':y['surface']['miou']-x['surface']['miou'],
                      'surface_accuracy_delta':y['surface']['accuracy']-x['surface']['accuracy']})
    return pairs


def build_report(old, new):
    pairs = paired_records(old,new)
    tests = [p for p in pairs if p['role']=='test']
    groups = []
    for name,lo,hi in [('1-20',1,20),('21-32',21,32),('33-63',33,63)]:
        selected = [r for r in tests if lo<=r['parts']<=hi]
        groups.append({'parts':name,'count':len(selected),
                       **{side:statistics.mean(r[side]['surface']['miou'] for r in selected) if selected else None
                          for side in ('before','after')}})
    scalar = lambda s: {f'{g}.{k}':s[g][k] for g in ('voxel','surface','projection','hidden_surface')
                        for k in ('accuracy','miou','small_part_recall')}
    a,b=scalar(old['summary']),scalar(new['summary'])
    for k in ('boundary_f1','isolated_error_area_fraction','false_internal_boundary_rate','surface_miou_p10'):
        a[k],b[k]=old['summary'][k],new['summary'][k]
    metrics={k:{'before':a[k],'after':b[k],'delta':None if a[k] is None or b[k] is None else b[k]-a[k]} for k in a}
    return {'created_at':datetime.now().astimezone().isoformat(),
            'before_checkpoint':old['checkpoint'],'after_checkpoint':new['checkpoint'],
            'manifest_digest':old['manifest_digest'],'test_count':len(tests),
            'note':'Development regression evaluation, not a fresh blind test. No checkpoint selection from test. Additional steps and permutation both changed; no unaugmented continuation control.',
            'metrics':metrics,'groups_by_parts':groups,
            'improved_over_1pp':sum(r['surface_miou_delta']>.01 for r in tests),
            'regressed_over_1pp':sum(r['surface_miou_delta']<-.01 for r in tests),
            'regressed_over_5pp':sum(r['surface_miou_delta']<-.05 for r in tests),
            'gains':sorted(tests,key=lambda r:-r['surface_miou_delta'])[:10],
            'regressions':sorted(tests,key=lambda r:r['surface_miou_delta'])[:10],
            'pairs':pairs}


def main():
    from PIL import Image,ImageDraw,ImageFont
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',type=Path,required=True)
    ap.add_argument('--after',type=Path,required=True)
    args=ap.parse_args()
    old=json.loads((args.before/'metrics.json').read_text())
    new=json.loads((args.after/'metrics.json').read_text())
    # Do not rely on merely matching IDs when rendering old/new panels.
    pa=json.loads((args.before/'plan.json').read_text());pb=json.loads((args.after/'plan.json').read_text())
    if pa['samples'] != pb['samples']:
        raise ValueError('Cache identities/cohort changed')
    report=build_report(old,new)
    atomic_json(args.after/'paired_comparison.json',report)
    target=args.after/'before_after';target.mkdir(exist_ok=True)
    font_path=Path('/usr/share/fonts/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    for r in report['pairs']:
        uid=r['instance']
        with Image.open(args.before/'objects'/uid/f'{uid}_comparison.png') as im:before=im.convert('RGB')
        with Image.open(args.after/'objects'/uid/f'{uid}_comparison.png') as im:after=im.convert('RGB')
        if before.size!=(1536,560) or after.size!=before.size:
            raise ValueError('Unexpected six-view panel layout')
        if before.crop((0,0,1536,280)).tobytes() != after.crop((0,0,1536,280)).tobytes():
            raise ValueError(f'GT/camera/palette changed: {uid}')
        canvas=Image.new('RGB',(1536,892),'white');draw=ImageDraw.Draw(canvas)
        draw.text((10,8),f'{uid} | {r["split"]}/{r["role"]} | {r["parts"]} parts | surface mIoU {r["before"]["surface"]["miou"]:.4f} -> {r["after"]["surface"]["miou"]:.4f}',font=font,fill='black')
        panels=[before.crop((0,0,1536,280)),before.crop((0,280,1536,560)),after.crop((0,280,1536,560))]
        titles=['GT',f'Before step {pa["selection"]["step"]}',f'Permutation step {pb["selection"]["step"]}']
        views=['front','right','back','left','top','bottom']
        for index,(im,title) in enumerate(zip(panels,titles)):
            y=52+index*280;canvas.paste(im,(0,y))
            draw.rectangle((0,y,1535,y+23),fill='white')
            for i,view in enumerate(views):draw.text((i*256+3,y+2),title+': '+view,fill='black')
        canvas.save(target/f'{uid}.png')
    pairs={p['instance']:p for p in report['pairs']}
    def card(uid):
        r=pairs[uid]
        return (f'<section><h3>{uid} ({r["parts"]} parts, {r["split"]})</h3>'
                f'<p>Surface mIoU: {r["before"]["surface"]["miou"]:.4f} → {r["after"]["surface"]["miou"]:.4f} '
                f'({r["surface_miou_delta"]*100:+.2f} pp) · '
                f'<a href="objects/{uid}/{uid}.glb">new GLB</a> · <a href="objects/{uid}/{uid}_gt.glb">GT GLB</a></p>'
                f'<a href="before_after/{uid}.png"><img loading="lazy" src="before_after/{uid}.png"></a></section>')
    focus=['251c8125765149c0bd46b3a1a51ff4b6','d1d58f05c346470e9805612d562e9dde',
           '007d2949e5b0450c995c54809b43af0f','11c88bc30d37479281c30cc49d25084a','141329918e4945ed82d0f48ba5903b8f']
    body=('<!doctype html><meta charset="utf-8"><title>VAE before and after permutation continuation</title>'
          '<style>body{max-width:1536px;margin:auto;padding:24px;font:16px sans-serif}img{width:100%}section{margin:32px 0}pre{background:#eee;padding:12px}</style>'
          '<h1>VAE reconstruction: before / after permutation continuation</h1>'
          f'<p>{html.escape(report["note"])}</p><p>Rows: GT / before / after. Fixed original labels; no postprocessing.</p>'
          '<p><a href="index.html">New model gallery and downloads</a></p>'
          f'<pre>{html.escape(json.dumps({k:report[k] for k in ("metrics","groups_by_parts","improved_over_1pp","regressed_over_1pp","regressed_over_5pp")},indent=2))}</pre>'
          '<h2>Previously identified cases</h2>'+''.join(card(u) for u in focus if u in pairs)+
          '<h2>Largest regressions (all retained)</h2>'+''.join(card(r['instance']) for r in report['regressions'])+
          '<h2>All test objects in fixed ID order</h2>'+''.join(card(u) for u in sorted(old['ids'])))
    (args.after/'comparison.html').write_text(body)
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','gains','regressions')},indent=2))
    print('Saved',len(report['pairs']),'verified GT/before/after six-view comparisons')


if __name__=='__main__':main()
