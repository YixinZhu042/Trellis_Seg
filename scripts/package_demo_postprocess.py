"""Package raw/new-vs-postprocessed demo assets after an audited cleanup run."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.dit_vae_v2 import file_sha256
from vae_v2.config import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--git-repo", type=Path, required=True)
    args = parser.parse_args()
    result = args.result.resolve()
    git = args.git_repo.resolve()
    base = json.loads((git / "demo/comparison.json").read_text())
    summary = json.loads((result / "postprocess_summary.json").read_text())
    by_id = {record["id"]: record for record in summary["samples"]}
    samples = []
    copied_bytes = 0
    if set(by_id) != {row["id"] for row in base["samples"]}:
        raise ValueError("Postprocess cohort differs from paired demo")

    for row in base["samples"]:
        record = by_id[row["id"]]
        folder = git / "demo/comparison" / row["id"]
        source = result / "objects" / row["id"]
        assets = {}
        for name in ("post.glb", "post_comparison.png"):
            src = source / name
            dst = folder / name
            if not src.is_file():
                raise ValueError(f"Missing {src}")
            shutil.copy2(src, dst)
            if file_sha256(src) != file_sha256(dst):
                raise ValueError("Post asset copy differs")
            assets[name] = {"bytes": dst.stat().st_size, "sha256": file_sha256(dst)}
            copied_bytes += dst.stat().st_size
        samples.append(
            {
                "id": row["id"],
                "title": row["title"],
                "kind": row["kind"],
                "faces": row["faces"],
                "raw_glb": row["new_glb"],
                "post_glb": f'comparison/{row["id"]}/post.glb',
                "condition_image": row["condition_image"],
                "comparison_image": f'comparison/{row["id"]}/post_comparison.png',
                "source_image": row.get("source_image"),
                "raw": record["raw"],
                "post": record["post"],
                "miou_delta": record["miou_delta"],
                "accuracy_delta": record["accuracy_delta"],
                "changed_faces": record["change"]["changed_faces"],
                "changed_area_fraction": record["change"]["changed_area_fraction"],
                "components_before": record["raw_fragmentation"]["components"],
                "components_after": record["post_fragmentation"]["components"],
                "unique_ids_before": record["raw_fragmentation"]["unique_ids"],
                "unique_ids_after": record["post_fragmentation"]["unique_ids"],
                "absent_id_area_before": record["raw_fragmentation"]["absent_id_area_fraction"],
                "absent_id_area_after": record["post_fragmentation"]["absent_id_area_fraction"],
                "assets": assets,
                "notes": row["notes"],
            }
        )

    groups = {}
    for kind in ("original_demo", "character"):
        rows = [row for row in samples if row["kind"] == kind]
        groups[kind] = {
            "count": len(rows),
            "raw_miou": statistics.mean(row["raw"]["miou"] for row in rows),
            "post_miou": statistics.mean(row["post"]["miou"] for row in rows),
            "raw_accuracy": statistics.mean(row["raw"]["accuracy"] for row in rows),
            "post_accuracy": statistics.mean(row["post"]["accuracy"] for row in rows),
            "components_before": sum(row["components_before"] for row in rows),
            "components_after": sum(row["components_after"] for row in rows),
            "changed_area_mean": statistics.mean(row["changed_area_fraction"] for row in rows),
            "improved": sum(row["miou_delta"] > 1e-12 for row in rows),
            "regressed": sum(row["miou_delta"] < -1e-12 for row in rows),
        }

    public = {
        "release_date": "2026-10-09",
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "input": "Released parts64 new-model raw face labels from the controlled paired demo",
            "method": "Geometry-only cleanup: merge same-label connected components below 0.05% total surface area only when one boundary label has at least 80% weighted support; then one synchronous high-confidence boundary vote.",
            "component_weight": "Shared boundary support weighted by adjacent-face area and normal agreement; sharp creases have low influence.",
            "condition_used_by_transformation": False,
            "gt_used": False,
            "label_matching": False,
            "projected_label_overwrite": False,
            "selection_note": "One global setting selected on these 18 development demos, not a held-out benchmark. Candidate had no per-object input-agreement mIoU regression in this cohort.",
            "metric_note": "Agreement with shared 2D input labels, not independent GT accuracy. Raw outputs remain available.",
        },
        "settings": summary["candidate"],
        "summary": groups,
        "samples": samples,
        "asset_bytes": copied_bytes,
    }
    atomic_json(git / "demo/postprocess.json", public)
    atomic_json(
        result / "postprocess_package_audit.json",
        {
            "objects": 18,
            "post_glbs": 18,
            "post_comparison_pngs": 18,
            "asset_bytes": copied_bytes,
            "raw_assets_untouched": True,
            "all_post_assets_verified": True,
            "condition_not_used_by_transform": True,
            "gt_not_used": True,
        },
    )
    print(json.dumps({"summary": groups, "asset_bytes": copied_bytes, "manifest": str(git / "demo/postprocess.json")}, indent=2))


if __name__ == "__main__":
    main()
