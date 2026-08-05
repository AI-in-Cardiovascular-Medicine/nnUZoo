"""
Rewrite `network_class_name` in existing plans.json files.

nnUZoo used to build its networks from the external dynamic-network-architectures package; that code now
lives in nnunetv2/nets/. plans.json stores the architecture as an import-path string, so files written
before the switch name modules that no longer exist.

nnunetv2.utilities.plans_handling.plans_handler already translates those names on read, so nothing is
broken in the meantime. This command exists so you can update the files themselves and eventually delete
the compatibility table.

Typical use:

    nnUNetv2_migrate_plans $nnUNet_preprocessed --dry_run     # see what would change
    nnUNetv2_migrate_plans $nnUNet_preprocessed
    nnUNetv2_migrate_plans $nnUNet_results

Both trees are safe to point it at: it recurses, only touches files named plans.json (or
*Plans*.json), and only ever rewrites the one `network_class_name` field.
"""
import argparse
import json
import os
import shutil
from typing import List, Tuple

from nnunetv2.utilities.plans_handling.plans_handler import LEGACY_NETWORK_CLASS_NAMES


def find_plans_files(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "plans.json" or (fn.endswith(".json") and "Plans" in fn):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def migrate_one(path: str, dry_run: bool, backup: bool) -> Tuple[int, List[str]]:
    """-> (number of configurations rewritten, human-readable notes)"""
    try:
        with open(path) as f:
            plans = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return 0, [f"  !! could not read ({type(e).__name__}: {e})"]

    if not isinstance(plans, dict) or "configurations" not in plans:
        return 0, []

    n, notes = 0, []
    for cfg_name, cfg in plans["configurations"].items():
        if not isinstance(cfg, dict):
            continue
        arch = cfg.get("architecture")
        if not isinstance(arch, dict):
            continue
        old = arch.get("network_class_name")
        if not isinstance(old, str) or not old.startswith("dynamic_network_architectures."):
            continue
        new = LEGACY_NETWORK_CLASS_NAMES.get(old)
        if new is None:
            notes.append(f"  !! {cfg_name}: unknown legacy class '{old}' -- left unchanged")
            continue
        notes.append(f"  {cfg_name}: {old}\n      -> {new}")
        if not dry_run:
            arch["network_class_name"] = new
        n += 1

    if n and not dry_run:
        if backup:
            shutil.copy2(path, path + ".pre_vendoring_backup")
        # write to a sibling temp file then replace, so an interrupted run cannot truncate the original
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(plans, f, sort_keys=False, indent=4)
        os.replace(tmp, path)
    return n, notes


def entry_point():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=str,
                   help="folder to search recursively for plans.json files, e.g. $nnUNet_preprocessed")
    p.add_argument("--dry_run", action="store_true",
                   help="report what would change without writing anything")
    p.add_argument("--no_backup", action="store_true",
                   help="do not leave a .pre_vendoring_backup copy next to each rewritten file")
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        raise NotADirectoryError(args.folder)

    files = find_plans_files(args.folder)
    print(f"found {len(files)} candidate plans file(s) under {args.folder}\n")

    touched = total_cfgs = 0
    for f in files:
        n, notes = migrate_one(f, args.dry_run, backup=not args.no_backup)
        if n or notes:
            print(f"{f}")
            for line in notes:
                print(line)
            print()
        if n:
            touched += 1
            total_cfgs += n

    verb = "would rewrite" if args.dry_run else "rewrote"
    print(f"{verb} {total_cfgs} configuration(s) across {touched} file(s).")
    if args.dry_run and total_cfgs:
        print("re-run without --dry_run to apply.")
    elif touched and not args.no_backup:
        print("originals kept alongside as *.pre_vendoring_backup")


if __name__ == "__main__":
    entry_point()
