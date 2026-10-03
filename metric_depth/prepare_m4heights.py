"""Download a bounded subset on the training server; train directly from ZIPs."""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
import shutil

from dataset.m4heights_files import index_tiffs


REPO_ID = 'Rituxx96x/M4Heights'
LABEL_FILE = 'TRAIN/LABEL/LABEL_10m.zip'
METADATA_FILE = 'metadata/train_data.csv'


def selected_files(shards):
    if not shards or len(set(shards)) != len(shards) or any(s < 1 or s > 10 for s in shards):
        raise ValueError('Choose distinct ORTHO shard numbers between 1 and 10')
    return [f'TRAIN/ORTHO/ORTHO_1m_NOGEO_{s}.zip' for s in shards] + [LABEL_FILE, METADATA_FILE]


def make_download_plan(info, filenames, max_download_gb):
    if not math.isfinite(max_download_gb) or max_download_gb <= 0:
        raise ValueError('--max-download-gb must be finite and positive')
    sizes = {entry.rfilename: entry.size for entry in info.siblings}
    files = []
    for name in filenames:
        size = sizes.get(name)
        if size is None or size <= 0:
            raise ValueError(f'Missing file or size on Hugging Face: {name}')
        files.append({'path': name, 'size': size})
    total = sum(entry['size'] for entry in files)
    if total > max_download_gb * 1_000_000_000:
        raise ValueError(f'Selection is {total / 1e9:.3f} GB, above the {max_download_gb:g} GB budget. '
                         'Choose fewer --ortho-shards or increase --max-download-gb.')
    return {'repo_id': REPO_ID, 'revision': info.sha, 'files': files, 'total_bytes': total}


def split_samples(samples, val_fraction=0.1, seed=42, group_size=100, val_countries=None):
    if not 0 < val_fraction < 1 or group_size < 1:
        raise ValueError('Expected 0 < --val-fraction < 1 and --split-group-size >= 1')
    groups = {}
    for sample in samples:
        match = re.fullmatch(r'img_([A-Z]{3})_(\d{4})_(\d+)\.(?:tif|tiff)', sample['id'])
        if match is None:
            raise ValueError(f'Unexpected M4Heights tile name: {sample["id"]}')
        country, year, number = match.groups()
        key = (country, year, int(number) // group_size)
        groups.setdefault(key, []).append(sample)
    for records in groups.values():
        records.sort(key=lambda sample: sample['id'])
    if val_countries:
        countries = set(val_countries)
        absent = countries - {key[0] for key in groups}
        if absent:
            raise ValueError(f'Validation countries absent from subset: {sorted(absent)}')
        val_keys = {key for key in groups if key[0] in countries}
    else:
        if len(groups) < 2:
            raise ValueError('Need at least two tile groups for train/val. Select more shards '
                             'or reduce --split-group-size (1 gives a random tile split).')
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        count = max(1, min(len(keys) - 1, round(len(keys) * val_fraction)))
        val_keys = set(keys[:count])
    train = [sample for key in sorted(groups) if key not in val_keys for sample in groups[key]]
    val = [sample for key in sorted(groups) if key in val_keys for sample in groups[key]]
    if not train or not val:
        raise ValueError('Selection must contain both training and validation tiles')
    return train, val


def prepare_manifests(root, shards, val_fraction=0.1, seed=42, group_size=100, val_countries=None):
    root = Path(root)
    filenames = selected_files(shards)
    images = index_tiffs(root, filenames[:-2])
    labels = index_tiffs(root, [LABEL_FILE])
    with (root / METADATA_FILE).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if 'fname' not in (reader.fieldnames or []):
            raise ValueError(f'Expected fname column in {METADATA_FILE}')
        known = {Path(row['fname']).name for row in reader}
    unknown = images.keys() - known
    if unknown:
        print(f'Skipping {len(unknown)} ORTHO tiles absent from train metadata: {sorted(unknown)[:5]}')
        images = {name: reference for name, reference in images.items() if name in known}
    if not images:
        raise ValueError('No selected ORTHO tiles are listed in train metadata')
    missing_labels = images.keys() - labels.keys()
    if missing_labels:
        raise ValueError(f'Training tiles listed in metadata have missing labels: {sorted(missing_labels)[:5]}')
    samples = [{'id': name, 'image': images[name], 'height': labels[name]} for name in sorted(images)]
    train, val = split_samples(samples, val_fraction, seed, group_size, val_countries)
    split_dir = root / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = root / 'subset_download.json'
    provenance = {'repo_id': REPO_ID, 'revision': None, 'files': filenames}
    if provenance_path.exists():
        downloaded = json.loads(provenance_path.read_text(encoding='utf-8'))
        entries = {entry['path']: entry for entry in downloaded['files']}
        if all(name in entries and (root / name).stat().st_size == entries[name]['size'] for name in filenames):
            provenance = {**downloaded, 'files': [entries[name] for name in filenames],
                          'total_bytes': sum(entries[name]['size'] for name in filenames)}
    for name, records in [('train', train), ('val', val)]:
        manifest = {'version': 1, 'data_root': '..', 'source': provenance,
                    'split': {'seed': seed, 'group_size': group_size, 'val_fraction': val_fraction,
                              'val_countries': val_countries}, 'samples': records}
        (split_dir / f'{name}.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    countries = sorted({sample['id'].split('_')[1] for sample in samples})
    print(f'Prepared {len(train)} train / {len(val)} val tiles; countries: {", ".join(countries)}')
    print(f'Manifests: {split_dir / "train.json"} and {split_dir / "val.json"}')
    return train, val


def download_subset(args):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    # Default HF authentication: HF_TOKEN or `hf auth login`. Never store the token.
    api = HfApi()
    try:
        info = api.dataset_info(REPO_ID, revision=args.revision, files_metadata=True)
    except GatedRepoError as exc:
        raise ValueError(access_instructions()) from exc
    plan = make_download_plan(info, selected_files(args.ortho_shards), args.max_download_gb)
    for entry in plan['files']:
        print(f'{entry["size"] / 1e9:8.3f} GB  {entry["path"]}')
    print(f'Total selected files: {plan["total_bytes"] / 1e9:.3f} GB (decimal GB); revision {plan["revision"]}')
    if args.dry_run:
        return
    root = Path(args.data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    missing_bytes = sum(entry['size'] for entry in plan['files']
                        if not (root / entry['path']).is_file()
                        or (root / entry['path']).stat().st_size != entry['size'])
    if shutil.disk_usage(root).free < missing_bytes * 1.05:
        raise ValueError(f'Insufficient disk space: need approximately {missing_bytes / 1e9:.2f} GB plus overhead')
    for entry in plan['files']:
        try:
            hf_hub_download(REPO_ID, entry['path'], repo_type='dataset', revision=plan['revision'], local_dir=root)
        except GatedRepoError as exc:
            raise ValueError(access_instructions()) from exc
    (root / 'subset_download.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')


def access_instructions():
    return (f'Accept access conditions at https://huggingface.co/datasets/{REPO_ID}, '
            'then run hf auth login on this server or set HF_TOKEN.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, help='Destination on the training server')
    parser.add_argument('--ortho-shards', nargs='+', type=int, default=[2, 10], help='Default: 2 10, about 10.834 GB total')
    parser.add_argument('--max-download-gb', type=float, default=20.0, help='Maximum total size of selected files, decimal GB')
    parser.add_argument('--revision', default='main')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='Print sizes without downloading or writing files')
    modes.add_argument('--prepare-only', action='store_true', help='Build manifests from existing ZIPs, with no network access')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--val-countries', nargs='+', help='Hold out entire countries, e.g. NLD')
    parser.add_argument('--split-group-size', type=int, default=100, help='Keep consecutive tile-ID blocks together; not a geographic guarantee')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    try:
        selected_files(args.ortho_shards)
        if not 0 < args.val_fraction < 1 or args.split_group_size < 1:
            raise ValueError('Expected 0 < --val-fraction < 1 and --split-group-size >= 1')
        if not args.prepare_only:
            download_subset(args)
        if not args.dry_run:
            prepare_manifests(Path(args.data_root).resolve(), args.ortho_shards,
                              args.val_fraction, args.seed, args.split_group_size, args.val_countries)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
