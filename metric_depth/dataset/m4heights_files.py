"""Portable references to M4Heights TIFFs, including members of local ZIPs."""

import io
import json
from pathlib import Path
import zipfile


def load_manifest(path, data_root=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or not manifest.get('samples'):
        raise ValueError(f'Expected a nonempty M4Heights v1 manifest: {path}')
    root = (Path(data_root).expanduser().resolve() if data_root is not None
            else (path.parent / manifest['data_root']).resolve())
    for sample in manifest['samples']:
        for key in ('image', 'height'):
            resolve_reference(root, sample[key])
    return root, manifest['samples']


def resolve_reference(root, reference):
    root = Path(root).resolve()
    path = (root / reference['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Reference escapes data root: {reference["path"]}')
    return path


def read_tiff(root, reference, archive_cache=None):
    # Lazy import keeps the existing US3D environment usable without tifffile.
    import tifffile

    path = resolve_reference(root, reference)
    if 'member' in reference:
        # Only one tile is decompressed; nothing is extracted to disk.
        if archive_cache is None:
            with zipfile.ZipFile(path) as archive:
                source = io.BytesIO(archive.read(reference['member']))
        else:
            if path not in archive_cache:
                archive_cache[path] = zipfile.ZipFile(path)
            source = io.BytesIO(archive_cache[path].read(reference['member']))
    else:
        source = path
    with tifffile.TiffFile(source) as tiff:
        array = tiff.asarray()
        axes = tiff.series[0].axes
        tag = tiff.pages[0].tags.get(42113)  # GDAL_NODATA
        nodata = float(str(tag.value).strip('\x00')) if tag is not None else None
    return array, axes, nodata
