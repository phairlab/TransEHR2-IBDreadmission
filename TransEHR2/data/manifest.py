"""``manifest.csv`` lookup and checksum verification.

The global lookup tables are *order-sensitive*: a row of
``text_embeddings.npy`` is meaningful only as the embedding of the string
at that position of ``text_strings.pkl``, and a row of
``drug_embeddings.npy`` only as the ClinVec vector the drug preparation
step indexed against. Neither shape nor dtype can catch a table that
copied wrong, so the checksum is verified **when the table is opened**
rather than only when it is fetched, and this module is what the loader
and ``embed.py`` share to do it.

Design decisions this module commits to
---------------------------------------

* **``update_manifest.sh``'s rule holds here too: a row is updated, never
  added.** The shell script refuses to add entries so that data which
  cannot legally be shared is not registered for distribution by
  accident; a Python path into the same file that added rows freely would
  walk straight around that. ``record_checksum`` therefore declines a
  path that is not already a row, and the lookup-table rows were added to
  ``manifest.csv`` by hand.
* **``DATA_ROOT`` is resolved the way the shell scripts resolve it** --
  ``SHARED_DATA_ROOT`` if set, else ``<project root>/../data`` -- because
  the manifest's paths are relative to it and a second convention would
  make the same file mean two things.
* **A table outside ``DATA_ROOT``, or inside it but unregistered, is
  reported and not verified.** Fixtures and temporary-directory cohorts
  are not distributed artifacts and have no manifest row; refusing them
  would make every test carry a checksum of a file it just wrote. The
  report goes to stdout so that an unregistered table in a *real* run is
  visible rather than silently unchecked.
"""

import csv
import hashlib
import os

from typing import Dict, Optional


# .../TransEHR2-IBDreadmission/TransEHR2/TransEHR2/data/this_file.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'manifest.csv')


def data_root() -> str:
    """``SHARED_DATA_ROOT``, else ``<project root>/../data``."""
    shared = os.environ.get('SHARED_DATA_ROOT')
    if shared:
        return os.path.abspath(shared)
    return os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, 'data'))


def manifest_key(path: str) -> Optional[str]:
    """The manifest's name for ``path``, or None if it is outside DATA_ROOT.

    Manifest paths are relative to ``DATA_ROOT`` and always use forward
    slashes, which is what ``setup_data.sh`` writes and reads.
    """
    root = data_root()
    absolute = os.path.abspath(path)
    if os.path.commonpath([root, absolute]) != root:
        return None
    return os.path.relpath(absolute, root).replace(os.sep, '/')


def read_manifest() -> Dict[str, dict]:
    """``manifest.csv`` as ``{path: row}``.

    Returns an empty mapping when the file is absent, so a checkout that
    has not run ``setup_data.sh`` reports rather than crashes.
    """
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, newline='') as f:
        return {
            row['path'].lstrip('﻿'): row
            for row in csv.DictReader(f)
            if row.get('path') and not row['path'].startswith('#')
        }


def sha256_of(path: str) -> str:
    """Stream ``path`` through SHA-256.

    Streamed rather than read whole: ``text_embeddings.npy`` is tens of
    gigabytes once the cohort's unique strings are embedded.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(path: str) -> None:
    """Check ``path`` against its ``manifest.csv`` row.

    Raises:
        ValueError: The file is registered and its checksum disagrees.
    """
    key = manifest_key(path)
    if key is None:
        print(f"  {path} is outside DATA_ROOT ({data_root()}); "
              f"not checksum-verified.")
        return
    row = read_manifest().get(key)
    if row is None:
        print(f"  {key} is not in {MANIFEST_PATH}; not checksum-verified.")
        return
    expected = row['sha256'].strip()
    if expected in ('', 'n/a', 'pending'):
        print(f"  {key} has no checksum in {MANIFEST_PATH} "
              f"('{expected}'); not verified. Rebuild it with embed.py, "
              f"which records the checksum it wrote.")
        return
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"{key} does not match its checksum in "
            f"{MANIFEST_PATH}.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"The lookup tables are order-sensitive, so a table that "
            f"copied wrong gathers the wrong rows without changing "
            f"shape. Re-fetch it with setup_data.sh, or rebuild "
            f"it with embed.py and re-run update_manifest.sh."
        )


def record_checksum(path: str) -> Optional[str]:
    """Rewrite the ``sha256`` of ``path``'s existing row, never add one.

    Returns the checksum written, or None when the file has no row --
    a fixture or a tmp-dir cohort, neither of which is distributed.

    Raises:
        FileNotFoundError: ``manifest.csv`` is missing, but the file is
            under ``DATA_ROOT`` and so ought to have a row in it.
    """
    key = manifest_key(path)
    if key is None:
        return None
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found, but {key} is under DATA_ROOT "
            f"and is verified against the manifest when it is opened."
        )
    with open(MANIFEST_PATH, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    checksum = sha256_of(path)
    for row in rows:
        if row['path'].lstrip('﻿') == key:
            row['sha256'] = checksum
            break
    else:
        # update_manifest.sh's rule: new entries are added by hand, so
        # that data which cannot be shared is never registered for
        # distribution by a script.
        print(f"  {key} is not in {MANIFEST_PATH}; its checksum was not "
              f"recorded. Add the row by hand, then re-run this script.")
        return None

    with open(MANIFEST_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return checksum
