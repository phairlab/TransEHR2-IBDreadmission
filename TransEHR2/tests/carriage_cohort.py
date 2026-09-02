"""The cohort the sparse-carriage probes and their baseline share.

Its own module because ``fixtures/sparse_lookup_baseline.npz`` is
captured from the tree as it stood before the carriage changed, where
``densify_lookup_slots`` does not exist yet and so
``test_sparse_lookup_carriage`` cannot be imported.
"""

from pathlib import Path

from .conftest import MiniRoot


def build_carriage_root(tmp_path):
    """Three episodes: no entries, one entry, several entries.

    The three cases the densification has to get right, present for both
    members of the family at once. ``UB`` goes for the reason
    ``test_lookup_family`` drops it -- two levels is one short of what
    ``BetaLoss`` needs -- and nothing here builds a model anyway.
    """
    mini = MiniRoot(Path(tmp_path))
    mini.config = dict(mini.config)
    mini.config['VALUED_FEATS'] = ['NUM', 'CAT', 'ORD']
    del mini.var_properties['UB']

    # No text, no drugs: every lookup timestep stays a zero vector.
    mini.add_patient(
        2001,
        timeseries=[
            ['2019-01-01T00:00:00Z', 1.0, 'L', '0', '', '', ''],
            ['2019-01-02T00:00:00Z', 2.0, 'U', '1-24', '', '', 1],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-02T00:00:00Z')],
    )
    # One text record and one drug timestep, the latter two slots deep.
    mini.add_patient(
        2002,
        timeseries=[
            ['2019-02-01T00:00:00Z', 1.5, 'L', '0', '', 'a note', ''],
            ['2019-02-02T00:00:00Z', 2.5, 'U', '25-50', '', '', 1],
        ],
        stays=[('AMB', '2019-02-01T00:00:00Z', '2019-02-02T00:00:00Z')],
        drugs=[('2019-02-01T00:00:00Z', 0, 2, 1.0),
               ('2019-02-01T00:00:00Z', 1, 3, 0.5)],
    )
    # Several of both, and a drug timestep that fills every slot beside
    # one that fills a single slot.
    mini.add_patient(
        2003,
        timeseries=[
            ['2019-03-01T00:00:00Z', 0.5, 'U', '1-24', '', 'a note', 1],
            ['2019-03-02T00:00:00Z', 2.0, 'L', '25-50', '', 'second', ''],
            ['2019-03-03T00:00:00Z', 3.0, 'U', '0', '', 'third note', 1],
            ['2019-03-04T00:00:00Z', 4.0, 'L', '1-24', '', 'fourth', ''],
        ],
        stays=[('DAD', '2019-03-01T00:00:00Z', '2019-03-04T00:00:00Z')],
        drugs=[('2019-03-01T00:00:00Z', 0, 0, 1.0),
               ('2019-03-01T00:00:00Z', 1, 1, 2.0),
               ('2019-03-01T00:00:00Z', 2, 2, 0.5),
               ('2019-03-03T00:00:00Z', 0, 3, 1.5)],
    )
    mini.add_fold('fold0', train=[0, 1, 2], val=[0, 1, 2], test=[0, 1, 2])
    return mini
