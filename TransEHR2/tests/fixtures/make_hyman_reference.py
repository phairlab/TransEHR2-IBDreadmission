"""Regenerate ``hyman_reference.npz`` from R's own ``splinefun``.

The Python port in ``TransEHR2/hyman.py`` has no independent authority:
it is a transcription, and the only thing that makes it trustworthy is
agreeing with the implementation it was transcribed from. This script
runs that implementation and records what it said, so the test can check
the port without R being installed.

Needs Rscript on PATH. Run it from anywhere:

    python TransEHR2/tests/fixtures/make_hyman_reference.py

The cases are kept small enough to live in the repository: a hundred-odd
curves at 121 points apiece is far more than a transcription error can
hide in, and a wrong branch shows up in the first few.

The cases are chosen for where a monotone filter earns its keep: curves
that saturate inside the first bin and then sit flat, curves that are
flat throughout, and curves whose knots are far enough apart that an
unfiltered cubic would overshoot between them. Decreasing curves are in
there too -- nothing in this project feeds the spline one, but R accepts
both directions and a port that quietly handled only one would be a port
with an untested branch.
"""

import numpy as np
import subprocess
import tempfile

from pathlib import Path


R_SCRIPT = r'''
set.seed(20261001)
out <- list()

make <- function(tag, x, y) {
  xout <- seq(min(x), max(x), length.out = 121)
  fit <- t(apply(y, 1, function(r)
    stats::splinefun(x, r, method = "hyman")(xout)))
  write.csv(x,    sprintf("%s/%s_x.csv", DIR, tag), row.names = FALSE)
  write.csv(y,    sprintf("%s/%s_y.csv", DIR, tag), row.names = FALSE)
  write.csv(xout, sprintf("%s/%s_xout.csv", DIR, tag), row.names = FALSE)
  write.csv(fit,  sprintf("%s/%s_fit.csv", DIR, tag), row.names = FALSE)
}

# The production grid: the default cuts with the known F(0) = 0 anchor.
x6 <- c(0, 30, 60, 90, 365, 1095, 1825)
rand <- t(sapply(1:60, function(i) {
  p <- c(0, cumsum(runif(6)^3))
  p / (p[7] / runif(1, 0.02, 0.98))
}))
edge <- rbind(
  rep(0, 7),                                  # never fails
  c(0, rep(0.9, 6)),                          # saturates inside bin 0
  c(0, 0, 0, 0, 0, 0, 0.6),                   # nothing until the last bin
  c(0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0),         # flat, then a late jump
  seq(0, 1, length.out = 7),                  # straight line
  c(0, 1e-18, 1e-17, 0.4, 0.4, 0.4, 0.4)      # ~zero, then a step
)
make("cif6", x6, rbind(rand, edge))

# A short grid: four knots is the fewest at which FMM's end conditions
# have four points to take a divided difference over.
x3 <- c(0, 30, 90, 365)
make("cif3", x3, t(sapply(1:20, function(i) {
  p <- c(0, cumsum(runif(3)^2)); p / (p[4] / runif(1, 0.02, 0.98))
})))

# Decreasing, survival-shaped, on the same production grid.
make("surv6", x6, t(sapply(1:20, function(i) {
  p <- c(0, cumsum(runif(6)^3)); 1 - p / (p[7] / runif(1, 0.02, 0.98))
})))
'''


def main() -> None:
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / 'reference.R'
        script.write_text(f'DIR <- "{tmp}"\n{R_SCRIPT}')
        subprocess.run(['Rscript', '--vanilla', str(script)], check=True)

        arrays = {}
        for path in sorted(Path(tmp).glob('*.csv')):
            arrays[path.stem] = np.loadtxt(path, delimiter=',', skiprows=1,
                                           ndmin=2)

    target = here / 'hyman_reference.npz'
    np.savez_compressed(target, **arrays)
    print(f'Wrote {target} with {len(arrays)} arrays: '
          f'{", ".join(sorted(arrays))}')


if __name__ == '__main__':
    main()
