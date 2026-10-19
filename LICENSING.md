# Licensing and dependency scope

The copyright holder authorizes distribution of the first-party source in this
MoleditPy 5 migration of **moleditpy_pmeff-plugin** under **LGPL-3.0-or-later**.
Copyright: Hiromichi Yokoyama / HiroYokoyama and the project's existing notices.
This is an additional permission for these newly distributed copies; historical
releases, tags and their original grants are not withdrawn or rewritten.

`LICENSE` contains the LGPLv3 additional permissions and `COPYING` the incorporated
GPLv3 terms. Keep both files, this notice and existing copyright notices with
source and binary distributions. Modifications to covered library code retain
the LGPL terms. Provide corresponding source and allow modification/replacement
of covered libraries when distributing a combined binary.

Third-party code, libraries, executables, models and data keep their own licenses.
This grant does not relicense another author's work. Python/package dependencies
are not included in this grant; preserve their notices and comply with their
distribution terms. Optional programs that exchange ordinary files are distinct
from libraries loaded into the Python process; do not infer LGPL compatibility
merely from running a command in another process.

Independent plugins may choose their own license under MoleditPy's documented
plugin-API permission. This project's LGPL choice does not impose the same choice
on every independently developed plugin. See the main app's LICENSING.md.

The audit covers declared dependencies, imports, source provenance, bundled
components and release packaging. Common dependencies include PySide6/Qt
(LGPLv3 option), RDKit/NumPy/SciPy/VTK (BSD-family), PyVista (MIT), Matplotlib
(PSF-based), Pillow (MIT-CMU), ASE/Paramiko (LGPL) and cryptography (Apache/BSD).
Actual dependency versions and any native/transitive libraries retain their
separate notices; this document does not grant rights to those components.
