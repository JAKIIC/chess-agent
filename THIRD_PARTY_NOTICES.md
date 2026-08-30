# Third-party notices

## Pikafish

The application can use Pikafish as a separately launched UCI engine. Pikafish is not
stored in this repository and is not covered by this repository's MIT license.

- Project: `official-pikafish/Pikafish`
- Pinned release: `Pikafish-2026-01-02`
- Pinned source commit: `ce0679e00ee196f7ba17f6ec18941b9a5036f8cf`
- License: GNU General Public License version 3 (`GPL-3.0`)
- Exact source: <https://github.com/official-pikafish/Pikafish/tree/ce0679e00ee196f7ba17f6ec18941b9a5036f8cf>
- License text: <https://github.com/official-pikafish/Pikafish/blob/ce0679e00ee196f7ba17f6ec18941b9a5036f8cf/Copying.txt>

The installer downloads the unmodified release archive from the official GitHub release,
verifies its pinned size and SHA-256 digest, and keeps it under the gitignored `.local/`
directory. If a future distribution includes Pikafish binaries, that distribution must
also satisfy Pikafish's GPLv3 source and license obligations. No engine binary or neural
network file may be committed to this repository.

### Pikafish NNUE weights

The official release archive also contains `pikafish.nnue` and `NNUE-License.md`. The
weights have separate terms: use must be legal, and commercial use requires permission
unless the user or organization is listed by the Pikafish project. The learning assistant
uses the weights only for the project's explicitly non-commercial, legal modes: practice
against the computer, endgame training, and post-game review. It must not be used for
online cheating. The exact notice is available in the verified local archive and at:

<https://github.com/official-pikafish/Pikafish/blob/ce0679e00ee196f7ba17f6ec18941b9a5036f8cf/NNUE-License.md>
