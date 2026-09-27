# Releasing

Releases are published to [PyPI](https://pypi.org/p/video-beep-remover) by the [Release workflow](../.github/workflows/release.yml). It uses PyPI's [trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token is stored in the repository or its secrets.

## One-time setup

1. **PyPI.** Sign in to PyPI and open *Your projects → Publishing → Add a new pending publisher*. Enter:

   | Field | Value |
   |---|---|
   | PyPI project name | `video-beep-remover` |
   | Owner | `AshitakaLax` |
   | Repository name | `video-beep-remover` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The first upload then creates the project. After that, the publisher appears under the project's own *Publishing* settings.
2. **TestPyPI (optional).** For trial releases, add the same pending publisher on [TestPyPI](https://test.pypi.org), with the environment name `testpypi`.
3. **GitHub.** Create the environments `pypi` and `testpypi` under *Settings → Environments*. Add required reviewers to `pypi` if a person should approve each release.

## Cutting a release

1. Pick the version: `MAJOR.MINOR.PATCH`, following [Semantic Versioning](https://semver.org/).
2. Set it in `src/video_beep_remover/__init__.py`.
3. In `CHANGELOG.md`, rename `[Unreleased]` to that version and add an empty `[Unreleased]` above it. Update the links at the bottom of the file.
4. Merge the change to `main`. CI's `package` job builds the wheel and sdist, checks them with `twine check`, and runs the installed `vbr`.
5. *(Optional)* Try the release on TestPyPI first. Run the Release workflow from the *Actions* tab, which publishes the build to TestPyPI, then install it:

   ```console
   $ pipx install --index-url https://test.pypi.org/simple/ --pip-args="--extra-index-url https://pypi.org/simple/" video-beep-remover
   ```

   A version can be uploaded to TestPyPI only once, like on PyPI.
6. Tag the merge commit and push the tag:

   ```console
   $ git tag v0.1.0
   $ git push origin v0.1.0
   ```

   The workflow then does the following:
   - checks that the tag matches `__version__` and that `CHANGELOG.md` has a section for it;
   - builds and checks the package;
   - publishes the package to PyPI (after approval, if the `pypi` environment requires it);
   - creates a GitHub release with the changelog section as its notes and the built files attached.

A version on PyPI can't be replaced. If something is wrong with a release, fix it and release the next patch version.
