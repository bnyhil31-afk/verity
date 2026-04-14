## Setup (one-time, before first release)

Trusted Publishing must be configured on PyPI, TestPyPI, and GitHub
before the workflow can publish. Do this once:

### On TestPyPI
1. Go to https://test.pypi.org/manage/account/publishing/
2. Add a pending publisher:
   - PyPI project name: verity
   - Owner: bnyhil31-afk
   - Repository name: verity
   - Workflow name: release.yml
   - Environment name: testpypi

### On PyPI
1. Go to https://pypi.org/manage/account/publishing/
2. Add a pending publisher:
   - PyPI project name: verity
   - Owner: bnyhil31-afk
   - Repository name: verity
   - Workflow name: release.yml
   - Environment name: pypi

### On GitHub
1. Go to Settings → Environments
2. Create environment: testpypi (no protection rules needed)
3. Create environment: pypi
   - Add protection rule: Required reviewers → add yourself
   - This gates the PyPI publish step behind manual approval

Once setup is complete, skip to "Publishing a release" below.

## Publishing a release

    git tag v0.2.0
    git push origin v0.2.0

The workflow will:
  1. Build the distribution
  2. Publish to TestPyPI automatically
  3. Pause for your manual approval (pypi environment protection)
  4. Publish to PyPI

---

# Verity v0.1.0 Release Checklist

## Pre-upload verification

- [ ] All CI checks green on main branch
- [ ] `python -m build` succeeds locally
- [ ] `twine check dist/*` passes with no errors
- [ ] Wheel contents look correct (run `python -m zipfile -l dist/*.whl`)

## Check name availability

Verify `verity` is not already taken on PyPI:

    pip install verity  # should fail with "No matching distribution found"

If the name is taken, consider: `verity-memory`, `verity-ai`, or `verity-cognitive`.
Update `name` in pyproject.toml, rebuild, and re-check.

## Upload to TestPyPI first

    # Upload (you will be prompted for API token)
    twine upload --repository testpypi dist/*

    # Install and verify from TestPyPI
    pip install --index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ \
                verity

    # Quick smoke test
    python -c "from verity import Memory; m = Memory(); m.add('test'); print(m.search('test'))"

If TestPyPI install and smoke test pass, proceed.

## Upload to PyPI

    twine upload dist/*

Verify the release page: https://pypi.org/project/verity/

## Tag the release

    git tag v0.1.0
    git push origin v0.1.0

## Create GitHub Release

Go to: https://github.com/bnyhil31-afk/verity/releases/new
- Tag: v0.1.0
- Title: v0.1.0 — Initial Release
- Body: paste the [0.1.0] section from CHANGELOG.md
- Attach: dist/verity-0.1.0-py3-none-any.whl and dist/verity-0.1.0.tar.gz

## Announce

Update README.md badge if you add a PyPI version badge:

    [![PyPI version](https://badge.fury.io/py/verity.svg)](https://badge.fury.io/py/verity)

---
Generated for: verity v0.1.0 release
