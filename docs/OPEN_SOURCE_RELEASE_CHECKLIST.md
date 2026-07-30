# SpectralReasoner Release Checklist

## Before Publishing

- [ ] Confirm `https://github.com/lll178/spectral-reasoner` exists and is public.
- [ ] Confirm the author or business contact method is visible on the GitHub profile or repository discussion page.
- [ ] Review `LICENSE` with counsel if commercial enforcement matters.
- [ ] Review dependency licenses for the exact packaged artifact.
- [ ] Confirm no external datasets, model weights, run outputs, caches, logs, or private files are included.
- [ ] Run smoke tests for `/chat`, `/generate-chat`, and disabled legacy debug endpoints.

## Suggested Commands

```powershell
Get-ChildItem src\spectral_reasoner -Recurse -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }

python tools\package_spectral_reasoner.py --out dist\spectral_reasoner_release_clean --clean

rg "GPL|AGPL|LGPL|Apache|MIT|BSD|license" -S dist\spectral_reasoner_release_clean
```

## Publish

Include only:

- `src/`
- `README.md`
- `LICENSE`
- `COMMERCIAL_LICENSE.md`
- `THIRD_PARTY_NOTICES.md`
- `requirements.txt`
- `pyproject.toml`
- `.gitignore`
- `.gitattributes`
- selected `docs/`
- selected `tools/`

Do not include generated local assets, private files, caches, logs, historical
experiments, or unrelated research artifacts.
