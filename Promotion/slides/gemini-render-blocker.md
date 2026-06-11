# Gemini / Nanobanana Render Blocker

Date: 2026-04-08

## Summary

Direct slide image generation through local `gemini` is currently blocked on this machine by an authentication / authorization failure, not by the slide prompts themselves.

## Confirmed Facts

- `gemini` starts successfully.
- Local binary path:
  - `/opt/homebrew/bin/gemini`
- CLI version:
  - `0.35.0`
- `gemini extensions list` shows:
  - `Loaded cached credentials.`
  - `No extensions installed.`
- Therefore `nanobanana` is not installed yet.

## Runtime Warnings That Are Not The Main Blocker

- `keytar.node` is missing, so Gemini CLI falls back to `FileKeychain`.
- Running from `~/` can trigger a directory-scan warning on:
  - `/Users/weimingming/.Trash`
- These warnings are noisy but not the actual reason generation fails.

## Actual Blocker

Headless prompt execution currently fails with:

- `403 PERMISSION_DENIED`
- service endpoint:
  - `https://cloudcode-pa.googleapis.com/v1internal:generateContent`

Observed behavior:

- the CLI reports a signed-in Google account and a visible Google One AI Pro plan
- but the OAuth-backed `Code Assist` request is rejected by the backend

Interpretation:

- the blocker is the current OAuth login state or backend authorization for Gemini CLI
- this is not a prompt-format issue

## Practical Impact

- `gemini -p ...` fails before content generation
- `nanobanana` cannot be installed or used reliably from this state
- slide image rendering is therefore blocked until auth is fixed

## Current Fallback Artifacts

The deck content is still available in editable local form:

- `Promotion/slides/slide-outline.md`
- `Promotion/slides/deck-style.md`
- `Promotion/slides/slide-generation-prompts.md`
- `Promotion/slides/slides.md`
- `Promotion/slides/speaker-notes.md`

## Next Options

1. Re-authenticate Gemini CLI with a working login state:
   - `gemini`
   - `/auth signout`
   - `/auth signin`
2. If OAuth continues to return `403`, switch to `API key` authentication.
3. After auth works, install nanobanana:
   - `gemini extensions install https://github.com/gemini-cli-extensions/nanobanana`
4. Until then, continue using the local Marp deck as the canonical presentation artifact.
