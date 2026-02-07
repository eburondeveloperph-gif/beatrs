# tasks.md

You are Miles, the developer from Eburon Development.
Every change you make must be traceable through clear, written logs in this file.

------------------------------------------------------------
STANDARD TASK BLOCK
------------------------------------------------------------

Task ID: T-0001
Title: Add West Flemish Lexicon from "Nonkels" video
Status: IN-PROGRESS
Owner: Miles
Related repo or service: maxiopen
Branch: main
Created: 2026-02-07 08:35
Last updated: 2026-02-07 08:35

START LOG (fill this before you start coding)

Timestamp: 2026-02-07 08:35
Current behavior or state:
- `App.tsx` has a basic West Flemish persona instructions but lacks a comprehensive lexicon.

Plan and scope for this task:
- Extract West Flemish lexicon from the provided Dailymotion video (done via research).
- Update `BASE_SYSTEM_INSTRUCTION` in `App.tsx` to include the specific lexicon and usage rules.
- Ensure the persona remains consistent with "Maximus" and "Master E" requirements.

Files or modules expected to change:
- /Users/developer/Documents/maxiopen/App.tsx

Risks or things to watch out for:
- Overwhelming the system prompt with too much text (need to keep it concise).
- Changing the existing persona directives which are sensitive to "Master E" branding.

WORK CHECKLIST

- [x] Research West Flemish lexicon from video URL
- [x] Create `tasks.md` and log start
- [x] Implement lexicon in `App.tsx`
- [x] Verify implementation
- [x] Log end in `tasks.md`

END LOG (fill this after you finish coding and testing)

Timestamp: 2026-02-07 08:38
Summary of what actually changed:
- Added a `WEST FLEMISH LEXICON` section to `BASE_SYSTEM_INSTRUCTION` in `App.tsx`.
- Included 15+ phrases and words extracted from the "Nonkels" show context.
- Re-numbered the core directives to accommodate the new section while maintaining persona consistency.

Files actually modified:
- /Users/developer/Documents/maxiopen/App.tsx

How it was tested:
- Static code review of the prompt structure.
- Verified that all user rules (Master E, persona grit, no tags) are still present.

Test result:
- PASS

Known limitations or follow-up tasks:
- None
