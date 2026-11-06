# Take-home: ICD-11 coding at real scale

Sugen Technologies, Engineering application

You have 3 days from the moment we send this. We do not review late submissions. There is no cap on hours inside the window. The score is a number, and it keeps rising with what you put in.

Use whatever AI tooling you normally work with: Claude Code, Cursor, Codex, anything. Commit as you go. We read the history as part of the work.

## The job

Sugen runs outpatient clinics. A clinician sees a patient and either types a note or lets an ambient scribe record the consultation. Every visit needs an ICD-11 code, because the code drives the insurance claim, and a wrong code means a rejected claim.

Build the system that reads the note and proposes the code.

## What you get

- `data/dev_notes.jsonl`: 60 consult notes with their gold codes. Each line looks like `{"id", "text", "gold": [codes], "style"}`.
- `data/mms-2026-01.json`: every diagnostic category in ICD-11 MMS 2026-01 (chapters 01 to 26), with code, title, chapter, leaf and residual flags. These are the only codes that score.
- `score.py`: the scorer. We grade you with this exact script.

That is all we give you, and it is deliberate. The code list has titles only. WHO publishes a great deal more about ICD-11 than titles, and whether you go and get it is your call.

## What we grade on

We run your system, unmodified, on 600 notes you will never see, then run `score.py`. Nothing else is scored.

| Points | What | How |
|---|---|---|
| 75 | Core: 450 notes from the same clinics and styles as the dev set | 50 x top-1 accuracy + 25 x top-3 accuracy |
| 25 | Generalisation: 150 notes from a different distribution than the dev set | 16.7 x top-1 + 8.3 x top-3 |

Top-1 means your first code is in the note's gold set. Top-3 means a gold code is anywhere in your first three.

Gold is the most specific code the note supports. A parent code does not count when a child fits, and an unspecified (`.Z`) code is right when the note does not give the detail.

A word on the generalisation slice. The 60 dev notes are a small, narrow sample. The hidden set is not. Real clinics see every chapter of ICD-11, and real consultations are not always in English. A system tuned to the 60 notes will lose most of those 25 points.

## The run

From a clean clone, this must work:

```
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
```

The input is JSONL, one `{"id", "text"}` per line. The output is JSONL, one `{"id", "codes": [up to 3, best first]}` per line, as in `data/predictions.example.jsonl`. Every input id must appear in the output. A missing note scores 0, and so does a code that is not in `mms-2026-01.json`.

Our grading box has 8 CPUs, 16 GB of RAM and no GPU. On it:

- You have 60 minutes for all 600 notes.
- You may call OpenAI or Anthropic models. Read `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` from the environment. We supply the keys, capped at US$6 for the whole run, which is about 1 cent per note.
- At run time the network is model APIs only. `docker compose build` may use the network. `docker compose run` cannot reach anything except `api.openai.com` and `api.anthropic.com`, and only through an HTTPS proxy we pass in `HTTPS_PROXY`. The Python OpenAI and Anthropic SDKs honour it by default. Any other client has to be configured to use it, or every API call fails and you fall back to your no-key path. Anything else you need (indexes, embeddings, downloaded data) must be built ahead of time and ship in your repo or image.
- Notes not finished when you hit a cap score 0, so write predictions as you go.
- With no key set, the system must still produce an answer for every note. It will just be worse.
- If the image does not build from a clean clone, or more than 10% of notes come back without a prediction, we reject the submission without review. Run the exact command above from a fresh clone before you submit.

## Deliverable

1. Your repo as a git bundle, made with `git bundle create <yourname>.bundle --all`, attached to a reply on the email thread that sent you this pack. The email's timestamp is your submission time. Keep the real history, not a squash.
2. A README of one page at most. It has to cover:
   - your dev score from `score.py`, and your own estimate of your hidden score;
   - what data you added and where it came from;
   - how you evaluated beyond the 60 dev notes;
   - the one place your AI tooling was wrong and you caught it.
3. `DECISIONS.md`, a log you keep as you go rather than write at the end. Add an entry whenever you make a real choice: an approach, a model, a data source, a rule, a trade-off against the time or money cap, or something you tried and dropped. Each entry gives:
   - the date and time, and the commit it goes with;
   - the decision, and the options you considered;
   - what you measured or saw that decided it (a score, a failure case, a cost);
   - what you would revisit with more time.

   Short entries are fine. We expect roughly 10 to 30 over the window. Commit the log with the work it describes, so its history matches your code's.
4. One line saying how many hours you spent.

## Questions

Email is fine during the window, and a good question costs you nothing. We will not tell you anything about the hidden notes beyond what is written here.
