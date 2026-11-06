FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them. Network is allowed at build time only.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code and data. Everything the run needs ships in the image: the run has no network
# except the model APIs through HTTPS_PROXY.
COPY src/ src/
COPY public/data/mms-2026-01.json public/data/mms-2026-01.json
# Search synonyms generated ahead of time (tools/make_synonyms.py); no network at run time.
COPY data/synonyms.jsonl data/synonyms.jsonl

# Fail the build, not the graded run, if the index or the code is broken.
RUN python -c "import sys; sys.path.insert(0, 'src'); from icd_index import IcdIndex; \
idx = IcdIndex.load('public/data/mms-2026-01.json'); assert len(idx.codes) == 18375; \
assert len(idx.synonyms) > 5000, 'synonyms missing'; print('index ok', len(idx.synonyms), 'codes with synonyms')"

ENTRYPOINT ["python", "/app/src/run.py"]
