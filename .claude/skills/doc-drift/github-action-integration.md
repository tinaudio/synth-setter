# GitHub Action Integration

Wire doc-drift into a post-merge GitHub Action that automatically checks for
documentation staleness and opens a PR with fixes.

______________________________________________________________________

## Workflow

```yaml
# .github/workflows/doc-drift.yml
name: doc-drift-check
on:
  push:
    branches: [main]
    paths-ignore:
      - 'docs/**'           # doc-only changes can't cause code drift
      - 'doc-map.yaml'      # mapping changes don't need a drift check
      - '.github/**'        # workflow changes

jobs:
  check-drift:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2  # parent commit for diff

      - name: Get changed files
        id: diff
        run: |
          echo "files<<EOF" >> "$GITHUB_OUTPUT"
          git diff --name-only HEAD~1 HEAD >> "$GITHUB_OUTPUT"
          echo "EOF" >> "$GITHUB_OUTPUT"

          echo "status<<EOF" >> "$GITHUB_OUTPUT"
          git diff --name-status HEAD~1 HEAD >> "$GITHUB_OUTPUT"
          echo "EOF" >> "$GITHUB_OUTPUT"

      - name: Find candidate docs (grep-based)
        id: grep-candidates
        run: |
          candidates=""
          while IFS= read -r file; do
            [ -z "$file" ] && continue
            base=$(basename "$file")
            # Find docs referencing this file by basename or full path
            hits=$(grep -rl "$base" docs/ --include="*.md" 2>/dev/null || true)
            if [ -n "$file" ]; then
              hits="$hits"$'\n'"$(grep -rl "$file" docs/ --include="*.md" 2>/dev/null || true)"
            fi
            candidates="$candidates"$'\n'"$hits"
          done <<< "${{ steps.diff.outputs.files }}"

          # Deduplicate
          candidates=$(echo "$candidates" | sort -u | grep -v '^$')
          echo "docs<<EOF" >> "$GITHUB_OUTPUT"
          echo "$candidates" >> "$GITHUB_OUTPUT"
          echo "EOF" >> "$GITHUB_OUTPUT"

      - name: Find candidate docs (mapping-based)
        id: map-candidates
        run: |
          if [ ! -f doc-map.yaml ]; then
            echo "docs=" >> "$GITHUB_OUTPUT"
            exit 0
          fi
          # Parse doc-map.yaml for source patterns matching changed files
          # This is a simplified matcher — production version should use
          # a proper YAML parser (yq or a Python script)
          python3 -c "
          import yaml, fnmatch, sys

          with open('doc-map.yaml') as f:
              mapping = yaml.safe_load(f)

          changed = '''${{ steps.diff.outputs.files }}'''.strip().split('\n')
          candidates = set()

          for entry in mapping.get('docs', []):
              for source in entry.get('sources', []):
                  pattern = source['pattern']
                  for cf in changed:
                      if fnmatch.fnmatch(cf, pattern):
                          candidates.add(entry['doc'])
                          break

          print('\n'.join(sorted(candidates)))
          " > /tmp/map-candidates.txt

          echo "docs<<EOF" >> "$GITHUB_OUTPUT"
          cat /tmp/map-candidates.txt >> "$GITHUB_OUTPUT"
          echo "EOF" >> "$GITHUB_OUTPUT"

      - name: Combine candidates and check drift
        if: steps.grep-candidates.outputs.docs != '' || steps.map-candidates.outputs.docs != ''
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          # Combine and deduplicate candidate docs
          all_docs=$(echo -e "${{ steps.grep-candidates.outputs.docs }}\n${{ steps.map-candidates.outputs.docs }}" | sort -u | grep -v '^$')

          if [ -z "$all_docs" ]; then
            echo "No candidate docs found. Skipping drift check."
            exit 0
          fi

          echo "Candidate docs:"
          echo "$all_docs"

          # Build context payload: diff + candidate docs + mapping
          python3 scripts/ci/doc_drift_check.py \
            --diff <(git diff HEAD~1 HEAD) \
            --changed-files <(echo "${{ steps.diff.outputs.files }}") \
            --candidate-docs $all_docs \
            --mapping doc-map.yaml \
            --output /tmp/drift-report.md

      - name: Create PR if drift found
        if: steps.grep-candidates.outputs.docs != '' || steps.map-candidates.outputs.docs != ''
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const reportPath = '/tmp/drift-report.md';

            if (!fs.existsSync(reportPath)) {
              console.log('No drift report generated. Skipping.');
              return;
            }

            const report = fs.readFileSync(reportPath, 'utf8');
            if (report.includes('No drift detected')) {
              console.log('No drift detected.');
              return;
            }

            // Check for existing doc-drift PR
            const { data: prs } = await github.rest.pulls.list({
              owner: context.repo.owner,
              repo: context.repo.repo,
              state: 'open',
              head: `${context.repo.owner}:doc-drift/auto`,
            });

            if (prs.length > 0) {
              // Update existing PR body with new report
              await github.rest.pulls.update({
                owner: context.repo.owner,
                repo: context.repo.repo,
                pull_number: prs[0].number,
                body: report,
              });
              console.log(`Updated existing PR #${prs[0].number}`);
            } else {
              // Open new PR
              // (assumes the drift check step already committed fixes
              //  to a doc-drift/auto branch)
              await github.rest.pulls.create({
                owner: context.repo.owner,
                repo: context.repo.repo,
                title: 'docs: fix documentation drift',
                body: report,
                head: 'doc-drift/auto',
                base: 'main',
              });
            }
```

______________________________________________________________________

## Design notes

### Batching

The workflow checks for an existing open `doc-drift/auto` PR before creating
a new one. If multiple merges happen in sequence, the later runs update the
existing PR rather than creating duplicates. This prevents PR spam during
active development periods.

### paths-ignore

Changes to docs/ alone can't cause doc-vs-code drift (they might fix it).
Filtering them out avoids unnecessary runs. If someone edits a doc *and* code
in the same commit, the code paths still trigger the workflow.

### The check script

`scripts/ci/doc_drift_check.py` is a thin wrapper that:

1. Reads the diff, changed files, candidate docs, and mapping.
1. Builds a prompt with all context.
1. Calls the Anthropic API (Claude) to analyze drift.
1. Parses the response into a structured report.
1. If drift is found, applies fixes to doc files and writes the report.

The prompt should instruct Claude to:

- Focus on factual inconsistencies, not style.
- Report confidence levels.
- Produce exact replacement text, not vague suggestions.
- Check cross-doc consistency when multiple docs are candidates.

### Cost control

Each workflow run makes one API call with the diff + candidate docs as context.
For a typical merge touching 5-10 files with 2-3 candidate docs, this is ~10K
tokens input, ~2K tokens output — roughly $0.03-0.05 per run with Sonnet.

For large diffs (50+ files), the check script should truncate to high-confidence
candidates only and note in the report that a full audit may be needed.

### False positive management

The workflow should not auto-merge its own PRs. A human reviews the drift
report and proposed fixes. Over time, if the false positive rate is low,
you can add auto-merge for high-confidence fixes.

If the same drift is flagged repeatedly across runs (doc hasn't been fixed
yet), the PR update mechanism handles this — it updates the existing PR body
rather than creating noise.
