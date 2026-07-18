# Pi Review Host Contract

Claude Code and Codex must route `repo-review-full` and
`repo-review-full-no-comments` through this contract. Do not perform or delegate
review work in the host harness.

1. Select exactly the review skill named by the request.

2. When the request supplies a target, accept it only if it matches
   `[1-9][0-9]*`; reject anything else. Assign the validated digits to `N` and
   append `--target "$N"` as a quoted argument.

3. Run exactly one command:

   ```bash
   agent/_shared/run_pi_review.sh <skill>
   ```

4. Claude Code must set Bash `run_in_background: true`, then call `TaskOutput`
   with a `600000` ms timeout repeatedly until Pi exits. Codex must use one
   foreground blocking call; if the tool yields a shell session, poll that
   session until Pi exits.

5. Never return a monitoring status while Pi is running. After Pi exits, return
   its deliverable verbatim. Do not launch native Claude or Codex review
   workers.
