# Pi Review Host Contract

Claude Code and Codex must route `repo-review-full` and
`repo-review-full-no-comments` through this contract. Do not perform or delegate
review work in the host harness.

1. Set `skill` only when the request names exactly `repo-review-full` or
   `repo-review-full-no-comments`; reject every other value.

2. Unset any inherited `N` before handling the request. When the request supplies
   a target, accept it only if it matches
   `[1-9][0-9]*` in full; do not trim or normalize whitespace and do not extract
   a numeric prefix. Reject anything else without invoking the launcher. Assign
   the validated digits to `N` and append `--target "$N"` as a quoted argument.
   Never omit a supplied valid target or fall back to branch resolution.

3. Build and run exactly one command with a Bash array. Branch explicitly on
   whether the request supplied a target. The branch must use only the `N`
   assigned by step 2, never an inherited environment value:

   ```bash
   if [[ ${N+x} ]]; then
     command=(agent/_shared/run_pi_review.sh "$skill" --target "$N")
   else
     command=(agent/_shared/run_pi_review.sh "$skill")
   fi
   "${command[@]}"
   ```

4. Claude Code must set Bash `run_in_background: true`, then call `TaskOutput`
   with a `600000` ms timeout repeatedly until Pi exits. Codex must use one
   foreground blocking call; if the tool yields a shell session, poll that
   session until Pi exits.

5. Never return a monitoring status while Pi is running. After Pi exits, return
   its deliverable verbatim. Do not launch native Claude or Codex review
   workers.
