# AutoOptimize

This is an experiment to have the LLM optimize the time complexity of Matrix Multiplication for Sparse Matrices. A sparse matrix is a matrix whose at least 70% elements are 0. The goal of this AutoOptimize framework is to find the best algorithm for doing matrix multiplication for two sparse matrices with the best complexity.

## Setup

You are given - 
- An initial program `matrix_mul.py`
    - You should only edit this file
- A set of test cases - `test_cases.txt`

1. Run tag - based on the time and date (e.g. apr06_<time>), create a branch (sparse_matrix_mul/<tag>) for every run. Make sure the tag doesn't exist already
2. Create the branch - `git checkout -b sparse_matrix_mul/<tag>`
3. Read in-scope files: The repo is small. Read the files for full context
    - `matrix_mul.py`: the file you modify. Core logic, algorithm, design.
    - `program.md`: when in doubt to undertand the scope of the optimization problem
4. Initialize `all_results.tsv` with the header row - branch_tag, experiment_id, commit_hash, solutions_passedd, average_latency, observation
5. Confirm and go: Confirm trhe setup looks good

Once you get confirmation, kick off the experimentation

## Experimentation

You launch each experiment using `python3 matrix_mul.py`

### Goal

Get the lowest average latency while making sure that all test cases pass

### Methodology

- Modify `matrix_mul.py` : This is the only file you edit. Everything is fair game. You can go all out. You can use web to find latest advancement in this field. If you believe there are new subproblems that can help solving the sparse matrix multiplication problem, you can also develop solution for those subproblems.
- Do not limit yourself based on what you see on web. Your goal is to create the best algorithm. Such an algorithm might not exist. The only thing you should make sure are (1) all test cases passes corectly and (2) average latency decreases
- Invent and Simplify: Make sure that the algorithmic changes are innovative and simple. Do not try to solve everything at once. All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

The first run: Your very first run should always be to establish the baseline, so you will run the training script as is.

### Output Format

Check the `results.tsv` once the script is completely run. The final row will contain the SUMMARY of the run. You should identify the number of solutions passed, average latency from the SUMMARY row and add a new row in the appropriate format in the `all_results.tsv`. 

Also, the `results.tsv` will contain all the test case latency and success. Based on the code changes and the experiment results, note your observation.

- If you made certain algorithmic change which resulted in latency reduction of most long running experiments, probably it is the correct direction, you should capture it in the observation column. Rememeber the user will review the `all_results.tsv` to decide further steps, so it should be clear about (1) the decision taken, (2) whether it imporoved the overall system (3) the positives (if any) (4) the negatives (if any) (5) next steps

Note that `matrix_mul.py` will always run on the test cases. Do NOT try to hack the test cases to reduce latency. The algorithm should be robust. There is a private test set which will be tested later.

#### Logging results

branch_tag, experiment_id, commit_hash (short), solutions_passed (from SUMMARY), average_latency (from SUMMARY), status (keep/discard/crash) observation (inferred)

**Example:**
branch_tag, experiment_id, commit_hash, solutions_passed, average_latency, status, observation
<date1>_<time1>, 1, a1b2c3d, 50/50, 8006.7623 ms, keep, baseline
<date1>_<time2>, 2, b2c3d4e, 50/50, 7560.6350 ms, keep, <idea1>: modified row handling - improved 40 high latency test cases
<date2>_<time3>, 3, c3d4e5f, 35/50, 2300.54 ms, discard, 15 test cases failed, restarting from previous commit
<date3>_<time4>, 4, d4e5f6g, 50/50, 12007.635 ms, discard, avg latency increased by 50%, <idea2> might be useful later, but restarting form previous commit
<date3>_<time5>, 5, e5f6g7h, None, None, crashed, Issues with the run - Error Trace/Infinite Loop/Out of Memory etc

### The Experiment Loop

The experiment runs on a dedicated branch (e.g. sparse_matrix_mul/<date>_<time>).

**LOOP FOREVER:**

1. Look at the git state: the current branch/commit we're on
2. Modify `matrix_mul.py` with an experimental idea. Based on the `all_metrics.tsv`, decide the idea. You can go all out, do web searches, read papers if needed in the domain, read blogs, experiment something completely innovative. Remember we are optimizing for the best, hence you need to innovate and create ideas which are never explored before.
3. `git commit -m "Appropriate message about the experimental idea`
4. Run the experiment: `python3 matrix_mul.py` (do NOT use tee or let output flood your context)
5. Once the experiment is completed, It should generally take at most 5 minutes, Read out the results from `results.tsv` - The final row contains the summary
6. If the output is empty even after 5 min, the run crashed. Run `tail -n 50 output.log` to read the Python stack trace and attempt a fix. If you can't get things to work after two attempts, give up this idea and restart from the last successful run branch.
7. Record the results in the `all_metrics.tsv`
8. If average_latency improved (lower) while all test_cases passed, you "advance" the branch, keeping the git commit
9. If average_latency is much worse or even one test_case failed, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

Timeout: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!