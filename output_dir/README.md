# Model Outputs

This directory contains model outputs on all evaluation datasets. Each dataset has a directory, which contains the model predictions on the test set under `test/`.

Then, each model configuration has its own subdirectory under `test/`. For example, the predictions of the `codex_NL+SL` model configuration on the `GSM8K` dataset are under `output_dir/GSM8K/test/codex_NL+SL/`.

For all datasets, there are at least the following configurations:
- `codex_NL+SL`: Faithful CoT prompt (with both NL comments and SL programs) and the code-davinci-002 LM, under the greedy decoding strategy.
- `codex_NL+SL_n:40`: Faithful CoT prompt (with both NL comments and SL programs) and the code-davinci-002 LM, under the self-consistency decoding strategy (number of generations=40).

For the datasets used in Section 6 - Analysis (GSM8K, Saycan, Date Understanding, and CLUTRR), there are additionally the following configurations (all of which are variations based on `codex_NL+SL`):
- For Section 6.1 - Ablation Study:
  - `codex_norationale`: we remove the rationales, i.e., everything in the brackets from the NL comments, e.g., `"independent, support: [‘There are 15 trees’]"`.
  - `codex_noNLbutnudge`: we remove all NL comments except the "nudge" line: e.g., "# To answer this question, we write a Python program to answer the following subquestions".
  - `codex_noNL`: we remove all NL comments.
  - `codex_nosolver`: instead of calling the external solver, we add "Answer: {answer}" to the end of every exemplar and let the LM predict the answer itself.
- For Section 6.2 - Robustness to Exemplars:
  - `codex_NL+SL_exemplarset1`: same prompt format as `codex_NL+SL`, but with a different set of exemplars (set1). 
  - `codex_NL+SL_exemplarset2`: same prompt format as `codex_NL+SL`, but with a different set of exemplars (set2). 
  - `codex_NL+SL_exemplarset3`: same prompt format as `codex_NL+SL`, but with a different set of exemplars (set3).
  - `codex_NL+SL_exemplarset4`: same prompt format as `codex_NL+SL`, but with a different set of exemplars (set4).
  - `codex_NL+SL_exemplarset5`: same prompt format as `codex_NL+SL`, but with a different set of exemplars (set5).
- For Appendix C.2 - Effect of LM:
  - `text001` (optional): using `text-davinci-001` as the underlying LM. Due to the prompt length limit, text-davinci-001 only allows us to experiment on CLUTRR.
  - `code001`: using `code-davinci-001`.
  - `text002`: using `text-davinci-002`.
  - `text003`: using `text-davinci-003`.
  - `gpt-3.5-turbo`(new): using `gpt-3.5-turbo`.
  - `gpt4`(new): using `gpt-4`.

- For Appendix C.4 - Robustness to Prompt Phrasing (to be added in the paper): 
  - `codex_prompt_variation1`: Randomly permuting the order of independent subquestions/reasoning steps.
  - `codex_prompt_variation2`: Changing the variable names.
  - `codex_prompt_variation3`: Changing the nudge line (e.g. from “# To answer this question, write a Python program to answer the following subquestions” to “# To solve this question, we answer each of the following subquestions with a Python program”).

In every model configuration folder, the outputs are saved in the `predictions.jsonl` file. Each line in the file is a JSON object with the following fields:

- `id`: the unique ID of the example.
- `answer`: the predicted answer.
- `completion`: the final predicted reasoning chain.
- `completions`: a list of all the predicted reasoning chains.