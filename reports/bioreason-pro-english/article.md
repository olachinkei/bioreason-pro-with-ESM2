# AI Drug Discovery in Early 2026: Reasoning About Protein Function with BioReason-Pro

This report preserves the article's perspective and September 2026 context, with interactive W&B results in place of static training and evaluation screenshots.

In 2026, research incorporating reasoning has begun to emerge in AI-driven drug discovery. One study attracting particular attention is **BioReason-Pro**, which applies a reasoning model to protein function prediction.

[BioReason-Pro](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1) is a multimodal reasoning LLM that combines biological information about protein sequences, structures, domains, and interactions to predict what a protein does. Its defining feature is that it generates a biological reasoning trace leading to its prediction, rather than returning function labels alone. The study reports improvements in prediction in data-scarce settings and in interpretability. BioReason-Pro also comes up frequently in my conversations with drug discovery researchers.

I have been interested in the potential of reasoning since last year. It is particularly interesting to see reasoning in biology attract attention for the interpretability it can offer.

The [official GitHub repository](https://github.com/bowang-lab/BioReason-Pro) is available, but as of September 2026 it focused mainly on inference, with limited information for reproducing training. Some model components also came with restrictions on commercial use, creating an additional barrier to experimentation. To understand BioReason-Pro more deeply and provide a reference for others working on similar research, I built my own training implementation:

[BioReason-Pro with ESM2 — implementation and training code](https://github.com/olachinkei/bioreason-pro-with-ESM2)

There may still be gaps in my review of licensing conditions, as well as differences from the paper's problem setting and implementation. If you notice anything, please share your feedback in the comments on this report.

This article explains how BioReason-Pro works and how it is trained. First, I will outline three broader trends I have observed while researching and experimenting with AI for drug discovery in early 2026.

## Trends in AI for Drug Discovery in Early 2026

Since the arrival of AlphaFold and Transformers, a succession of models and tools has influenced AI for drug discovery. In early 2026, I see three broad directions: conditional generation that takes desired activity or ADMET properties as inputs; a shift from competing on individual model performance toward designing workflows that connect multiple models and tools; and adapting models to a company's domain or a specific task with limited data.

### 1. Making Greater Use of Conditional Generation

Many of the models attracting attention use **conditional generation**: they generate candidates with desired conditions specified in advance. Traditionally, a common approach was to generate a broad pool of candidates, then narrow it using predictors of activity and ADMET properties. More recent work supplies the desired conditions as inputs and aims to generate candidates that are more likely to satisfy them from the outset. Several factors help explain this shift.

**It can be more efficient than filtering afterward.** Generating a large number of candidates and discarding most of them is inefficient when the chemical search space is enormous. Downstream predictors can narrow the pool, but separating generation from evaluation still leads to wasted effort. Conditional generation can focus the search by steering candidates toward the properties of interest from the beginning.

**Models can better learn the relationship between conditions and structures.** More expressive models are making it easier to learn which structures or sequences are likely under particular conditions. Transformers and diffusion models can combine different types of input, including sequences, structures, and expression information, and incorporate those conditions into generation. Pretrained models and reinforcement learning are also making conditional generation more practical when data is limited.

**Drug discovery requires multi-objective optimization.** Activity is only one requirement. Selectivity, toxicity, solubility, and ease of synthesis also matter. Applying filters one after another can make balancing these objectives slow. Conditional generation can incorporate multiple requirements at the start, helping search for candidates that satisfy them together.

Generated candidates still need downstream prediction, ranking, and experimental validation. Even so, the ability to steer generation toward the desired outcome represents an important change in how generative models can be used in drug discovery.

General-purpose generators may still need adaptation to each company's targets, assays, and evaluation criteria.

### 2. Moving from Individual Models to Workflows

A single large model often cannot solve an entire drug discovery problem. Target discovery, literature review, hypothesis generation, evaluation, and experimental design require different knowledge and different inputs and outputs. This has increased interest in designing complete workflows that connect data, models, and evaluation systems.

[Biomni](https://github.com/snap-stanford/Biomni) is a useful example. It attracted attention as a system that uses an LLM as an interface to multiple biological tools and knowledge sources. Similarly, drug discovery increasingly depends on how specialized models are combined and used throughout a workflow.

### 3. Adapting Models to a Company's Domain

Workflows and generative models do not remove the need for strong individual models. Companies differ in the targets, assays, evaluation criteria, and data formats they use, so adapting models to a particular domain can make a substantial difference.

For small molecules, new chemical spaces or company-specific conditions can create a distribution shift between training and deployment. A general-purpose model may then perform poorly without adaptation. The challenge can be greater for biologics such as antibodies and proteins, where data is often scarcer and molecular representations can be harder to learn reliably.

Two approaches are receiving particular attention:

- Multi-stage training: first learn an intermediate task with relatively abundant data, then adapt to the target task.
- Adjusting only part of a model to make domain adaptation efficient with limited data.

Developing methods that are easier to tune is especially important in drug discovery, where models often need continued adaptation.

Against this background, BioReason-Pro, introduced in March 2026, is an interesting example of combining specialized models, using an LLM as the interface for analysis, and bringing reasoning into a biological setting.

## BioReason-Pro

### Overview

Advances in genome analysis have produced an enormous number of protein sequences. The study describes a setting in which UniProt, a major database of protein sequences and functions, contains more than 250 million sequences, while fewer than 0.1% have experimentally established functions. For most proteins, what they actually do remains unclear.

Traditional protein function prediction has broadly followed two approaches. Sequence-similarity methods such as BLAST infer function from similarity to known proteins, but struggle when close matches are unavailable. Machine learning methods predict functional labels, often treating each **Gene Ontology (GO) term** independently. That can produce awkward combinations, such as predicting kinase activity without ATP binding. Both approaches also tend to provide little explanation of why a prediction was made. BioReason-Pro addresses this limitation.

BioReason-Pro combines several models and information sources. ESM3 converts a protein sequence into residue-level representations. GO-GPT, developed in the study, produces preliminary GO predictions. The system also incorporates domain information from InterPro, protein–protein interaction information from STRING, and organism information. A reasoning model based on Qwen3-4B-Thinking integrates this evidence and produces a reasoning trace, a functional summary, and GO annotations.

The reasoning trace expresses how particular observations support proposed functions. The aim is to learn biological reasoning that helps the model handle unfamiliar or poorly characterized proteins, rather than relying only on memorized examples.

For an introduction to reinforcement learning, see [my earlier article](https://note.com/olachin/n/n9706c13c8678) (Japanese).

### What Makes the Study Interesting

Two aspects of model training, and the results they enable, are particularly noteworthy.

<!-- EMBED:architecture -->

**GO-GPT**

The first is GO-GPT, a model dedicated to GO prediction. It generates related labels sequentially, making relationships between functions easier to represent. For example, kinase activity is naturally associated with ATP binding and protein phosphorylation.

GO is a hierarchy organized as a directed acyclic graph (DAG), rather than a flat list of labels. Moving down the hierarchy leads from general concepts such as biological processes to more specific concepts such as metabolic processes, phosphorylation, and protein tyrosine phosphorylation. GO-GPT takes protein representations, the target GO aspect, and organism information as inputs, then generates related terms from general to specific. This design helps preserve hierarchical relationships and connections between categories.

The paper reports that GO-GPT outperformed leading publicly available methods on CAFA5, an international protein function prediction benchmark. Improvements were also reported in weighted F_max, which gives greater weight to informative terms, suggesting more accurate predictions deeper in the GO hierarchy. The study then goes beyond GO-GPT's predictions by adding a reasoning model.

**Supervised Fine-Tuning and Reinforcement Learning**

BioReason-Pro uses Qwen3-4B-Thinking to integrate sequences, GO predictions, domain information, and interaction information, and explain why a function is plausible.

Training has two stages. The first is **supervised fine-tuning (SFT)**. A central challenge is the lack of ground-truth examples of how to reason about protein function. The authors address this by giving GPT-5 InterPro domains, STRING interactions, PDB structures, organism information, and existing GO annotations, and asking it to synthesize expert-like reasoning traces. Roughly 130,000 training examples are generated to teach the reasoning process.

The second stage uses **reinforcement learning (RL)** to improve GO prediction directly. The paper uses GSPO, developed by the Qwen team. GRPO assigns a reward to a complete response while optimizing at the token level; this can become unstable for long reasoning traces. GSPO instead uses a sequence-level treatment to reduce that mismatch. The paper reports that RL improved greedy-decoding weighted F_max from 0.64 to 0.66, shortened reasoning traces by about 60 words on average, and improved a hallucination-avoidance measure. The result was a greater concentration of probability on correct GO terms and somewhat more concise reasoning.

BioReason-Pro exceeded GO-GPT's performance on CAFA5. To me, the numerical improvement alone did not seem particularly large; the more interesting findings emerged in the detailed analysis. The paper reports that predictions and functional summaries remained relatively strong even for proteins with limited similarity to the training data. This suggests usefulness for proteins that are difficult to handle through sequence similarity alone.

Expert evaluation was also encouraging. In the paper's study, BioReason-Pro annotations were preferred to manually curated UniProt descriptions in 79% of cases. This reflects how researchers judged the resulting descriptions, including their explanatory value, beyond the numerical benchmark scores.

The paper presents two illustrative cases:

- **eEFSec (P57772): a specific prediction beyond a close sequence match.** This protein participates in translation involving selenocysteine, a specialized amino acid. Its sequence identity to training examples was 44%. BioReason-Pro identified a role in recognizing selenocysteine-specific tRNA, rather than stopping at generic tRNA binding. It also identified SBP2 as an interaction partner from the C-terminal structure. Structural evidence from PDB 7ZJW supported this interpretation, and the paper's analysis indicated attention to relevant regions. In contrast, GPT-5.2 Thinking High interpreted it as a different class of protein.
- **CFAP61 (Q8NHU2): interpreting a misleading domain.** CFAP61 is a scaffold protein required for sperm flagellum formation. It contains a Rossmann-like fold that resembles an enzyme, leading some databases to assign oxidoreductase activity incorrectly. Its sequence identity to training data was only 29%. BioReason-Pro considered the surrounding domain architecture and family context and interpreted it as a structural component. Key residues needed for enzymatic activity were absent, while part of the fold served as an interface with another protein.

These examples illustrate two complementary possibilities. In eEFSec, the model proposed a specific interaction partner that was not explicitly provided in the input. In CFAP61, it resisted a misleading structural cue. Together, they suggest that BioReason-Pro can help fill gaps in evidence and interpret ambiguous clues, beyond simply locating a similar protein.

### Potential Applications

BioReason-Pro could support several kinds of research:

- Initial triage of newly identified proteins, helping decide which candidates to investigate first.
- Generating experimentally testable hypotheses, such as whether a protein functions through a particular interaction partner.
- Reviewing existing database descriptions and identifying annotations that deserve reconsideration.
- Studying proteins with few close homologs, including metagenomic proteins and disease-associated proteins whose roles remain poorly understood.

Across these uses, the explanation accompanying a prediction is valuable. Researchers can inspect the proposed evidence, assess the result themselves, and decide how it should inform the next experiment.

## Building My Own Implementation

Because the training code was not available when I carried out this work, I implemented the training workflow myself. I followed the paper's general setup, while replacing its ESM3 protein encoder with ESM2 because of ESM3's restrictive licensing conditions. The RL algorithm and training budget also differ, as described below.

Training follows two stages. SFT first teaches the pattern of reading a sequence, reasoning about it, and returning GO terms. RL then refines the model. As described in the original article, the reward is based primarily on agreement between predicted and ground-truth GO terms, with a small bonus for faithfulness to the evidence used in the reasoning.

For this experiment, I used **GRPO rather than the paper's GSPO**. GSPO is designed to improve stability for long reasoning responses, with differences in how optimization corrections are applied. My initial priority was to determine whether a straightforward RL setup would provide a measurable benefit.

### Data

Training uses a dataset derived from CAFA5, containing real protein sequences and target answers with reasoning traces that lead to their functional annotations. During SFT, the complete reasoning trace and answer serve as the demonstration. During RL, the model repeatedly generates predictions for sequences from the same training pool and learns from their rewards.

Final evaluation uses **bioreason_pro_test**, a held-out dataset of **8,630 proteins** whose functional annotations were not available at the time the training data was constructed.

### Model Training

I tracked training and evaluation with Weights & Biases. The panels below embed the original training runs and their recorded metrics.

<!-- EMBED:training -->

During SFT, loss generally decreased as training progressed. Non-empty reasoning was produced in 99.2% of the training run's validation examples, indicating that the model had learned the intended reasoning-before-answer format at the SFT stage.

RL then optimized the reward over **100 steps**. The average reward increased from approximately **0.56 over the first 10 steps** to **0.77 over the last 10 steps**. These are averages across windows, rather than the reward at the final individual step.

I used [W&B Skills](https://github.com/wandb/skills) to retrieve metrics and evaluation results, then had Claude Code analyze problems while I adjusted the training configuration. During development I used validation data, keeping the final holdout separate. See [this article on development with coding agents and W&B](https://note.com/wandb_jp/n/n3a6b92288a9c) (Japanese) for more context.

### Evaluation Results

The final evaluation uses the same type of temporal holdout described in the paper: 8,630 proteins that received new experimental functional annotations between March 2023 and February 2024.

The native panels below compare the original SFT and RL evaluation runs. The associated Weave evaluations expose the per-protein outputs and ground-truth labels across all 8,630 examples: [SFT evaluation](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/weave/evaluations?name=eval-full-holdout-sft-eval-full-holdout-2027) and [RL evaluation](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/weave/evaluations?name=eval-full-holdout-sft_to_grpo-eval-full-holdout-2028).

**A brief explanation of the metric**

The model associates GO predictions with confidence scores. The threshold for accepting a prediction affects the balance between precision, the fraction of predictions that are correct, and recall, the fraction of ground-truth labels recovered. Higher thresholds generally favor precision at the cost of recall.

**F_max** is the maximum F-score, the harmonic mean of precision and recall, obtained by sweeping the threshold between 0 and 1. A score of 1.0 is perfect. Ordinary F_max treats all terms equally, so broad, common labels can contribute substantially without providing much biological detail.

**Weighted F_max** addresses this using **Information Accretion (IA)**, which measures the information a term adds beyond its parents. Informative, specific terms receive more weight than common terms. The metric therefore rewards more specific functional predictions and is central to the paper's evaluation.

Scoring extracts formally written GO identifiers, such as `GO:0003341`, using the pattern `GO:` followed by seven digits. A biologically correct statement does not receive credit as a GO prediction unless it is expressed using the corresponding identifier.

<!-- EMBED:evaluation -->

| Weighted F_max | SFT | RL (GRPO) | Change |
| --- | ---: | ---: | ---: |
| Overall | 0.498 | 0.581 | +0.083 |
| Molecular Function (MF) | 0.527 | 0.583 | +0.056 |
| Biological Process (BP) | 0.355 | 0.479 | +0.123 |
| Cellular Component (CC) | 0.612 | 0.680 | +0.068 |

Values are rounded to three decimals. Changes are calculated from the unrounded run metrics.

The three GO aspects describe different dimensions of function:

- **MF — Molecular Function:** what the molecule does chemically, such as binding ATP or cleaving a protein.
- **BP — Biological Process:** the biological process it participates in, such as cilium movement or regulation of cell division.
- **CC — Cellular Component:** where it is located, such as the nucleus or the mitochondrial inner membrane.

RL outperformed SFT in all three aspects. The largest improvement was in BP (**+0.123**), which also had the lowest SFT score (**0.355**). Biological processes can be difficult to infer directly from domain architecture.

I also compared SFT and RL protein by protein. For this analysis, a sample was classified as correct when its per-sample IA-weighted F1 was at least 0.5. This is a separate sample-level diagnostic, not the aggregate threshold-swept F_max metric above.

| Group | Proteins | Share | Mean SFT F1 | Mean RL F1 |
| --- | ---: | ---: | ---: | ---: |
| Both correct | 1,293 | 15.0% | 0.696 | 0.742 |
| Both incorrect | 6,061 | 70.2% | 0.145 | 0.172 |
| Correct only after RL | 1,053 | 12.2% | 0.235 | 0.669 |
| Correct only with SFT | 223 | 2.6% | 0.603 | 0.375 |

The 12.2% of examples that became correct after RL exceeded the 2.6% that regressed, supporting the benefit of RL in this experiment.

Absolute performance remained below the paper's reported result: the overall weighted F_max of **0.581** is about **0.08 below 0.66**. Possible next steps include increasing the RL budget from 100 steps toward the paper's 1,200 steps, adopting GSPO, and using a larger encoder such as ESM2-3B. These are directions to investigate, rather than established explanations for the gap. Still, the experiment shows that a relatively simple setup can achieve useful performance.

## A Case Improved by RL: A Zebrafish Aftiphilin-like Protein (A0A8M9PEU8)

Consider A0A8M9PEU8, a zebrafish aftiphilin-like protein. Interestingly, the two models begin with almost identical reasoning:

> I begin with the InterPro architecture. The sequence is assigned to IPR046359 (Aftiphilin-like family) spanning residues 1–696... Embedded within this, residues 517–585 form IPR029205 (Aftiphilin, clathrin-binding box).

Their answers then diverge. SFT follows the conspicuous clathrin-binding-box cue:

| GO aspect | SFT predictions |
| --- | --- |
| Molecular Function | `GO:0030276` clathrin heavy chain binding; `GO:0035615` clathrin adaptor activity; `GO:0005515` protein binding |
| Biological Process | `GO:0072583` clathrin-mediated endocytosis; `GO:0006897` endocytosis; `GO:0046907` intracellular transport |
| Cellular Component | `GO:0005737` cytoplasm; `GO:0005905` clathrin-coated pit; `GO:0030136` clathrin-coated vesicle |

These nine terms form a coherent, confident account centered on clathrin and endocytosis. The ground-truth labels, however, center on **cilium movement (`GO:0003341`)** and **microtubule-based movement (`GO:0007018`)**. SFT captured little beyond cytoplasm, producing an IA-weighted **F1 of 0.038**.

The RL model starts from the same evidence but produces a more restrained answer:

| GO aspect | RL predictions |
| --- | --- |
| Molecular Function | `GO:0005515` protein binding |
| Biological Process | `GO:0007018` microtubule-based movement; `GO:0003341` cilium movement |
| Cellular Component | `GO:0005737` cytoplasm |

For molecular function, it stops at the broader protein-binding label instead of committing to clathrin binding. Its explanation then connects a proposed role linking clathrin-coated vesicles and microtubule motors to the functional outcome of cilium movement. With four terms, its IA-weighted **F1 is 0.960**.

The same pattern appears in another protein with the same domain architecture, **A0A8M1PV90**. SFT again predicts clathrin-related functions and scores **0.041**, while RL predicts protein binding and cilium movement and scores **0.960**. SFT repeats the same mistake; RL avoids it in both examples.

This resembles the paper's CFAP61 case, where the model interpreted an enzyme-like fold in the context of a scaffold protein. The distinction is between transferring a conspicuous domain name directly into a functional label and reasoning about its wider functional consequences. Here, the measured improvement is agreement with the held-out annotations; the proposed mechanism remains something researchers can inspect and test.

<!-- EMBED:traces -->

The embedded table contains the full generated outputs and recorded scores from the four original Weave calls, captured for this report. Source calls: **A0A8M9PEU8** [SFT](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/weave/calls/01a0d5fa-9b33-7ba8-a39f-17bf79ad15dd) / [RL](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/weave/calls/01a0d4e1-48f7-768a-aca7-f849e61ce2f1); **A0A8M1PV90** [SFT](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/weave/calls/01a0d5fa-68e8-736d-a177-3ebaae3ea3c4) / [RL](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/weave/calls/01a0d4e1-1700-7412-b7c1-4193aea12cac).

## Reflections and Next Steps

Although this implementation does not reach the paper's accuracy, it shows that ESM2 can support meaningful performance in place of ESM3. A larger RL budget and adoption of GSPO are promising next experiments.

The appeal of BioReason-Pro also lies in its interpretability. Researchers can read the proposed reasoning and decide which claims to pursue or discard. This is particularly relevant to tasks such as initial protein triage and generating hypotheses for experiments, where a person is expected to examine the output critically.

RL has many configuration choices and can require substantial compute time, but implementation is becoming more accessible. In AI-driven drug discovery, where explanations matter, I hope it becomes a practical option for more researchers. I hope this implementation and report provide a useful starting point.
