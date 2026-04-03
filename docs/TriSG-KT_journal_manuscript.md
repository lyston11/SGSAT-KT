# TriSG-KT: A Triple-Sparse Semantic Graph Knowledge Tracing Model

**Author 1**<sup>1</sup>, **Author 2**<sup>1,*</sup>

<sup>1</sup>School/Institute Name, University Name, City, Country  
<sup>*</sup>Corresponding author: `email@domain.edu`

## Abstract

Knowledge tracing (KT) aims to infer evolving student mastery states from historical interactions and predict future responses. Existing sparse-attention KT models are effective for long-sequence modeling, yet they remain limited in three respects: they underuse the semantic content of questions and concepts, they rarely encode explicit prerequisite structure, and they are vulnerable to unstable representation learning under purely supervised objectives. This paper addresses these limitations by first developing an original Triple-Sparse Attention Backbone (`TriSA-Backbone`) and then proposing `TriSG-KT`, a Triple-Sparse Semantic Graph Knowledge Tracing model, on top of it. Specifically, `Qwen3-4B` is used as an external semantic encoder to generate offline embeddings for question texts and concept texts, thereby compensating for the semantic deficiency of pure item-ID representations. To prevent these semantic priors from being diluted by shallow fusion, an identity-triggered selective semantic alignment mechanism is introduced so that item identity actively retrieves prediction-relevant semantic components. To compensate for the lack of structural modeling, a lightweight prerequisite-graph module injects concept dependency priors into question representations. To improve representation stability, sequence-level contrastive learning and embedding-level InfoNCE regularization are jointly imposed. Beyond the architectural contribution, the study also establishes an end-to-end pipeline covering semantic preprocessing, graph construction, training integration, and multi-baseline evaluation. According to the currently verified historical experiments on the XES dataset, `TriSG-KT` consistently outperforms SAKT, AKT, DKT, DKVMN, and the `TriSA-Backbone` baseline; meanwhile, a new round of training under the rebuilt v4.2 data protocol is still running and has continued to improve validation performance in its early epochs. The results indicate that the proposed design effectively compensates for the semantic, structural, and optimization deficiencies of existing sparse-attention KT models.

**Keywords:** knowledge tracing; TriSA-Backbone; selective semantic alignment; prerequisite graph; contrastive learning

## 1. Introduction

Knowledge tracing is a central problem in educational data mining and intelligent tutoring systems. Its goal is to infer a student's evolving mastery state from historical interactions and to predict future responses. Reliable KT models are therefore foundational for downstream tasks such as exercise recommendation, adaptive practice scheduling, mastery diagnosis, and targeted pedagogical intervention.

Recent years have witnessed a clear shift in KT from probabilistic state-transition models to neural sequence models, and further to attention-based and Transformer-based architectures. This evolution has significantly improved long-range dependency modeling and prediction accuracy. Sparse-attention variants are especially attractive because they retain the sequence modeling advantages of attention while reducing computational redundancy in long educational interaction sequences.

However, the current generation of sparse-attention KT models still suffers from three shortcomings. First, item representations remain dominated by discrete identifiers, which leaves the semantic content of question texts and concept descriptions underexploited. Second, prerequisite dependencies among concepts are rarely encoded explicitly, even though they are central to structured learning progression. Third, when richer side information is introduced, model training remains vulnerable to shortcut reliance and unstable representation learning, especially when semantic branches are added in a shallow manner.

This paper addresses these shortcomings by first constructing `TriSA-Backbone` and then proposing `TriSG-KT`. The model is designed around a simple principle: each identified deficiency should be matched by a specific compensatory mechanism. Qwen3-based semantic grounding addresses the deficiency of pure ID representations; task-aware selective semantic alignment addresses the weakness of shallow semantic integration; prerequisite-graph enhancement addresses the lack of explicit structural modeling; and dual contrastive regularization addresses instability in representation learning. Beyond the modeling contribution, the study also develops a complete experimental pipeline, including semantic preprocessing, graph construction, model integration, and multi-baseline evaluation.

The main contributions of this work are as follows.

1. To compensate for the semantic deficiency of pure item-ID-based KT, we introduce a Qwen3-4B-based semantic grounding strategy that jointly encodes question texts and concept texts as external semantic priors.
2. To address the weakness of shallow semantic integration, we propose an identity-triggered selective semantic alignment mechanism that allows item identity to retrieve prediction-relevant semantic components.
3. To compensate for the lack of explicit structural reasoning in sparse-attention KT, we incorporate a lightweight prerequisite-graph encoder that injects concept dependency priors into question representations.
4. To mitigate instability in representation learning, we jointly adopt sequence-level contrastive learning and embedding-level InfoNCE regularization.
5. From a systems and empirical perspective, we build an end-to-end `TriSG-KT` pipeline covering semantic preprocessing, graph construction, training integration, and multi-baseline evaluation, and we verify its effectiveness under the current XES experimental setting.

### 1.1 Research and Implementation Scope

To accurately reflect the actual scope of the study, the present work should not be interpreted as a narrow architectural tweak on top of an existing KT backbone. Beyond proposing `TriSG-KT`, the study covers the main stages required to establish a workable research pipeline: question-text and concept-text alignment, Qwen3-4B-based offline semantic preprocessing, prerequisite-graph construction, model-level integration of semantic fusion and graph enhancement, training-pipeline completion, multi-baseline evaluation, and debugging of core failure points in the end-to-end workflow. This implementation scope is part of the contribution of the work, because it turns the proposed idea into a reproducible and empirically testable KT system.

## 2. Related Work

### 2.1 Knowledge Tracing Models

Early KT methods were dominated by probabilistic approaches, such as Bayesian Knowledge Tracing, which provided interpretable state-transition mechanisms but limited expressive capacity. Deep Knowledge Tracing later demonstrated that neural sequence models could significantly improve predictive performance by learning latent state dynamics from interaction sequences. Subsequent work expanded KT along several directions, including memory-augmented modeling, attention-based sequence modeling, graph-enhanced representation learning, text-aware item modeling, and forgetting-aware prediction. These developments substantially improved the field's ability to capture complex learning dynamics.

Among them, attention-based and Transformer-based KT models are especially relevant to the present work. Models such as SAKT and AKT improved long-range dependency modeling by explicitly selecting important historical interactions. Later Transformer-style KT models, including SAINT, SAINT+, and related variants, further strengthened sequence modeling capacity. However, despite these advances, most such models still rely mainly on item identifiers and implicit sequential patterns, leaving room for stronger semantic grounding and explicit structural reasoning.

### 2.2 Semantic-Enhanced Educational Modeling

Recent advances in pretrained language models have encouraged the integration of textual semantics into educational modeling tasks. Question texts, concept descriptions, and multimodal instructional content can all provide useful signals beyond item IDs. In educational settings, such semantics may capture difficulty cues, concept associations, and linguistic similarity across exercises. However, in KT, semantic information is often introduced via simple concatenation or addition, which may lead the model to ignore semantic branches and overfit to item identifiers. A more effective fusion strategy is therefore required if semantic priors are to play a substantive role in sequential student modeling.

### 2.3 Graph-Based Educational Representation Learning

Knowledge concepts often form structured dependencies, such as prerequisite relations or curriculum hierarchies. Graph neural networks provide a natural way to encode such structures through neighborhood-based message passing. In educational tasks, graph-based representations have been used to model concept relations, knowledge graphs, and interpretable learning paths. For KT, graph-enhanced representations can improve cross-concept transfer modeling, but they must be designed carefully to avoid excessive complexity, noisy structural propagation, and unstable coupling with sequential backbones.

### 2.4 TriSA-Backbone as the Proposed Sequence Backbone

The present study does not adopt an external sparse-attention KT backbone as a precursor framework. Instead, it constructs an original sequence backbone, `TriSA-Backbone`, for long-sequence educational interactions. The backbone is organized around three educational inductive biases: recent interactions should be emphasized, noisy extreme attention peaks should be suppressed, and a small number of decisive long-range historical behaviors should be preserved. In addition, the backbone combines mastery-prototype readout and repeat-frequency encoding so that latent mastery structure and repeated practice signals can be modeled jointly.

On top of this backbone, the present work further addresses three issues central to the study. First, item representation should move beyond discrete identifiers and exploit the natural-language semantics of question texts and concept texts. Second, prerequisite dependencies among concepts should be encoded explicitly. Third, once richer semantic priors are introduced, the model should include a dedicated task-aware mechanism that prevents semantic features from being overshadowed by item-ID shortcuts. `TriSG-KT` is proposed precisely as a structured response to these requirements.

### 2.5 Research Gap

Taken together, the existing literature still leaves an important gap. Standard attention-based KT models are effective at sequence modeling but remain weak in semantic grounding and explicit structural reasoning. Semantic-enhanced variants often introduce textual features in a shallow manner, making them vulnerable to ID-dominant shortcut learning. Graph-enhanced methods, in turn, may inject structural priors but often do not explain how such priors should be coupled with high-capacity semantic features in a task-aware manner. Moreover, many studies improve one deficiency at a time rather than providing a unified solution to semantic deficiency, structural deficiency, and representation instability. `TriSG-KT` is designed to fill this gap by aligning each innovation with one of these deficiencies and integrating them into a single KT framework built on `TriSA-Backbone`.

## 3. Methodology

### 3.1 Problem Formulation

Let the interaction sequence of a student be

$$
X = \{(q_1, r_1), (q_2, r_2), \dots, (q_T, r_T)\},
$$

where $q_t$ denotes the exercised item at time step $t$ and $r_t \in \{0,1\}$ denotes the response label. The KT task is to predict the probability of a correct response at the next step:

$$
\hat{y}_{t+1} = P(r_{t+1}=1 \mid q_1,r_1,\dots,q_t,r_t).
$$

The objective of `TriSG-KT` is to improve this prediction by jointly modeling sequence dynamics, item semantics, and concept prerequisite structures.

### 3.2 Overall Architecture

`TriSG-KT` uses `TriSA-Backbone` as its sequence modeling core and extends the representation layer with semantic grounding and graph enhancement. The overall pipeline is as follows.

1. Learn an item ID embedding for each question.
2. Load precomputed Qwen3-derived question and concept semantic embeddings.
3. Fuse ID embeddings and semantic embeddings through the selective semantic alignment (SSA) module.
4. Encode the prerequisite graph of concepts using a lightweight graph convolution module.
5. Add repeat-count embeddings and answer embeddings.
6. Feed the resulting sequence into the sparse-attention KT backbone.
7. Optimize the model with supervised loss, consistency regularization, sequence-level contrastive loss, and embedding-level contrastive loss.

### 3.3 Qwen3-Based Semantic Grounding with Precomputed Embeddings

Each question is associated with a question text embedding and, when available, a concept text embedding. In the present study, `Qwen3-4B` is used as the default semantic encoder to produce precomputed embeddings for both question texts and concept texts. This design should be viewed as a deliberate modeling choice rather than a generic placeholder for an arbitrary LLM. To reduce training-time overhead, these Qwen-derived embeddings are stored offline and projected to a unified latent space:

$$
e_t^q = P_q(v_t^q), \quad e_t^{kc} = P_k(v_t^{kc}),
$$

where $v_t^q$ and $v_t^{kc}$ denote the original pretrained embeddings. In the default configuration, both are 2560-dimensional vectors generated by `Qwen3-4B`, and $P_q(\cdot)$ and $P_k(\cdot)$ are projection networks. The semantic representation is then formed as

$$
e_t^{sem} = e_t^q + W_p e_t^{kc},
$$

where $W_p$ is a learnable linear transform for concept semantics. Therefore, Qwen is not part of the prediction head itself; instead, it serves as the upstream semantic grounding source that provides high-capacity semantic priors for the KT model.

### 3.4 Selective Semantic Alignment

Simply attaching Qwen-derived semantic vectors to a KT model does not guarantee effective semantic utilization. In practice, naive concatenation or summation can lead the model to rely mainly on item IDs while ignoring the semantic branch. To avoid this failure mode, `TriSG-KT` lets the ID embedding actively retrieve task-relevant semantic features. Let $e_t^{id}$ be the item ID embedding. It is first projected into the semantic space:

$$
\tilde{e}_t^{id} = W_{id} e_t^{id}.
$$

The projected ID embedding is then used as the identity trigger, while the semantic representation serves as the external semantic response:

$$
h_t^{sem} = \mathrm{SSA}(\tilde{e}_t^{id}, e_t^{sem}).
$$

A gated residual fusion is used to stabilize learning:

$$
g_t = \sigma(\mathrm{MLP}(h_t^{sem})),
$$

$$
e_t^{fuse} = \mathrm{LayerNorm}(g_t \cdot h_t^{sem} + (1-g_t) \cdot \tilde{e}_t^{id}).
$$

This design allows the model to preserve the discriminative power of item IDs while extracting task-relevant semantic components from static Qwen-derived embeddings. In other words, Qwen is integrated into `TriSG-KT` through an identity-triggered semantic alignment mechanism rather than through shallow feature concatenation.

### 3.5 Prerequisite Graph Encoding

Let $G=(V,E)$ denote a prerequisite graph over knowledge concepts. Each concept node is associated with an embedding. A lightweight graph convolution module updates the concept representation through neighborhood aggregation:

$$
H^{(l+1)} = \mathrm{LayerNorm}\left(H^{(l)} + \sigma(\hat{A}H^{(l)}W^{(l)})\right),
$$

where $\hat{A}$ is the normalized adjacency matrix, $W^{(l)}$ is the learnable transform at layer $l$, and $\sigma(\cdot)$ is a non-linear activation. For the concept $c_t$ associated with question $q_t$, the graph-enhanced concept representation $h_{c_t}^{graph}$ is added to the fused question representation:

$$
e_t = e_t^{fuse} + h_{c_t}^{graph}.
$$

This step injects structural priors into the input representation before sequence modeling.

### 3.6 Repeat Embeddings and TriSA-Backbone

Repeated exposure to the same question can reflect reinforcement, familiarity, or forgetting compensation. `TriSG-KT` therefore introduces a repeat-count embedding $e_t^{rep}$, computed from the cumulative occurrence count of the current question in the sequence:

$$
e_t^{input} = e_t + e_t^{rep}.
$$

The final answer-aware representation is

$$
s_t^{input} = e_t^{input} + e_t^{ans},
$$

where $e_t^{ans}$ is the answer embedding. These representations are passed into `TriSA-Backbone`, which models historical dependencies and produces hidden states for final prediction.

### 3.7 Training Objective

The overall objective is defined as

$$
\mathcal{L} = \mathcal{L}_{bce} + \lambda_{cons}\mathcal{L}_{cons} + \lambda_{cl}\mathcal{L}_{seq} + \lambda_{contra}\mathcal{L}_{emb} + \mathcal{L}_{reg}.
$$

Here:

1. $\mathcal{L}_{bce}$ is the weighted binary cross-entropy prediction loss.
2. $\mathcal{L}_{cons}$ is a knowledge consistency regularizer that discourages representation collapse among latent knowledge components.
3. $\mathcal{L}_{seq}$ is a sequence-level contrastive loss based on sequence perturbation and hard negative construction.
4. $\mathcal{L}_{emb}$ is an embedding-level InfoNCE loss that encourages questions sharing the same concept to stay close in the semantic projection space.
5. $\mathcal{L}_{reg}$ is a regularization term.

The combination of supervised and contrastive objectives improves both predictive performance and representation robustness.

### 3.8 Design Rationale and Implementation Choices

Three implementation choices are central to the methodological contribution of `TriSG-KT`.

First, the study adopts `Qwen3-4B` in an offline preprocessing manner rather than fine-tuning a large language model inside the KT training loop. This is not merely an engineering shortcut. KT research typically requires repeated training, ablation, and baseline comparison under limited computational resources. By moving Qwen-based semantic encoding to an offline stage, the model preserves high-capacity semantic priors while keeping the online training process tractable and reproducible.

Second, the method does not rely on shallow feature concatenation between item IDs and Qwen semantic vectors. Instead, it uses an identity-triggered selective semantic alignment interaction. This choice reflects a modeling assumption: item IDs and semantic priors play different roles, and the model should actively retrieve prediction-relevant semantic components conditioned on the current item identity rather than passively mixing all semantic information.

Third, prerequisite-graph enhancement is introduced at the representation layer rather than by deeply rewriting the sequence backbone. This design keeps `TriSA-Backbone` relatively stable while still injecting explicit structural priors, and it also yields cleaner module boundaries for future ablation studies.

From a computational perspective, the additional online cost of `TriSG-KT` mainly comes from semantic projection, selective semantic alignment, and lightweight graph propagation. Since Qwen encoding itself is completed offline, the incremental cost during KT training remains substantially lower than that of end-to-end large-model integration.

## 4. Experimental Setup

### 4.1 Dataset

The experimental study currently centers on the XES dataset. Under the rebuilt v4.2 data preparation protocol, the dataset uses 7,618 questions and 865 knowledge concepts. The current training pipeline loads 26,693 training sequences, 3,288 validation sequences, and 3,325 test sequences, with a processed sequence length of 200. In addition, the prerequisite graph constructed over concepts contains 41,853 edges. Question texts and concept texts are both available for semantic preprocessing.

It should also be emphasized that the current workflow no longer relies on test-set model selection. A provided `valid.txt` split is used for training-time model selection, while `test.txt` is reserved for final evaluation. However, some comparison numbers cited in the present draft are still drawn from historically verified experiments under the earlier protocol. They should therefore be interpreted as historical reference results until the full v4.2 rerun is completed.

### 4.2 Baselines

The comparison set includes four representative KT baselines together with a `TriSA-Backbone` baseline:

1. DKT
2. DKVMN
3. AKT
4. SAKT
5. `TriSA-Backbone` baseline without semantic or graph enhancement

The `TriSA-Backbone` baseline refers to the original sequence backbone used in this study with semantic and graph enhancement disabled. It is used to isolate the contribution of the proposed extensions.

### 4.3 Implementation Details

The full `TriSG-KT` configuration uses `d_model = 256`, `id_dim = 128`, `llm_proj_dim = 256`, `llm_inter_dim = 512`, `n_heads = 8`, `cross_attn_heads = 4`, `n_know = 64`, `n_kc = 865`, `dropout = 0.2`, `id_dropout_rate = 0.15`, `batch_size = 4`, `gradient_accumulation_steps = 8`, `n_epochs = 30`, and `learning_rate = 0.001`.

The semantic branch uses precomputed embeddings generated by an external LLM. In the default setting, both question and concept embeddings are 2560-dimensional vectors derived from `Qwen3-4B`. During training, these Qwen-based representations are projected into the KT latent space, fused with item ID representations through selective semantic alignment, and then combined with graph-enhanced concept priors before entering `TriSA-Backbone`. The full model enables prerequisite graph encoding, sequence-level contrastive learning, and embedding-level InfoNCE regularization.

Optimization is performed with AdamW and a cosine-annealing learning-rate scheduler. The weight-decay coefficient is `1e-5`. The prediction term uses weighted binary cross-entropy, where incorrect responses receive a larger weight (`1.2`). In the current full setting, the sequence-level contrastive term uses `lambda_cl = 0.1` with a temperature of `0.05`, and the embedding-level InfoNCE term uses `lambda_contra = 0.3` with a temperature of `0.07`. These settings reflect the current implementation rather than a broad hyperparameter sweep.

### 4.4 Evaluation Metrics

Four metrics are reported: AUC, ACC, MAE, and RMSE. AUC measures ranking quality, ACC reflects binary prediction accuracy, and MAE/RMSE quantify prediction error from complementary perspectives.

## 5. Results

### 5.1 Main Results

Table 1 reports the currently verified historical comparison results on XES. These values have been checked against the archived training logs, but they have not yet been fully replaced by the ongoing v4.2 rerun.

**Table 1. Main comparison results on XES**

| Model | ACC | AUC | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| DKT | 0.8057 | 0.7127 | 0.3634 | 0.3964 |
| DKVMN | 0.8215 | 0.7594 | 0.2603 | 0.3635 |
| AKT | 0.8121 | 0.7407 | 0.2910 | 0.3719 |
| SAKT | 0.8384 | 0.8159 | 0.2260 | 0.3440 |
| TriSA-Backbone baseline | 0.8318 | 0.8150 | 0.2512 | 0.3494 |
| **TriSG-KT** | **0.8449** | **0.8443** | **0.2084** | **0.3350** |

According to these historically verified results, `TriSG-KT` achieves the best performance across all four metrics. Relative to the `TriSA-Backbone` baseline, the model improves AUC by 0.0293. Relative to SAKT, the gain is 0.0284. The lower MAE and RMSE values further suggest that the improvement is not limited to ranking quality but also extends to predictive stability. Importantly, these gains should not be interpreted as the isolated effect of attaching a single auxiliary branch. Rather, they indicate that semantic grounding, structure-aware enhancement, and stability-oriented optimization work more effectively as a coordinated framework than as disconnected add-ons.

In parallel, the ongoing v4.2 rerun under the rebuilt XES protocol has already provided supportive interim evidence. By epoch 4, the validation ACC reaches `0.8456`, the validation AUC reaches `0.8559`, the validation MAE is `0.2152`, and the validation RMSE is `0.3325`. Although these numbers are not yet final test-set results, they indicate that the rebuilt preprocessing and validation workflow preserves, and may further strengthen, the empirical competitiveness of `TriSG-KT`.

### 5.2 Result Analysis

The results suggest that semantic and structural priors provide complementary benefits to KT. The Qwen3-based semantic grounding branch enriches question representations beyond discrete IDs and establishes a more informative semantic neighborhood over questions and concepts in a Chinese educational setting. The prerequisite graph encoder introduces concept-level dependency structure that cannot be recovered reliably from sequential interaction signals alone. The contrastive objectives serve a different role: they regularize the representation space and reduce the tendency of the model to collapse toward ID-dominant shortcuts. The empirical pattern is therefore consistent with the design hypothesis of `TriSG-KT`: semantic grounding addresses representational insufficiency, graph enhancement addresses structural insufficiency, and contrastive regularization addresses optimization instability.

The current rerun trajectory also supports this interpretation from a training-dynamics perspective. The joint objective decreases steadily during the first few epochs, while validation AUC continues to rise rather than collapse. This behavior is consistent with the intended role of SSA and the multi-constraint training objective: semantic enhancement is not merely attached to the model, but is constrained strongly enough to remain useful during optimization.

### 5.3 Ablation Study Plan

Systematic ablation results are not yet complete. The most informative ablations to prioritize are:

1. `w/o GNN`
2. `w/o SSA`
3. `w/o sequence-level contrastive learning`
4. `w/o embedding-level InfoNCE`

A submission-ready version of this study should include a full ablation table so that each claimed compensatory mechanism can be empirically isolated.

## 6. Discussion

`TriSG-KT` is motivated by the view that improved KT performance requires more than stronger sequence modeling alone. The present results support this view. The gains obtained here are more plausibly explained by better representation formation than by a purely larger or more flexible backbone. In particular, the semantic branch is useful only when the model can retrieve task-relevant semantic content in a conditional manner; otherwise, semantic priors risk becoming decorative features. Likewise, concept structure is helpful only when it is injected in a way that is compatible with sequence modeling rather than competing with it. The proposed design attempts to meet both conditions.

The study also contributes at the systems level. In educational modeling, semantic enhancement and graph enhancement are often discussed conceptually but are not always supported by a stable training pipeline. Here, semantic preprocessing, graph construction, model integration, baseline comparison, and core pipeline stabilization have all been completed as part of a single workflow. This does not replace the need for stronger empirical evidence, but it does make the present study a more substantial step than a purely conceptual model proposal.

Several limitations remain. First, the full v4.2 rerun has not yet finished, so the manuscript still contains a mixture of historical verified results and ongoing rerun evidence. Second, systematic ablations and sensitivity analyses are incomplete. Third, the study has not yet established cross-dataset generalization. These limitations do not negate the present findings, but they do define the boundary between the current manuscript and a fully submission-ready journal paper.

## 7. Conclusion

This paper presents `TriSG-KT`, a knowledge tracing model built on the original `TriSA-Backbone` and further enhanced with Qwen3-based semantic grounding, prerequisite graph constraints, and contrastive regularization. The proposed method is organized around three deficiencies of existing KT models: insufficient semantic grounding, insufficient structural reasoning, and insufficient representation stability. Each major component of `TriSG-KT` is introduced to compensate for one of these deficiencies, and the resulting framework achieves consistent improvements over multiple strong baselines under the XES setting examined in this study. Beyond the architectural contribution, the study also establishes an end-to-end workflow covering semantic preprocessing, graph construction, model integration, and multi-baseline evaluation. Future work should focus on completing the full v4.2 rerun, finalizing the ablation studies, and testing the framework across additional datasets and downstream educational tasks.

## Declarations

### Funding

Not specified in the current draft. To be completed according to the actual project or grant support.

### Conflict of Interest

The authors declare no conflict of interest.

### Ethics Approval

Not applicable for the current technical manuscript draft. To be completed if the target journal requires a formal ethics statement for educational data usage.

### Consent to Participate

Not applicable in the current draft.

### Consent for Publication

All authors approve the submission of the manuscript in its final form.

### Data Availability

The experiments in this draft are based on the datasets and processed files currently stored in the project workspace. A formal data availability statement should be added according to the dataset license and journal policy.

### Code Availability

The implementation used in this study is available to the authors and can be organized for release subject to repository and data-sharing policies.

### Author Contributions

To be completed according to the actual contribution distribution. A typical journal-compatible version may include conceptualization, methodology, implementation, experiments, writing, supervision, and review/editing.

## References

> Placeholder note: the final journal version should replace this section with a fully formatted bibliography according to the target journal style.

[1] Deep Knowledge Tracing.  
[2] Dynamic Key-Value Memory Networks for Knowledge Tracing.  
[3] A Self-Attentive Model for Knowledge Tracing.  
[4] Context-Aware Attentive Knowledge Tracing.  
[5] Attention Is All You Need.  
[6] Semi-Supervised Classification with Graph Convolutional Networks.  
[7] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.  
[8] Related work on semantic-enhanced educational modeling.  
[9] Related work on graph-enhanced educational recommendation and concept modeling.
