
# Part 3 - Open-Answer Questions

#### Q1: How is BERTScore calculated? Read the first three paragraphs in Section 3 -- called "Token representation", "Similarity Measure", and "BERTScore" -- in [this paper](https://arxiv.org/pdf/1904.09675.pdf) and give a technical description of how the BERTScore precision/recall/f1 is calculated in ~6 sentences. You do not need to describe anything outside the scope of these specific paragraphs.


To calculate BERTScore, the process can be summarized in the following steps:

*1. Use a contextual embedding model, such as BERT, to transform the tokens of the reference and candidate sentences into vector representations.*
*2. Calculate cosine similarity between vectors of tokens from the reference and candidate sentences by taking the dot product of the pre-normalized vectors, where pre-normalization ensures vectors have a magnitude of 1.*
*3. For precision, match each token in the candidate sentence to the highest cosine similarity score with tokens in the reference sentence, sum these scores, and divide by the total number of tokens in the candidate sentence.*
*4. For recall, match each token in the reference sentence to the highest cosine similarity score with tokens in the candidate sentence, sum these scores, and divide by the total number of tokens in the reference sentence.*
*5. Calculate the F1 measure of BERTScore by taking the harmonic mean of precision and recall, which involves multiplying their product by 2 and then dividing by the sum of precision and recall.*
*6. This approach employs greedy matching to ensure each token is matched to the most similar token in the other sentence, maximizing the similarity score.*

#### Q2: How is COMET trained and calculated? Read Section 2.4 -- "Translation Ranking Model" -- in [this paper](https://arxiv.org/pdf/2009.09025.pdf) and give a technical description in ~6 sentences.

*Training COMET:*

*Input Data Structure:* The model takes a 4-segment tuple as input: source sentence (s), reference translation (r), a better-ranked hypothesis (h+), and a worse-ranked hypothesis (h−). These segments are encoded independently through a cross-lingual encoder, followed by a pooling layer to obtain sentence embeddings for each segment.
*Loss Calculation:* The training involves computing a triplet margin loss for both the source and the reference. This loss, L(s,h+,h−) for the source and L(r,h+,h−) for the reference, is calculated by measuring the Euclidean distance between the source or reference and the better hypothesis, subtracting the distance between the source or reference and the worse hypothesis, adding a margin, and taking the maximum of this result and zero. The final loss is the sum of the loss from the source and the reference.
*Optimization:* The model optimizes the embedding space to minimize the distance between the “better” hypothesis and the “anchors” (source and reference), using the defined loss functions.

*Inference with COMET:*

*Single Hypothesis Evaluation:* At inference, the model processes a triplet (s, ĥ, r) with only one hypothesis. It calculates the harmonic mean between the Euclidean distances to the source (d(s, ĥ)) and to the reference (d(r, ĥ)).
*Similarity Scoring: *The resulting harmonic mean distance is transformed into a similarity score that is bounded between 0 and 1. This is achieved by adding 1 to the harmonic mean and taking the reciprocal of this sum.

This approach allows COMET to quantitatively evaluate the quality of a given translation hypothesis relative to a source sentence and its reference translation, using embeddings to capture the semantic distance in a cross-lingual context.

#### Q3: Given your understanding of BLEU, BERTScore and COMET, how would you interpret the Kendall's Tau correlation results? Which ones are the least and most correlated? What is your hypothesis regarding the reasons behind the lowest correlation for one metric and the highest correlation in the other?

Interpreting Kendall's Tau correlation results for BLEU, BERTScore, and COMET metrics yields insightful observations on how each metric aligns with human judgment in translation tasks:

*COMET shows the highest correlation with human evaluations, achieving a score of approximately 0.289.*This is likely due to COMET's training on a large corpus of translation pairs, enabling it to understand the semantic nuances of sentences deeply. COMET evaluates translations by considering semantic accuracy and fluency, making it highly correlated with human judgment.

*BLEU, with the lowest correlation score around 0.152, is the least aligned with human evaluations.* BLEU's approach of assessing translations based on n-gram overlap focuses on surface-level lexical similarities. This method may miss crucial aspects like semantic coherence and context, which are vital for accurate translation evaluation. Hence, BLEU's lower correlation can be attributed to its inability to capture the deeper linguistic qualities that human evaluators consider.

*BERTScore, positioned between COMET and BLEU, incorporates contextual and semantic information to some extent, thanks to the use of pre-trained models like BERT.* This allows BERTScore to surpass BLEU in correlation with human judgment. However, its direct training not on translation pairs but on general language understanding tasks might explain why its correlation score, though better than BLEU's, still lags behind COMET's.

*Hypothesis Behind Correlation Scores: *The highest correlation of COMET suggests that metrics trained specifically on translation tasks and capable of understanding semantic nuances align better with human judgments. COMET's approach to capturing semantic accuracy and fluency resonates more with how humans evaluate translations. On the other hand, BLEU's lowest correlation underscores the limitations of relying solely on n-gram overlap for quality evaluation, highlighting the need for metrics that can assess deeper semantic and contextual aspects of language.

In summary, the differences in correlation with human judgment among these metrics underscore the importance of semantic understanding and context in translation evaluation. Metrics like COMET, which are designed and trained to capture these aspects, naturally exhibit a higher correlation with human evaluations compared to those relying on surface-level analysis like BLEU.


#### Q4: Assume you have a large set of story beginnings and you would like to evaluate how well a model completes the stories. What problem would you run into with BLEU and COMET? Would the same disadvantages apply to BERTScore and why? Give your justification. Answer in ~6 sentences.

When evaluating story completions using BLEU, COMET, and BERTScore, several challenges emerge due to the unique nature of storytelling:

*1. BLEU's Limitations:* BLEU, focusing primarily on n-gram matching, falls short in assessing narrative coherence and creative diversity essential for evaluating story continuations. It is inclined to reward literal word overlap rather than plot continuity or thematic development, making it ill-suited for story evaluation where diverse, creative expressions are expected.

*2. COMET's Constraints:* Though COMET advances over BLEU by assessing semantic similarities and not just n-gram overlap, it still may struggle with the comprehensive evaluation of stories. Trained on translation data, COMET might not fully grasp the nuances of fluency, contextual coherence, and creativity that are vital in story continuation. Its dependency on a limited set of references can also hinder its ability to appreciate a wide array of creative continuations that diverge from the expected responses.

*3. BERTScore's Potential and Limitations:* BERTScore represents an improvement by considering the contextual coherence of tokens using BERT embeddings, thus not strictly requiring textual consistency with the reference. However, while BERTScore might better capture the context compared to BLEU and COMET, it still might not fully encapsulate narrative quality, creativity, or how well the model develops the plot and characters in a way that aligns with human judgment. This is because, like COMET, it evaluates against a fixed reference, which might not account for the myriad of plausible and creative directions a story could take.

The primary issue with these metrics in the context of story completion is their limited ability to appreciate the breadth of creativity and narrative development. While BLEU and COMET might fail to capture the diversity and creativity due to their reliance on reference matches, BERTScore, despite its contextual awareness, might not entirely overcome these limitations because creative story continuations can significantly differ from reference continuations while still being valid and engaging. The challenge lies in evaluating the continuity, emotional depth, and character development within the narratives, aspects that are crucial for stories but not fully addressed by these metrics.





