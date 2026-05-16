Seeing THINGS Like Humans: Evaluating Human-Informed Visual Embeddings

Gloria Stvol (GS)†, Bryan Roemelt (BR) & Sabrina Zaki (SZ)†
†Aarhus University
{au718303, au670705, au693815}@uni.au.dk

Abstract
Abstract goes here.
Keywords: convolutional neural networks (CNNs), embeddings, vision, THINGS, human judgement, metadata


Introduction (GS, BR & SZ)
Visual embeddings are central to modern computer vision as they provide compact representation that can support classification, retrieval, clustering and transfer to other tasks (Uelwer et al., 2025). However, while embeddings learned from image-label supervision are performing well, they may not necessarily be organized in the same way as human object representations. A model trained to distinguish object classes may become highly effective at separating categories while still failing to capture the graded semantic and perceptual relations that structures in human judgment
Background and Theoretical Framing
Structuring computational sorting starts from conceptual approaches to categorization. At the midpoint of the 20th century, Wittgenstein critiqued the philosophical traditions that stemmed from classical platonic essentialism (Griffin, 1974). An essentialist view posits that the commonalities between things are defined by a resemblance that follows a pre-existing, “essential” form. This raises problems with “universals”, rigid categories, vs embracing the ambiguity of natural language. Language in practice reveals truths in expression and denotation whose boundaries and precision are difficult, yet unnecessary for the speaker to be wholly aware of 
This advances/continues the scientific task of translating the complexities of multi-dimensional qualities into finely-tuning quantified data so that computational tools can better approximate/mimic cognition
In a framework of  “family resemblance”, a term emerging to summarize Wittgenstein, things of a common category are not collectively defined by a single feature, but by multiple compounding features of which one is most commonly shared.   (Rosch and Mervis, 1975). 

Hard boolean vs vector embeddings
probability
Visual embeddings…metadata
Exemplars and prototypes
	Are some categories more salient to basic physiology /  emotions? “Gradiaents of membership”R and M 1975


Visual embeddings and representational learning

ResNet-50 is a convolutional neural network (CNN) that has been trained on large training data, like the  Imagenet dataset. This model is a good stepping stone, as it allows for retraining or transfer learning, instead of training a model from scratch (Koonce, 2021).
Describe “bottleneck block”? 1x1,3x3,1x1, less computationally intensive

Semantic similarity and human judgment
	Example trios from the experiment? 

Human similarity judgments as supervisory signal

Human-aligned evaluation of visual representations

Related Work
(Roads & Love, 2021)
While prior work has shown that neural network representations can be aligned with human similarity judgments, less is known about what human similarity supervision adds beyond image-only adaptation on the same object set. This project addresses this gap through a controlled comparison between an image-only ResNet-50 model fine-tuned on THINGS images and a human-informed model trained with the same visual data plus human similarity structure. Rather than evaluating only on the training similarity signal, we test whether human-informed embeddings better recover independent semantic structure from THINGSplus, including category membership, typicality, nameability, and object properties. Our contribution is therefore not a new alignment algorithm, but a diagnostic analysis of when, where, and for which semantic properties human similarity supervision improves visual embeddings.

Similar but domain specific (birds) managed to improve
Research Objective and Questions
The present project evaluates the effect of human similarity judgments on visual embeddings by investigating whether human similarity supervision improves the semantic organization of learned image representations beyond image-only training alone. While prior work has shown that neural network representations can align with human similarity judgments, less is known about whether human similarity supervision contributes additional semantic structure beyond standard visual adaptation on the same object set.
To investigate this, the project compares an image-only ResNet-50 baseline model with several human-informed variants trained using similarity triplets derived from the THINGS Similarity dataset. The resulting embeddings are evaluated using both standard computer vision benchmarks and independent semantic benchmarks from THINGSplus, including category membership, typicality, nameability, and object property ratings. Rather than proposing a new alignment architecture, the project aims to evaluate whether human similarity judgments provide useful supervisory signals for improving visual embeddings and whether these signals add meaningful semantic structure beyond visual training alone.
More specifically, the project has the following research questions:
Do Human Similarity Judgments Improve Visual Embeddings?
Does incorporating human similarity judgments improve the semantic quality of visual embeddings?
Do human-informed embeddings better recover external semantic benchmarks such as categories, typicality, and nameability?
Does human similarity supervision improve practical performance on tasks such as retrieval or linear probing?
Where does human supervision add useful information, and where is image-only learning already sufficient?

HYPOTHESIS:



Data
This project builds on three properties from its data sources: a large and systematically sampled set of object concepts, large-scale human similarity judgments over those same concepts, and an independent set of semantic norms against which learned embeddings can be evaluated. The THINGS database (Hebart et al., 2019 link to zotero), THINGS Similarity (Hebart et al., 2020 link to zotero), and THINGSplus (Stoinski et al., 2024 link to zotero) jointly provide structure to this project pipeline. All three are aligned around a shared concept identifier covering 1,854 object concepts, which makes it possible to use one source as a training signal (THINGS images and THINGS Similarity triplets) while reserving another as a held-out evaluation target (THINGSplus norms). Existing alternatives such as ImageNet (Deng et al., 2009 link to zotero) provide far more images per concept but lack human similarity judgments and concept-level semantic norms, while smaller stimulus sets (e.g. Brodeur et al., 2014) lack the scale and naturalistic image quality needed for fine-tuning a deep visual model. The remainder of this section describes each dataset and the data-wrangling procedure used to align them.
THINGS (BR)
THINGS is a dataset containing 1,854 diverse objects concepts that are sampled systematically from concrete picturable and nameable nouns in the American English language (Hebart et al., 2019).
The THINGS dataset is a comprehensive database comprising 1,854 diverse object concepts and 26,107 high-quality, naturalistic object images (Hebart et al., 2019 link zotero). The object concepts were systematically sampled from concrete, picturable, and nameable nouns in American English through a three-stage procedure. First candidate nouns were drawn from an existing word database (Brysbaert et al., 2014 link zotero), clarified against WordNet synsets (Fellbaum, 1998 link zotero), and validated through an online naming task in which 1,395 workers labeled a representative image of each concept, retaining those named consistently. Based on everyday object naming, the majority of these concepts represent items that are inanimate (89%) and artificial (67%).
For each object concept, the dataset provides at least 12 image examples. These are color photographs in which the target object is the central component against a naturalistic background, rather than cropped against uniform backgrounds. Additional objects of other concepts were allowed to be present in an image (e.g., limbs in images of clothing), but they were not the leading object; faces were similarly omitted in most images (except for concepts defined by human faces such as "face," "man," "woman"), as humans are highly salient to faces. Candidate images were sourced through manual web searches and supplemented with images from ImageNet (Deng et al., 2009 link zotero), overly similar duplicates within each concept were removed using activations from a pretrained convolutional neural network (Hebart et al., 2019 link zotero). For consistency, all images have been uniformly cropped to a square format with a minimum size of 480 × 480 pixels and a mean size of 996 × 996 pixels across the dataset. The database and all associated data are publicly available through the Open Science Framework at http://doi.org/10.17605/osf.io/jum2f.
THINGS Similarity
The THINGS Similarity dataset (Hebart et al., 2020 link zotero) is a project building on the original THINGS database (reference section 5.1), made in an attempt to quantify human similarity judgments across all the 1,854 object concepts.
The motivation for this dataset formation came from a limitation of similarity judgments as traditionally used, while they reveal how much two objects resemble each other, they remain ignorant to which properties explain that resemblance and which dimensions any two representations share (Hebart et al., 2020 link zotero). For instance, most people would judge a dog and a cow to be more similar than a dog and a car, presumably because dogs and cows share dimensions such as animacy, naturalness, and softness, but a similarity score alone does not surface these underlying dimensions. Hebart et al. (2020 link zotero) argues that to move from describing similarity to understanding the structure of mental object representations, it is important to identify the dimensions that affect ultimate similarity judgment. They specify two criteria that those dimensions must meet, they must be predictive of behavior, so as to characterize the representational space, and they must be interpretable, so as to move beyond just the category description towards interpretability. With that said rather than using pairwise similarity ratings, which assume that all relevant dimensions are equally available to the observer regardless of how different two objects are, Hebart et al. (2020 link to zotero) adopted a triplet odd-one-out task. In each trial, participants viewed three object images side by side and selected the one least similar to the other two.
By choosing the odd one out, participants implicitly indicated which of the remaining pair was most similar within that specific context, and by varying the third object across trials, the task sampled a wide range of contexts in which pairs of objects could be compared. The benefit of this design is that the similarity of the two objects can be more isolated from the specific context (third object), allowing object similarity to be expressed as the probability of any two objects being grouped together across contexts.
Similarity judgments were collected through the online crowdsourcing platform Amazon Mechanical Turk from workers located in the United States. In total, 1.46 million unique trial responses were gathered from 5,301 participants. Although this is a large scale sample, it represents approximately 0.14% of the roughly 1.06 billion possible unique triplet combinations for the 1,854 objects, making exhaustive sampling of the full similarity matrix practically infeasible. Each participant completed the task in sets of 20 trials, with three object images displayed side by side in a browser window. Participants were instructed to focus their judgment on the object itself, and were given no further instructions on the strategy to minimize bias, and were told to base their choice on a best guess if they did not recognize an object.
Because the full similarity matrix could not be sampled directly, Hebart et al. (2020 link zotero) trained a computational model on the collected triplets to predict similarity for all pairs of the 1,854 objects. The model learns a low-dimensional, sparse, and non-negative embedding in which each object is represented as a vector across interpretable dimensions, and triplet choices are predicted from the dot products between object embeddings. After fitting, the model converged on a total of 49 reproducible dimensions reflecting both perceptual properties (e.g., colorful, round, shiny/transparent) and conceptual properties (e.g., animal-related, food-related, valuable). The model predicted held-out triplet choices at 64.60% accuracy, approaching the noise ceiling of 67.22% set by inter-participant consistency, and reconstructed an independently collected, fully sampled similarity matrix of 48 objects.
Both the raw odd-one-out responses and the learned 49-dimensional embedding are publicly available through the Open Science Framework at https://osf.io/z2784, and in the present project, this data serves as the similarity embeddings for the human-informed adaptation of ResNet-50.
THINGSplus
For the THINGS dataset, additional metadata was added by extending the dataset by adding concept and image-specific norms and metadata for all the images (Stoinski et al., 2024)
THINGSplus is a publicly available dataset through the Open Science Framework at https://osf.io/jum2f/ (Stoinski et al., 2024 link zotero). This is another extension to the original THINGS database that adds concept and image specific norms and metadata for all 1,854 object concepts and 26,107 images. The norms were collected through a series of crowdsourced experiments on Amazon Mechanical Turk and include three sets of measures used in the present project. First, the original 27 high-level categories from THINGS were expanded to 53 superordinate categories (e.g., "animal," "vehicle," "tool," "sea animal," "musical instrument"), with each of the 1,448 concepts that belong to a category independently assigned by two annotators and verified by a third. Second, typicality ratings were collected for every category member by having participants drag-and-drop sort objects by how representative they are of the category, allowing graded within-category structure to be measured beyond binary membership. Third, concept-level property ratings were gathered along 11 dimensions. Additionally, image-specific nameability scores were derived from human-generated labels of the prominent object in each of the 26,107 images. Internal rating consistency was high across nearly all measures and the norms correlated strongly with external validation datasets.
In the present project, THINGSplus provides the independent semantic structure against which the baseline and human-informed ResNet-50 embeddings are evaluated. Because the human-informed model is trained on similarity triplets rather than on category labels, typicality, or object properties, the THINGSplus norms function as held-out targets: the 53 categories support cross-validated category recovery analyses, the typicality ratings allow testing whether embedding distances within a category reflect graded human typicality, the 11 property ratings serve as candidate semantic axes that can be regressed against embedding dimensions, and the image-level nameability scores enable image-specific quality controls. This separation between training signal (THINGS Similarity triplets) and evaluation signal (THINGSplus norms) is what allows the project to test whether human similarity supervision generalizes to independent semantic structure rather than just fitting the data it was trained on.
Data wrangling and preprocessing
Before model training, the raw THINGS, THINGSplus, and THINGS Similarity files were converted into a set of analysis-ready tables. This step was necessary because the source data were distributed across different files and levels of measurement: concept-level metadata, image-level metadata, object-property ratings, category annotations, image archives, and human odd-one-out similarity judgments. The goal of preprocessing was to create one consistent concept index that could be used across visual training, human-similarity supervision, embedding extraction, and evaluation.

All sources were aligned around a canonical concept identifier, `concept_index`/`concept_id`, ranging from 0 to 1853. This identifier was derived from the THINGS concept metadata and joined to image-level metadata through the stable THINGS `unique_id` field. Using this identifier throughout the project reduced ambiguity from concept names alone and ensured that visual images, behavioral similarity judgments, and THINGSplus variables referred to the same object concepts.

The main raw inputs used during preprocessing were:

| Source | Role in preprocessing |
|---|---|
| `concepts-metadata_things.tsv` | Concept-level metadata, including THINGS unique IDs, concept names, category labels, and lexical/concreteness variables. |
| `_images-metadata_things.tsv` | Image-level metadata linking individual image files to THINGS concepts. |
| `_property-ratings.tsv` | THINGSplus object-property norms and image-label/nameability variables reserved for evaluation. |
| `category53_long-format.tsv` | THINGSplus annotations for 53 higher-level semantic categories. |
| `images_THINGS.zip` and `images_THINGSplus-CC0.zip` | Image archives used for model training and embedding extraction. |
| `data/processed/triplets.csv` | Human odd-one-out judgments used to construct concept-level pairwise similarities and triplets. |

The setup script first produced two processed tables, `data/processed/concepts.csv` and `data/processed/images.csv`. The concept table retained concept identity fields, category labels, lexical variables, nameability-related variables, and object-property ratings. The image table retained image identifiers, raw image paths, concept assignments, and the canonical concept index. The image-to-concept join was performed with `unique_id`, not with concept names, because `unique_id` is the stable identifier supplied by THINGS.

The raw image paths did not directly match the extracted archive layout, so a separate metadata step normalized them into local paths under the THINGS and THINGSplus image folders. This produced `data/baseline/image_metadata.csv`, containing the image ID, local image path, concept ID, concept name, unique ID, and a file-existence flag. The metadata audit confirmed that all 1,854 concepts were represented and that all 27,961 image files used by the pipeline were present.

The image metadata were then split into training, validation, and test sets with a 70/15/15 split within each concept. A concept-level split was not used for classifier training because the baseline and human-informed classifiers predict all 1,854 THINGS concepts; excluding concepts from training would make those labels unlearnable. The within-concept image split ensured that every concept appeared in every split while keeping held-out images for validation and testing. The resulting split contained 19,713 training images, 4,124 validation images, and 4,124 test images.

Human similarity required a separate preprocessing path. The available source was first inspected to determine whether it contained raw odd-one-out triplets, pairwise similarities, a full predicted similarity matrix, or 49-dimensional human embeddings. In the executed pipeline, the odd-one-out judgments were mapped to canonical concept IDs and converted into unordered pairwise evidence. For each trial, the two non-odd objects contributed positive evidence for their pair, whereas pairs involving the odd object contributed negative evidence. Repeated observations of the same unordered pair were aggregated into a similarity score between 0 and 1. Diagonal pairs were removed, and unordered duplicates such as `(A, B)` and `(B, A)` were treated as the same pair.

The resulting pairwise similarity table was split into non-overlapping train, validation, and test pair sets. These pair splits prevent direct reuse of identical concept pairs across training and evaluation diagnostics. However, because all pairwise scores originate from the same behavioral similarity source, held-out human-similarity pairs were treated as within-source alignment diagnostics rather than as fully independent semantic tests. THINGSplus variables were kept separate and used only for external evaluation.

For the triplet-based models, training pairs were converted into anchor-positive-negative triplets. For each anchor concept, positives were sampled from the upper tail of that anchor's similarity distribution and negatives from the lower tail. Pairs with very small similarity differences were avoided when the similarity scale supported a fixed gap threshold. Each saved triplet retained the anchor, positive, and negative concept IDs, their unique IDs, the positive and negative similarity values, the similarity gap, and the random seed. A shuffled-control triplet file was also generated. This control preserved the number of triplets and the anchor-frequency distribution while disrupting the meaningful positive/negative assignments.

For the joint matrix-training model, the pairwise training similarities were also converted into a sparse 1,854 x 1,854 human similarity matrix. Only finite off-diagonal entries were used during the matrix loss. A shuffled-matrix control was created by permuting the matrix across concept identities, preserving the distribution of similarity values while destroying the semantic mapping between concepts.

Throughout preprocessing, audit reports were written to verify concept coverage, image availability, missing or unmatched concepts, duplicate and diagonal pair removal, train/validation/test pair overlap, similarity scale, triplet counts, triplet similarity gaps, concepts with no triplets, and shuffled-control validity. These reports are important for interpretation because the project compares small differences between model variants. The preprocessing checks confirmed that the models used the same concept IDs, image splits, and evaluation variables, and that THINGSplus categories, nameability, typicality, and object-property ratings were not used in any training objective.


Figure X 
Data wrangling pipeline. Raw THINGS, THINGSplus, and THINGS Similarity files are aligned through the canonical THINGS concept identifier, converted into image splits and human-similarity supervision files, and checked with audit reports before model training.

Method
The analysis compared an image-only ResNet-50 baseline with a series of human-informed variants. The experimental design was intentionally incremental. First, a standard image-classification baseline was trained on THINGS object images. Second, human similarity was added during additional fine-tuning using several triplet-based objectives. Third, a joint matrix-training condition was introduced to test whether human similarity should be present from the beginning of THINGS adaptation rather than added after the visual classifier had already been formed. All models were evaluated using the same image splits, embedding extraction procedure, and benchmark suite.

Image-only baseline model
The baseline model was an ImageNet-pretrained ResNet-50. Pretraining was used because THINGS contains many object concepts but relatively few images per concept, making full training from random initialization impractical for this project. The ImageNet classification head was replaced with a new fully connected layer predicting the 1,854 THINGS concepts.

Images were resized to 224 x 224 pixels and normalized using the standard ImageNet mean and standard deviation. Training images were augmented using random horizontal flips and mild color jitter, whereas validation and test images were only resized and normalized. The baseline was trained with cross-entropy loss on the image-level THINGS concept labels. Training followed a two-stage transfer-learning schedule: first, only the new classification head was trained for five epochs with a learning rate of 1e-4; second, the final ResNet block (`layer4`) and the classification head were unfrozen and trained for ten additional epochs with a learning rate of 1e-5. AdamW was used as optimizer with weight decay of 1e-4. The checkpoint with the highest validation top-1 accuracy was used as the image-only baseline and as the initialization point for the triplet-based human-informed models.

Human-informed training conditions
Human similarity supervision was treated as concept-level relational supervision rather than as image-level labels. The odd-one-out judgments were converted into pairwise concept similarities and then into triplets of the form anchor concept, positive concept, and negative concept. For a given anchor, the positive concept was more similar according to human judgments, whereas the negative concept was less similar. The standard triplet-based objective was:

L = L_classification + lambda L_similarity

where L_classification is cross-entropy over the 1,854 concept labels, and L_similarity encourages the anchor representation to be closer to the positive concept than to the negative concept. Human-informed fine-tuning was deliberately compared with shuffled controls so that improvements shared by the real and shuffled conditions could be interpreted as generic fine-tuning or regularization rather than evidence for meaningful human-similarity structure.

Fixed-prototype triplet regularization
The first human-informed model used fixed concept prototypes derived from the trained image-only baseline. For each concept, a prototype was computed by averaging baseline embeddings from training images only. Validation and test images were not used to compute these prototypes. During fine-tuning, the current embedding of a training image served as the anchor, and a human-derived triplet supplied the positive and negative concepts. The similarity loss compared the anchor embedding to the fixed positive and negative prototypes using a cosine-margin triplet loss:

max(0, margin - cos(anchor, positive) + cos(anchor, negative))

This condition tested whether human similarity could act as a weak auxiliary regularizer while preserving the original image-classification objective.

Current-batch concept-prototype training
The second strategy applied the triplet loss to the model's current concept geometry rather than to fixed baseline prototypes. Training batches were constructed from human triplets. For each triplet, images were sampled from the anchor, positive, and negative concepts, and current embeddings were averaged within each role to form batch-level prototypes. The triplet loss was then applied to these current anchor, positive, and negative prototypes.

This design better matched the concept-level nature of the human similarity data because the loss acted on relations between concepts rather than on individual images. However, it was computationally expensive: using all triplets would require many more batches than the baseline classifier. The implemented run therefore used a capped number of training batches, making it a diagnostic condition rather than a fully matched full-budget training run.

Strong human-weighting condition
A third condition tested whether the previous triplet losses were too weak to substantially reshape the embedding space. This model returned to the computationally cheaper fixed-prototype design, but increased the relative influence of the human similarity objective by down-weighting the classification loss:

L = lambda_ce L_classification + lambda_similarity L_similarity

This condition was designed to test a possible trade-off. If human similarity and classification require different representational geometries, stronger human weighting should increase within-source human-similarity alignment but may not improve, and could even reduce, classification, retrieval, or THINGSplus transfer.

Joint matrix training
The triplet-based models all started from an already-trained THINGS classifier. This means that human similarity was introduced only after the model had already organized its representation around 1,854-way visual classification. To test whether this order of training mattered, an additional joint matrix-training condition was added. This model started from the same ImageNet-pretrained ResNet-50 initialization as the baseline, but included the human similarity objective from the first epoch of THINGS fine-tuning.

The joint model used the same two-stage training schedule as the image-only baseline: five head-only epochs followed by ten epochs in which `layer4` and the classification head were trained. The classification term was identical to the baseline cross-entropy loss. The human term was a matrix-alignment loss computed within each mini-batch. For each batch, image embeddings were grouped by concept and averaged to form current concept prototypes. The cosine similarity matrix among these current prototypes was then compared with the corresponding submatrix of the human pairwise similarity matrix. Both the model similarities and human similarities were z-scored within the available batch pairs before mean-squared error was computed:

L = CE(image concept) + lambda_matrix MSE(zscore(S_model), zscore(S_human))

Only off-diagonal concept pairs with finite human similarity values were used, and batches with too few valid concept pairs did not contribute a matrix loss. The default matrix-loss weight was lambda_matrix = 0.05, with a minimum of 16 human-similarity pairs required per batch. This strategy differs from the triplet methods in two ways: it exposes the model to human similarity while the THINGS representation is being formed, and it preserves more of the local relational structure of the human similarity matrix instead of reducing the signal to isolated anchor-positive-negative constraints.

A shuffled-matrix control was included for this condition by permuting the human similarity matrix across concept identities. This preserves the distribution of similarity values while disrupting the meaningful mapping between concepts and human similarity structure. As with the triplet controls, the shuffled-matrix model is necessary for distinguishing genuine human-structure learning from generic effects of adding an auxiliary loss.

Shuffled-control training
Shuffled controls were used wherever feasible. For triplet models, the number of triplets and the anchor-frequency distribution were preserved, but positive and negative assignments were shuffled. For the matrix model, the human similarity matrix was permuted across concept identities. These controls were essential because continued fine-tuning alone can improve classification or retrieval. A human-informed model was therefore interpreted as learning meaningful human structure only when it improved relative to both the image-only baseline and its matched shuffled control.

Embedding extraction
After training, embeddings were extracted by removing the final classification layer and using the penultimate ResNet-50 representation. This produced a 2,048-dimensional embedding for each image. Image-level embeddings were saved for all images, and concept-level embeddings were computed by averaging image embeddings within each concept. Both image and concept embeddings were L2-normalized before evaluation. Concept-level embeddings were used for concept-level semantic and human-similarity analyses, whereas image-level embeddings were used for retrieval-based evaluation.

Evaluation benchmarks
The models were evaluated with three benchmark families. First, held-out classification performance was measured using top-1 and top-5 accuracy on the test split. Second, practical embedding utility was evaluated with retrieval metrics, including image retrieval and image-to-concept retrieval at hit@1, hit@5, and hit@10. These metrics tested whether images from the same object concept were close in embedding space.

Third, semantic transfer was evaluated using THINGSplus variables that were not used during training. Category structure was assessed using nearest-neighbor and linear-probe approaches over concept embeddings. Continuous THINGSplus variables, including nameability, lexical/concept variables, and object-property norms, were evaluated using regression or correlation-based summaries such as Spearman's rho. These analyses tested whether human-informed training improved recovery of independent semantic structure beyond the human similarity source itself.

Finally, the learned concept embeddings were compared with human similarity judgments. Model cosine similarities between concept embeddings were correlated with held-out human pairwise similarities, and triplet satisfaction was computed as:

cos(anchor, positive) > cos(anchor, negative)

A margin-based version of this diagnostic was also computed. These human-similarity evaluations were treated as alignment diagnostics rather than fully independent semantic benchmarks because they came from the same family of behavioral judgments used for training.

Implementation and reproducibility
All preprocessing, training, embedding extraction, benchmarking, and figure generation were implemented as a reproducible numbered script pipeline. Model comparisons used the same image metadata, image splits, concept identifiers, embedding extraction code, and benchmark scripts across conditions. This shared pipeline was important for interpretation: differences between models reflect the training objective and control condition rather than differences in data processing or evaluation.
Results

Discussion
Summary and Interpretation of Results

Limitations
Too computationally intensive for quick revisions in data processing
The image-only baseline already encodes human-defined category structure. The ResNet-50 model is initialized from ImageNet-pretrained weights (Deng et al., 2009 link zotero) and subsequently fine-tuned on THINGS concept labels, both of which include forms of human-curated semantic supervision. The contrast investigated in the present project is therefore more accurately characterized as a comparison between two forms of human supervision (categorical labels alone versus categorical labels combined with similarity judgments) rather than between purely visual learning and human-informed learning. A stronger test would require a self-supervised initialization, which was beyond the computational scope of the present project.

Future Studies


Conclusion
Communication

IKEA - Human-like Product Recommendation Tool

To the product team
Plots - redo them

These are the questions we answered like this. 

“Why it fits: your project is about whether visual embeddings become more human-like when trained with similarity judgments. IKEA’s product world depends heavily on human-intuitive similarity: “find me a chair like this, but softer,” “similar lamp, more minimal,” “things that visually fit together.” IKEA also has app/room-planning products like IKEA Kreativ, where users try products in a room or scanned space.

For the assignment, you could define the client as:

Client: IKEA’s visual search and product recommendation team
They want to know whether human-informed visual embeddings can improve product discovery by retrieving objects that feel similar to people, rather than only objects that are visually/classification-wise similar.”




References






