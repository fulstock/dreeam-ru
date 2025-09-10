# SECCOL Test Set Evaluation Report

## Overview

This report presents the evaluation results of the DREEAM-RU model on the SECCOL test dataset for cybersecurity-focused document-level relation extraction in Russian text.

## Dataset Statistics

- **Total Files Processed**: 436 (100% success rate)
- **Total Gold Relations**: 613
- **Total Predicted Relations**: 476
- **Processing Success Rate**: 100%

## Overall Performance

| Metric | Score |
|--------|-------|
| **Precision** | 59.87% |
| **Recall** | 46.49% |
| **F1 Score** | 52.34% |
| **True Positives** | 285 |
| **False Positives** | 191 |
| **False Negatives** | 328 |

## Per-Relation Analysis

### Performance by Relation Type

| Relation | Precision | Recall | F1 Score | Support | Notes |
|----------|-----------|--------|----------|---------|-------|
| **LOCATED_IN** | 66.24% | 51.49% | 57.94% | 202 | Best overall performance |
| **INSTANCE_OF** | 56.38% | 51.96% | 54.08% | 306 | Most frequent relation |
| **ORIGIN_FROM** | 62.07% | 40.00% | 48.65% | 45 | Good precision, lower recall |
| **PART_OF** | 50.00% | 12.50% | 20.00% | 24 | Challenging relation |
| **SUBCLASS_OF** | 50.00% | 2.78% | 5.26% | 36 | Most difficult relation |

### Macro vs Micro Averages

- **Macro Average**: P=56.94%, R=31.74%, F1=37.19%
- **Micro Average**: P=59.87%, R=46.49%, F1=52.34%

The significant difference between macro and micro averages indicates that performance varies considerably across relation types, with the model performing better on more frequent relations.

## Performance Analysis

### Strengths

1. **Geographic Relations**: 
   - `LOCATED_IN`: Best F1 score (57.94%) with strong precision (66.24%)
   - Model effectively captures spatial relationships in cybersecurity contexts

2. **Classification Relations**:
   - `INSTANCE_OF`: Balanced performance (54.08% F1) on the most frequent relation type
   - Handles entity classification reasonably well

3. **Origin Relations**:
   - `ORIGIN_FROM`: Good precision (62.07%) for source attribution

### Challenges

1. **Hierarchical Relations**:
   - `SUBCLASS_OF`: Very low recall (2.78%), indicating difficulty with taxonomic relationships
   - Only 1 out of 36 gold relations correctly identified

2. **Compositional Relations**:
   - `PART_OF`: Low recall (12.50%), struggles with part-whole relationships
   - Only 3 out of 24 gold relations correctly identified

3. **Less Frequent Relations**:
   - Model shows bias toward more frequent relation types
   - Performance degrades significantly for relations with lower support

## Domain-Specific Insights

### Cybersecurity Context

The SECCOL dataset focuses on cybersecurity events and entities, which presents unique challenges:

1. **Technical Terminology**: 
   - High precision for `ORIGIN_FROM` suggests good handling of attack attribution
   - `LOCATED_IN` performance indicates effective geographic entity linking

2. **Entity Classification**:
   - Moderate success with `INSTANCE_OF` shows the model can classify cybersecurity entities
   - Lower performance on `SUBCLASS_OF` suggests difficulty with fine-grained taxonomies

3. **Complex Relationships**:
   - `PART_OF` relations in cybersecurity often involve technical components
   - Lower performance may reflect the complexity of technical part-whole relationships

## Comparison with NEREL Results

| Metric | SECCOL | NEREL | Difference |
|--------|---------|-------|------------|
| **Precision** | 59.87% | 59.16% | +0.71% |
| **Recall** | 46.49% | 57.55% | -11.06% |
| **F1 Score** | 52.34% | 58.34% | -6.00% |

### Key Observations

1. **Similar Precision**: Both datasets show comparable precision (~59%), indicating consistent model quality
2. **Lower Recall on SECCOL**: 11% lower recall suggests cybersecurity relations are harder to detect
3. **Domain Complexity**: SECCOL's specialized cybersecurity domain appears more challenging than NEREL's general domain

## Error Analysis

### Common Error Patterns

1. **False Negatives (328 total)**:
   - Model misses many `SUBCLASS_OF` and `PART_OF` relations
   - Likely due to complex technical terminology and hierarchical structures

2. **False Positives (191 total)**:
   - Over-prediction of `LOCATED_IN` and `INSTANCE_OF` relations
   - May indicate model bias toward more frequent relation types

### Improvement Opportunities

1. **Better Handling of Rare Relations**:
   - Augment training data for `SUBCLASS_OF` and `PART_OF`
   - Consider class balancing techniques

2. **Domain-Specific Tuning**:
   - Cybersecurity-specific entity embeddings
   - Fine-tuning on more cybersecurity-focused data

3. **Hierarchical Modeling**:
   - Specialized architectures for taxonomic relations
   - Multi-level classification approaches

## Conclusions

The DREEAM-RU model shows solid performance on the SECCOL cybersecurity dataset, with an F1 score of 52.34%. Key findings:

### Strengths
- Robust precision (59.87%) indicates quality predictions
- Excellent performance on geographic relations (`LOCATED_IN`)
- Good handling of entity classification (`INSTANCE_OF`)

### Areas for Improvement
- Low recall (46.49%) suggests many relations are missed
- Poor performance on hierarchical relations (`SUBCLASS_OF`, `PART_OF`)
- Domain-specific challenges in cybersecurity terminology

### Recommendations
1. **Data Augmentation**: Increase training examples for rare relation types
2. **Domain Adaptation**: Cybersecurity-specific model fine-tuning
3. **Architecture Improvements**: Specialized handling of hierarchical relations
4. **Ensemble Methods**: Combine multiple models for better coverage

The results demonstrate that while the model performs reasonably well on cybersecurity texts, there is significant room for improvement, particularly in handling the domain's specialized vocabulary and complex hierarchical relationships. 