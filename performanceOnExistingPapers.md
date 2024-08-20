# Performance of DistMult, TransE, and RotatE on Existing Papers

DistMult: Embedding Entities and Relations for Learning and Inference in Knowledge Bases
Learns embeddings for entities and relations ("matrix factorization") and score is calculated by multiplying matrices. 

DistMult achieved:
On FB15k:
    MRR: 0.35
    HITS@10: 57.7
On WN-18:
    MRR: 0.83
    HITS@10: 94.2


TransE: Translating Embeddings for Modelling Multi-relational Data
Learns embeddings in an embedding space s.t. for (h, r, t), h + r is approx t in the embedding space.

TransE achieved:
On FB15k:
    MR: 125 (Filtered)
    HITS@10: 47.1 (Filtered)


RotatE: Knowledge Graph Embeedding by Relational Rotation in Complex Space
Learns embeddings s.t. each relation is a rotation from the source entity to the target entity. 
Uses self-adversarial negative sampling to balance frequent and in-frequent entities. 

RotatE achieved:
FB15k:
    MRR: 0.797
    HITS@1: 0.746
    HITS@3: 0.830
    HITS@10: 0.884


RotatE has (updated) results for the other models so they're included below as updated results:

TransE:
On FB15k:
    MRR: 0.463
    HITS@1: 0.297
    HITS@3: 0.578
    HITS@10: 0.749
On WN18:
    MRR: 0.495
    HITS@1: 0.113
    HITS@3: 0.888
    HITS@10: 0.943
On FB15k-237:
    MRR: 0.249
    HITS@1:
    HITS@3:
    HITS@10: 0.465
On WN18RR:
    MRR: 0.226
    HITS@1:
    HITS@3:
    HITS@10: 0.501

DistMult:
On FB15k:
    MRR: 0.798
    HITS@1:
    HITS@3:
    HITS@10: 0.893
On WN18:
    MRR: 0.797
    HITS@1:
    HITS@3:
    HITS@10: 0.946
On FB15k-237:
    MRR: 0.241
    HITS@1: 0.155
    HITS@3: 0.263
    HITS@10: 0.419
On WN18RR:
    MRR: 0.43
    HITS@1: 0.39
    HITS@3: 0.44
    HITS@10: 0.49

RotatE:
On FB15k:
    MRR: 0.797
    HITS@1: 0.746
    HITS@3: 0.830
    HITS@10: 0.884
On WN18:
    MRR: 0.949
    HITS@1: 0.944
    HITS@3: 0.952
    HITS@10: 0.959
On FB15k-237:
    MRR: 0.338
    HITS@1: 0.241
    HITS@3: 0.375
    HITS@10: 0.533
On WN18RR:
    MRR: 0.476
    HITS@1: 0.428
    HITS@3: 0.492
    HITS@10: 0.571