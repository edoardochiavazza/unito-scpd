# Unito SCPD

Federated Learning (FL) is a paradigm where multiple clients collaborate on machine learning tasks using private data under an aggregator’s coordination.
Local data remains isolated, with learning occurring in rounds where clients compute model updates using private data, aggregate results on the server, and broadcast for subsequent rounds.
Two main federated settings exist: 
- **cross-device** FL involves numerous edge
devices (thousands) with limited computational power and reliability, while
- **cross-silo** FL involves organizations (typically 2-100 parties) where communication and computation constraints are less restrictive.
This work proposes cross-silo FL algorithms for classification, drawing inspiration from AdaBoost and distributed boosting literature. The algorithms impose minimal constraints on client learning settings, accommodating models not specifically designed for FL, such as decision trees and SVMs. Models trained at each epoch combine to form ensemble learners that strengthen performance by iterating and combining results.
