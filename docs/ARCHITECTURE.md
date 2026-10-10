# Architecture (planned)

```text
bank-a data -> bank-a trainer --\
bank-b data -> bank-b trainer ----> coordinator -> versioned model checkpoint
bank-c data -> bank-c trainer --/       |
                                    metrics / monitoring
```

All services run in separate Kubernetes pods and communicate using a versioned protocol. Only trained model updates are transmitted to coordinator (updates may still leak information). Deploy the inference API separately later; the Rust bank application is an optional demonstration frontend, not a core dependency.
