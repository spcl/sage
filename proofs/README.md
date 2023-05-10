Install [tamarin-prover](https://tamarin-prover.github.io/) version 1.6.1.
To get the output run the following command:
```
tamarin-prover SAKE.spthy --prove
```

Alternatively, use pre-built tamarin-prover inside container:

```
podman run --rm -v ./:/proofs --security-opt label=disable docker.io/darrenldl/tamarin-prover:1.6.1 tamarin-prover /proofs/SAKE.spthy --prove
```

Upon successful completion, output should match SAKE_proofs.txt.
