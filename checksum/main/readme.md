### Optimal small single-loop implementation. No self-modifying code, no inner loop.

```
make clean && make CAP=80 GENARGS= test_generated && make run_generated
while true; do make run_generated | grep Runtime; done
```

```
Runtime: 0.493969 s
Runtime: 0.494154 s
Runtime: 0.494253 s
Runtime: 0.494583 s
Runtime: 0.493944 s
Runtime: 0.493595 s
Runtime: 0.49398 s
Runtime: 0.49409 s
Runtime: 0.494441 s
Runtime: 0.493974 s
```

99 % of peak performance

### Small single-loop implementation with one adversarial NOP inside. No self-modifying code, no inner loop.

```
make clean && make CAP=80 GENARGS=--with_adversarial_nop test_generated && make run_generated
while true; do make run_generated | grep Runtime; done
```

```
Runtime: 0.496967 s
Runtime: 0.497494 s
Runtime: 0.497201 s
Runtime: 0.497759 s
Runtime: 0.497222 s
Runtime: 0.497271 s
Runtime: 0.497287 s
Runtime: 0.497297 s
Runtime: 0.502373 s
Runtime: 0.496825 s
```

98 % of peak performance

### Large implementation with self-modifying code and inner loop to hide its impact.

```
make clean && make CAP=80 GENARGS="--with_self_modification --num_iters 1000 --with_inner_loop --num_inner_iters 5000 --num_shifts 90" test_generated && make run_generated
```

Observed 100% of peak performance. Unfortunately adversarial NOP is not detectable with this approach.
