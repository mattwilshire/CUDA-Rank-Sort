# Rank Sort 

| Specs | Name |
| --- | ----------- |
| CPU | Ryzen 9 5900x |
| GPU | RTX 4080 |


Block size: 512
## Results

### Sorting array of 100,000 elements
#### CPU Results
```
CPU Sort: 11.132000s
```

#### GPU Results
Using global memory accessible to each thread on the GPU is **slow**, rank sort requires comparing from memory every iteration, the difference in runtimes is shown below.
```
Using Global GPU Memory: 0.061000s
```
```
Global value stored in local variable: 0.044000s
```
```
Using Shared Memory: 0.033000s
```

### Sorting array of 1,000,000 elements
Not testing this on CPU, waste of time.

```
Using Global GPU Memory: 4.638000s
```
```
Global value stored in local variable: 3.102000s
```
```
Using Shared Memory: 0.033000s
```

### Sorting array of 100,000,000 elements
```
Using Shared Memory: 0.039000s
```

You notice how the time is not alternating with shared memory when the size of the array goes up. This is because my GPU clock is 2535 MHz, this high clock speed allows the threads to run stupidly fast with a 512 block size.