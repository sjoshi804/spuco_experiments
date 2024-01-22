# Experiments with SpuCo Package

- Group inference tuning
- Robust training tuning
- E2E tuning
- ERM tuning

Run guild queues for gpu affinitization
```
for i in {2..3}; do guild run queue -b --gpus="$i" -y; done
```