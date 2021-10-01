# ML_Ops
=========================================

Creating Docker Image -
1. Create Docker file.
2. Use build command. 

'''
docker build --rm -f docker/Dockerfile . -t mnist:latest
'''
3. Docker run for creating and running the container. 

'''
docker run -it mnist:latest
'''

Previous Assignment ==> 
In the assignment the modified changes are as follows -
1. Added Validation Set
2. Storing and loading Models

To see the results, please run "plot.py" file.

I got the followig results for a train:test:val split of 70:15:15 -

```
=============================
Creating Validation Split    
=============================
Number of samples in train:val:test = 1257:270:270
Now training...
Gamma   Train Acc Val Acc
0.0001 = 0.99,   0.96
0.0005 = 1.00,   0.99
0.001 = 1.00,    0.99
0.005 = 1.00,    0.93
0.01 = 1.00,     0.76
0.05 = 1.00,     0.10
Skipping gamma 0.05 because of low validation accuracy
```

Observation - Best Gamma Value = 0.001 with a validation accuracy of 0.993 and a Test accuracy for best gamma 0.9481481481481482
