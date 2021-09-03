# ML_Ops
Submitted by: Anubhooti Jain (M20CS052)
=========================================

Accuracy table after running the code (quiz.py file)

![my-result-1](/results/table.png)

Observations -
1. For the resolution 16*16 and 32*32 we still have decent results. But for 64*64 the accuracy drops significantly. This aligns with the fact that resizing an image of 8*8 to 64*64 pixelates the image too much, probably destroying the information the pixel contains. 

2. Another observation is that, as image resolution increases, the accuracy decreases. It follows the same reasoning as 1. 
