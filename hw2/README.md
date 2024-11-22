# HW2: Lucas-Kanade Tracking

## Overview
This homework consists of 3 major sections.

In the first section, you will implement a simple Lucas-Kanade (LK) tracker with one single template.

The second section requires you to implement a motion subtraction method for tracking moving pixels in a scene.

In the final section, you shall study efficient tracking such as inverse composition.

## Downloading the assignment

Using `git pull` from the terminal while inside the git folder should help download all of the files.

Please download the `data` folder from this [Drive link](https://drive.google.com/drive/folders/1fBkNSKY_X1JF2fc_dCX4ILsCkBGRq2ya?usp=drive_link). Please let us know early if you are facing issues in downloading. The internal folder structure should look something like this

```
hw2
>code
>>...
>data
>>...
>16_820_F23_HW2_Lucas_Kanade_Tracking.pdf
>README.md
```

#### FAQs

**Visualization**: You are free to visualize the bounding boxes and frames any way you see fit. To help with questions 1.3 and 1.4
we have provided a helper script `plotRects.py` which you can use.

Example usage: `python plotRects.py q1.3 car`

