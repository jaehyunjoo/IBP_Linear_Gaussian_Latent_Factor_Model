# A Linear-Gaussin Latent Factor (Feature) Model using a IBP prior

This Python code implements a linear-Gaussian latent factor model using a Indian buffet process prior as illustrated in [Griffiths and Ghahramani (2011)](https://dl.acm.org/citation.cfm?id=2021039). 

It uses a truncated IBP prior that has an upperbound of the numebr of latent features K. The real-valued latent feactures was implemented using a slice sampling based on [Neal (2003)](https://www.jstor.org/stable/3448413?seq=1#page_scan_tab_contents). 

To test, run

```python
python demo.py
```

This demo is based on a simulated data set consisting of 6x6 images in Griffiths and Ghahramani (2011).

Default setting: use 100 6x6 images with an upperbound K = 6 via 1000 MCMC iterations. 

Resuting outputs will be saved in a separate folder and visualized using [David Andrzejewski's code](https://github.com/davidandrzej/PyIBP/blob/master/example/scaledimage.py) (Thanks!!).



