# A Linear-Gaussin Latent Factor (Feature) Model using a IBP prior

This Python code implements a linear-Gaussian latent factor model using a Indian buffet process prior as illustrated in [Griffiths and Ghahramani (2011)](https://cocosci.berkeley.edu/tom/papers/indianbuffet.pdf). 

It uses a truncated IBP prior that has an upperbound of the numebr of latent features K.The real-valued latent feactures was implemented using a slice sampling based on [Neal (2003)](https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461). 

To test

```python
python demo.py
```

This demo is based on a simulated data set consisting of 6x6 images in Griffiths and Ghahramani (2011).

Default setting: utilize 100 6x6 images, 1000 MCMC iterations with an upperbound K = 6. 

Resuting outputs will be saved in a separate folder and visualized using [David Andrzejewski's code](https://github.com/davidandrzej/PyIBP/blob/master/example/scaledimage.py). 

