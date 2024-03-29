**Experiment 1**

- What is the spectral distribution of meat and fat respectively?
  - see inspect_distributions function
- Is it a reasonable assumption that each pixel is either meat or fat?

**Experiment 2**

- Calculate the threshold value t for all spectral bands for day 1.
  - DONE - see calculate_threshold()
- Calculate the error rate for each spectral band.
  - DONE - see model_metrics()
- Identify the spectral band, which has the best discriminative properties for meat and fat.
- Classify the entire image of the salami for day 1, and visualise it.

  - See single_band_exp function

**Experiment 3**

- Calculate the multivariate linear discriminant function as described in (23) for day 1.
  - Done (see multi_band_exp)
- Calculate the error rate (disagreement between the model and the annotations) for the training set.
  - Done (see model_metrics)
- Classify the entire image of the salami for day 1 and visualise it.

  - See multi_band_exp

- Classify fat and meat for the remaining days with the models trained on day 1.
- Calculate the error rate for the annotated areas for the remaining days. How is the performance of the two models?
- Classify again the entire images for the remaining days and, and visualise them. Judged on the
  visualisations, which model performs best?
- Judging from the images, which day would you choose to train on and why??

**Experiment 4**

- CODE DONE: SEE both_models_all_images_exp()
- Train the linear discriminant function on day 1, day 6 and so on, and for each day calculate the
  error rate on all the other days.
- Show the error rate for all days (5 training days x 4 test days = 20 error rates) in an appropriate
  plot or table.
- Why do we exclude the day we have trained on from the comparison?
- Which day is the best to train the model on, and why?
- What are the error sources in the model? Can we trust that the calculated fat and meat content in
  the salami is the same as the real content?

**Experiment 5**

- Redo experiment 4 but with a prior
  - Prior already implemented
- Does it change your estimates?
