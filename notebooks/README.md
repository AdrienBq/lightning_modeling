# Notebooks

This folder presents some notebooks to reproduce and extend some results from the paper.

--- 
## 01 - Plot Events

The first notebooks is used to compute the probabilities of lightning for each hour and location of each extreme day in the test dataset (2008, 2015, 2023).

The probabilities are aggregated at the daily scale to show the total numer of hours of lightning observed during the day by ATDnet; and the expected number of hours of lightning predicted during the day by the different models.

They are then plotted similarly to what is show for the three examples in the paper.

---
## 02 - Evaluate Models

The second notebook is used to evaluate the models on the extreme events. It is a reproduction of the "Tail" columns of Figure 3 in the paper.

---
## 03 Interpretability

The third notebook is a quick peak at one interpretability method used in the paper : the saliency maps.
The notebook allows to comput the saliency map for a given event, at a given time and a given location.
It is possible to play with the event time and location to get an idea of how the saliency maps compare when we change those features.
