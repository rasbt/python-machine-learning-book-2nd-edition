Dear Readers, 

again, I tried my best to cut all the little typos, errors, and formatting bugs that slipped through the copy editing stage. Even so, while I think it is just human to have a little typo here and there, I know that this can be quite annoying as a reader!

To turn those annoyances into something positive, I will donate $5 to [UNICEF USA](https://www.unicefusa.org), the US branch of the United Nations agency for raising funds to provide emergency food and healthcare for children in developing countries, for each little unreported buglet you find!

Also below, I added a small leaderboard to keep track of the errata submissions and errors you found. Please let me know if you don't want to be explicitely mentioned in that list! 


- Amount for the next donation: 20$
- Amount donated: 0$


---


Contributor list:


1. Gogy ($10)
2. Christian Geier ($5)
3. Pieter Algra / Carlos Zada ($5)


<br>
<br>
<br>
<br>





---

### Errata

pg. 55

![](./images/pg55.png)

pg. 91

On the top of the page, it says "Here, p (i | t ) is the proportion of the samples that belong to class c." The "*c*" should be changed to *i*.

pg. 136

The print version is incorrectly shows 

```python
>>> plt.xticks(range(X_train.shape[1]),
...            feat_labels, rotation=90)
```

instead of 

```python
>>> plt.xticks(range(X_train.shape[1]),
...            feat_labels[indices], rotation=90)
```

It seems that I did it correctly in the notebook. Also, the list of feature importances and the plot seem to be correct in the book. However, somehow the [indices] array index went missing in the print version.


pg. 155

![](./images/pg155.png)
