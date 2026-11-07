# Deep learning projects

Three projects from Deep Learning Fundamentals, a course in my Master of Data Science at the University of Adelaide (2024). Each one asks a question, builds neural networks to answer it, and reports honestly on what worked and what did not.

A neural network is a model that learns patterns from examples instead of following rules someone wrote by hand. Show it enough labelled examples and it keeps adjusting its internal values, sometimes millions of them, until its answers match. These projects cover three common kinds: a single neuron for yes-or-no questions, convolutional networks for images, and recurrent networks for data that unfolds over time.

## The projects

### [Can a single artificial neuron spot diabetes?](perceptron-diabetes-prediction/)

Medical records from 768 patients, eight measurements each, and the simplest network there is. It caught about half the diabetic patients in the test group and was right 71% of the time overall. Along the way I found that almost half the insulin readings were secretly missing, recorded as zeros.

`Perceptron` `Medical data` `Missing data` `Classification`

### [Three networks, one image test](cnn-image-classification-cifar10/)

Three image-recognition networks compete to sort 32 × 32 pixel photos into 10 classes, from airplanes to trucks. The winner, ResNet-18, labelled 4 out of 5 new pictures correctly. Its favourite mistake was calling a dog a cat.

`CNN` `ResNet-18` `AlexNet` `Image classification`

### [What I learned trying to predict Google's share price](rnn-stock-price-prediction/)

Three memory-based networks read 30 days of prices and guess the next day's. Their predictions stayed within a few dollars, but none could call the direction of the next move better than a coin flip, and tuning made the best one worse. This README is mostly about why.

`RNN` `LSTM` `GRU` `Time series`

## At a glance

| | Diabetes | Images | Share price |
|---|---|---|---|
| Data | 768 patients | 60,000 photos | 4,875 trading days |
| Networks compared | 1 | 3 | 3 |
| Final result | 71% accuracy | 81% accuracy | About \$2 to \$6 typical daily error |
| Biggest lesson | A small practice set flatters lucky runs | Design matters more than size | Low error is not the same as a useful forecast |

## What's in each folder

Every project has the same pieces:

- a **notebook** with the full analysis and all its outputs, readable directly on GitHub
- a **README** explaining the project in plain language, with technical details in a fold-out section
- the **paper** I submitted for the course, in `report/`
- the **charts** in `figures/` and the sources in `references.bib`

## Running the notebooks

You need Python 3.12. From the top folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open each notebook from inside its own folder. The diabetes project runs on any laptop. The image project needs a GPU (I used Google Colab). The share price project downloads its data, so it needs an internet connection.

## About me

**Rubaiyat Alam Ruhin**, Master of Data Science, University of Adelaide  
[LinkedIn](https://linkedin.com/in/rubaiyatalamruhin) · [GitHub](https://github.com/ruhinb) · rubaiyatalam.ruhin@gmail.com

My SQL and Power BI work is in [Data_Analysis_Projects](https://github.com/ruhinb/Data_Analysis_Projects).
