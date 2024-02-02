<br/>
<div align="center" >

![Logo ENSC](images/ENSC.png)


# <u> ENSC Parcours IA </u>
## Data Challenge - Détection de clics d'odontocètes

</div>

## ✍️ Authors

- [Tristan Gonçalves](https://github.com/tristangclvs)
- [Thomas Chimbault](https://github.com/thomaschlt)

## 🐬 Context 

As part of the [Artificial Intelligence specialization](https://3aia.notion.site/3aia/Parcours-3A-IA-2023-9917027c682b457dae71fea68c067ad1) at the [ENSC](https://ensc.bordeaux-inp.fr/fr), we participated in a data challenge provided by the University of Toulon in the [ChallengeData](https://challengedata.ens.fr/) website. 

This challenge specifically aims to detect the presence of odontoceti clicks in underwater audio recordings in the Caribbean sea.
The model will be evaluated on the [ChallengeData](https://challengedata.ens.fr/) website.

## 📊 Data Description

The dataset is composed of 23,168 audio files in WAV format, each of duration 200ms. The clicks are labeled with a binary variable: 1 if the file contains a click, 0 otherwise.

## 🎯 Challenge

The objective of the challenge is to create a model that predicts the presence of odontoceti clicks in the test set with the highest accuracy.

## 📏 Evaluation

The submissions are evaluated on the ROC AUC (area under the curve) metric.

The results must be submitted as a CSV file with 950 lines. Each line corresponds to a file of the test set and contains the prediction for this file. The prediction in percentage should be indicated and must not be rounded to binary labels.

## 🔎 Our approach

We first used classical machine learning model, such as `Linear Regression` or `Random Forest`.

Then, we used the [`ReservoirPy`](https://github.com/reservoirpy/reservoirpy) library, created and maintained by the Inria of Bordeaux. This uses the reservoir computing theory, part of the RNN's domain.

We also used a Convolutional Neural Network to classify the audio files. We used the [Librosa](https://librosa.org/doc/latest/index.html) library to extract the audio features.

## Results
### Classical approaches

<div align="center">
<table>
    <tr>
        <th>Method</th>
        <th>Result</th>
    </tr>
    <tr>
        <td>Logistic Regression</td>
        <td> <b>0.5981</b> </td>
    </tr>
    <tr>
        <td>Decision  Tree</td>
        <td> <b>0.6124</b> </td>
    </tr>
    <tr>
        <td>Bagged Tree</td>
        <td> <b>0.6351</b> </td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td> <b>0.6460</b> </td>
    </tr>
    <tr>
        <td>XGBoost</td>
        <td> <b>0.6301</b> </td>
    </tr>
</table>
</div>

### Reservoir computing

We didn't get the result we expected by using reservoir computing. In fact, the issue we got resided in the format of the results.
As a matter of fact, ReservoirPy's results were not probabilities and could be above 1 or below 0. Therefore, we had to put a threshold and round these extreme values to either 1 or 0. This distorted our results and we obtained a score of only <b>0.48</b>. However, this method has been implemented very late during the project, so we maybe have wrongfully used this library.

### Convolution Networks
#### Conv1D


Final Score: **0.9566**


#### Conv2D

We tried to use 2D convolutions because it is well-known that audio files can be represented with spectrograms. \
However, results were really not convincing, in regard of the energetic consumption of our model training. Indeed, our project being related to the study of underwater life, we thought that having a very consuming and heavy model was totally inappropriate.

For information, our best result there was <b>0.86</b>.

## 💻 Installation

First of all, you may clone this repository on your computer.

```bash
git clone https://github.com/tristangclvs/spe_ia_clics_odontocetes.git
```

Then, download the `.dataset` archive [here](https://drive.google.com/file/d/1gNyw2PcUCYmpCm8lNTyPJ_ydeLdbDQiw/view?usp=sharing) and extract it in the main root of the cloned folder.

<details>
  <summary>Creating a <u><b>virtual environment</b></u></summary>


>You may want to create a virtual environment for python. \
>```bash
>python -m venv NameOfYourEnv
>```
>Then select your environment:
>
> ### ⊞ Windows:
>```bash
>NameOfYourEnv/Scripts/activate
>```
> ### 🍏 Mac:
>```bash
>source NameOfYourEnv/bin/activate
>``` 
> <br>

</details>

<br>

To run the code in this repository, you will need to install the necessary dependencies:
```bash
pip install -r requirements.txt
```

##  Repository Structure

The repository is structured as follows:

<!-- - **`.dataset`**: contains the training and test sets used for the challenge. -->
- **[`CNN_topic`](/CNN_topic/)**: contains the files used to train the convolutional neural networks (1D & 2D).
- **[`images`](/images/)**: contains the images used in the README file and a notebook for plotting some charts.
- **[`notebooks`](/notebooks/)**: contains the notebooks used for the challenge.

