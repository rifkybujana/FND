[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License: MIT][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/rifkybujana/FND">
    <img src="https://i.imgur.com/Wkx9XUI.png" width="180" height="140">
  </a>

  <h3 align="center">FND</h3>

  <p align="center">
    Fake News Detection AI
    <br />
    <a href="https://github.com/rifkybujana/FND/issues">Report Bug</a>
    ·
    <a href="https://github.com/rifkybujana/FND/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![screenshot](https://i.imgur.com/tp7hOov.png)
FND is an web application that can be used to predict whether a news is hoax or not. This app support 2 language, which is english and indonesian language.
the english logistic regression model have been trained with 5000 hoax news and valid news data from [kaggle](https://www.kaggle.com/c/fake-news/data) that give 93%-95% accuracy from the testing dataset. But the indonesian model is just trained with 600 hoax news dataset by [Faisal Rahutomo, Inggrid Yanuar, Rosa Andrie Asmara](https://data.mendeley.com/datasets/p3hfgr5j3m/1) that give us 77% accuracy from the testing dataset.



### Built With

* [Streamlit](https://www.streamlit.io/)
* [Sastrawi](https://sastrawi.github.io/)
* [NLTK](https://www.nltk.org/)
* [Pandas](https://pandas.pydata.org/)
* [Joblib](https://joblib.readthedocs.io/en/latest/)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

You can easly install all the required package with:
```sh
$ pip install -r requirements.txt && python setup.py install
$ python
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
```
  
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/rifkybujana/FND.git
   ```
2. Run this webapp in your localhost by simply:
   ```sh
   streamlit run app.py
   ```



<!-- USAGE -->
## Usage

To use this app to predict whether a news is real or not, simply just put a text of an article into the textbox, or you can write it in a file without any paragraf (just 1 line). Or you can put a list of article separated by enter (1 line for every article), hit enter and wait until the process is done.

```Note: Dont forget to make sure that you're using the correct language settings in the sidebar```




<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/rifkybujana/FND/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

* Rifky Bujana Bisri - [@rifkybujanabisri](https://www.instagram.com/rifkybujanabisri/) - rifkybujanabisri@gmail.com
* Aikyo Dzaki Aroef - [@aikibot](https://www.instagram.com/aikibot/) - aikyodzakiaroef@gmail.com

Project Link: [https://github.com/rifkybujana/FND](https://github.com/rifkybujana/FND)



## Acknowledgements

* [kaggle](https://www.kaggle.com/c/fake-news/data)
* [RAHUTOMO, FAISAL; Yanuar, Inggrid; ANDRIE ASMARA, ROSA (2018), “INDONESIAN HOAX NEWS DETECTION DATASET”, Mendeley Data, V1, doi: 10.17632/p3hfgr5j3m.1](10.17632/p3hfgr5j3m.1)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/rifkybujana/FND.svg?style=for-the-badge
[contributors-url]: https://github.com/rifkybujana/FND/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/rifkybujana/FND.svg?style=for-the-badge
[forks-url]: https://github.com/rifkybujana/FND/network/members
[stars-shield]: https://img.shields.io/github/stars/rifkybujana/FND.svg?style=for-the-badge
[stars-url]: https://github.com/rifkybujana/FND/stargazers
[issues-shield]: https://img.shields.io/github/issues/rifkybujana/FND.svg?style=for-the-badge
[issues-url]: https://github.com/rifkybujana/FND/issues
[license-shield]: https://img.shields.io/badge/License-GNU-yellow.svg?style=for-the-badge
[license-url]: ./LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/rifkybujana
