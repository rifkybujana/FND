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

FND is a machine learning project that were made to predict whether a news is fake or not. This project are using IndoBERT model trained by IndoLEM Team and fine-tuned on 3 different indonesian fake news dataset source with total 2220 news. And get 86.7% Validation Accuracy with 0.25 Validation Loss.

[How to use it](#usage)


### Built With

* [Streamlit](https://www.streamlit.io/)
* [Transformers](https://github.com/huggingface/transformers)
* [Newspaper3k](https://newspaper.readthedocs.io/en/latest/)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

You can easly install all the required package with:
```sh
$ python setup.py install
```
  
### Installation

1. Clone the repo
   ```sh
   $ git clone https://github.com/rifkybujana/FND.git
   ```



<!-- USAGE -->
## Usage

<small>Note: CLI Support comming soon</small>

```
$ streamlit run App.py
```



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

Distributed under the GPL-3.0 License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

* Rifky Bujana Bisri - [@rifkybujanabisri](https://www.instagram.com/rifkybujanabisri/) - rifkybujanabisri@gmail.com
* Aikyo Dzaki Aroef - [@aikibot](https://www.instagram.com/aikibot/) - aikyodzakiaroef@gmail.com

Project Link: [https://github.com/rifkybujana/FND](https://github.com/rifkybujana/FND)



## Acknowledgements

Special Thanks to Intel, Orbit Future Academy and Indonesian Ministry of Education and Culture for holding AI For Youth Indonesia, because without it we can't make this project real or even we won't know anything about machine learning.

* [kaggle](https://www.kaggle.com/c/fake-news/data)
* [RAHUTOMO, FAISAL; Yanuar, Inggrid; ANDRIE ASMARA, ROSA (2018), “INDONESIAN HOAX NEWS DETECTION DATASET”, Mendeley Data, V1, doi: 10.17632/p3hfgr5j3m.1](http://dx.doi.org/10.17632/p3hfgr5j3m.1)
* [Fajri Koto, Afshin Rahimi, Jey Han Lau, Timothy Baldwin](https://indolem.github.io/)




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
