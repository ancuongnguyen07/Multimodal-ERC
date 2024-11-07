This repo hosts the course project for multi-modal speech recognition

### Introduction
The aim of this project is to utilize the [MELD](https://github.com/declare-lab/MELD/tree/master?tab=readme-ov-file)
dataset and multi-modal emotion classifier to recognize emotions
caused in daily conversations.

References:
- [A review of Multi-modal Emotion Recognition Systems](https://www.sciencedirect.com/science/article/pii/S092523122300989X?via%3Dihub)
- [MELD dataset](https://arxiv.org/pdf/1810.02508)

### Data Exploration
- Download extracted audio and text features here: http://bit.ly/MELD-features
- Download extracted video features here: https://drive.google.com/file/d/1RjrYSMpXxg_6r_nUQaysaPyMsldLpMcb/view?usp=sharing

TODO!

### How to build the LaTeX report
Make sure that you have TeX packages installed in your machine.
Then run
```sh
cd report # change directory to `report`
make
```

The `report.pdf` should be in `report/build/final/report.pdf`.
In case you could not build the PDF file in your machine,
the up-to-date PDF report files are at `report/pdf`
