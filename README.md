# Система видеоаналитики для управления очередью

## Набор файлов

* dataset/train/images/*.jpg - фото с камер (для тренировки модели)

* dataset/val/images/*.jpg - фото с камер (для теста модели)

* dataset/train/labels/*.txt - координаты людей на фото (для тренировки модели)

* dataset/val/labels/*.txt - координаты людей на фото (для теста модели)

* templates/index.html - HTML файл главной страницы

* video/*.mp4 - видео с камер

* .gitignore - файл gitignore

* data.yaml - файл с путями до изображений и именами классов

* detect.pt - обученная модель YOLOv8

* main.py - главный файл, запускающий работу приложения

* README.md - описание проекта

* train.ipynb - файл для запуска обучения модели
