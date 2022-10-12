## Лабораторная работа № 1 «Классификация изображений с помощью сверточных нейронных сетей»

<br/>

"Source of fruits is my own garden and nearby stores."

* Обучите два и более классификатора на наборе данных [Fruits 360](https://www.kaggle.com/datasets/moltean/fruits)
* Сравните результаты полученных моделей

Fruits were planted in the shaft of a low speed motor (3 rpm) and a short movie of 20 seconds was recorded. A Logitech
C920 camera was used for filming the fruits. This is one of the best webcams available. Behind the fruits we placed a
white sheet of paper as background. However due to the variations in the lighting conditions, the background was not
uniform and we wrote a dedicated algorithm which extract the fruit from the background. This algorithm is of flood fill
type: we start from each edge of the image and we mark all pixels there, then we mark all pixels found in the
neighborhood of the already marked pixels for which the distance between colors is less than a prescribed value. We
repeat the previous step until no more pixels can be marked. All marked pixels are considered as being background (which
is then filled with white) and the rest of pixels are considered as belonging to the object. The maximum value for the
distance between 2 neighbor pixels is a parameter of the algorithm and is set (by trial and error) for each movie.
Pictures from the test-multiple_fruits folder were made with a Nexus 5X phone.