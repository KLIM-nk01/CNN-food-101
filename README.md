# Лабораторная работа №3

## 1. С использованием техники обучения Transfer Learning обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101 с использованием фиксированных темпов обучения 0.01, 0.001, 0.0001

### Графики обучения:
- Красный - темп 0.01 на обучении / Голубой - темп 0.01 на валидации
- Оранжевый - темп 0.001 на обучении / Синий - темп 0.001 на валидации
- Розовый - темп 0.0001 на обучении / Зеленый - темп 0.0001 на валидации

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/epoch_categorical_accuracy.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/epoch_loss.svg)

## 2. Реализовать и применить в обучении Косинусное затухание (Cosine Decay), а также определить оптимальные параметры для данной политики

### Графики обучения:
- Розовый - темп 0.01 на обучении / Зеленый - темп 0.01 на валидации
- Красный - темп 0.001 на обучении / Голубой - темп 0.001 на валидации
- Оранжевый - темп 0.0001 на обучении / Синий - темп 0.0001 на валидации

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/epoch_categorical_accuracy_cos.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/epoch_loss_cos.svg)

## 3. Реализовать и применить в обучении Косинусное затухание с перезапусками (Cosine Decay with Restarts), а также определить оптимальные параметры для данной политики

### Графики обучения:
- Розовый - темп 0.01 на обучении / Зеленый - темп 0.01 на валидации
- Красный - темп 0.001 на обучении / Голубой - темп 0.001 на валидации
- Оранжевый - темп 0.0001 на обучении / Синий - темп 0.0001 на валидации

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/epoch_categorical_accuracy_cos_res.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/epoch_loss_cos_res.svg)

## Анализ результатов
Если судить по выборке обучения то оптимальным фиксированным темпом обучения является 0.001, т.к точность на обучении достигает 83.77 %, при потерях 0.5468. При это стоит заметить, что при валидации лучшее значение получается при фиксированном темпе обучения 0.0001, которое равно 67.16% при потерях 1.222.

При использования политики косинусного затухания мы использовали функцию:  
`tf.keras.experimental.CosineDecay(`  
`    initial_learning_rate, decay_steps, alpha=0.0, name=None`  
`)`  
Где initial_learning_rate подбирался(0.01, 0.001, 0.0001), decay_steps был взят равный 1000, а alpha - ограничение шагов - равным 0. При таких значения прослеживалась следующая тенденция: на валидации лучше значение равное 67.25%  при потерях 1.221 было полученно когда initial_learning_rate = 0.0001, а на обучении лучшее значение равное 83.81% при потерях 0.5446 было полученно когда initial_learning_rate = 0.001.  
