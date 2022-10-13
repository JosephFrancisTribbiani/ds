# [Структуры данных](https://informatics.msk.ru/course/view.php?id=18#section-2)

## Тема 2. Куча и приоритетная очередь.

### [Задача №1164. Увеличение приоритета](https://informatics.msk.ru/mod/statements/view.php?id=1234#1)

См. <a href="https://informatics.msk.ru/mod/resource/view.php?id=1230">«Двоичная куча (пирамида). Пирамидальная сортировка. Приоритетная очередь» (PDF)</a>, стр. 9, задача 1.

Ограничение времени – 2 секунды

[Решение python](./topic_2/task_1164_increment.py)

### [Задача №1165. Уменьшение приоритета](https://informatics.msk.ru/mod/statements/view.php?id=1234&chapterid=1165#1)

См. <a href="https://informatics.msk.ru/mod/resource/view.php?id=1230">«Двоичная куча (пирамида). Пирамидальная сортировка. Приоритетная очередь» (PDF)</a>,, стр. 10, задача 2.

Ограничение времени – 2 секунды

[Решение python](./topic_2/task_1165_decrement.py)

### [Задача №1166. Извлечение максимального](https://informatics.msk.ru/mod/statements/view.php?chapterid=1166#1)
См. <a href="https://informatics.msk.ru/mod/resource/view.php?id=1230">«Двоичная куча (пирамида). Пирамидальная сортировка. Приоритетная очередь» (PDF)</a>, стр. 10, задача 3.

Ограничение времени – 1 секунда

[Решение python](./topic_2/task_1166_extract_max.py)

### [Задача №1167. Приоритетная очередь](https://informatics.msk.ru/mod/statements/view.php?id=1234&chapterid=1167#1)
См. <a href="https://informatics.msk.ru/mod/resource/view.php?id=1230">«Двоичная куча (пирамида). Пирамидальная сортировка. Приоритетная очередь» (PDF)</a>, стр. 11, задача 4.

Ограничение времени – 2 секунды

[Решение python](./topic_2/task_1167_priority_queue.py)

### [Задача №1168. Приоритетная очередь с удалением](https://informatics.msk.ru/mod/statements/view.php?id=1234&chapterid=1168#1)
См. <a href="https://informatics.msk.ru/mod/resource/view.php?id=1230">«Двоичная куча (пирамида). Пирамидальная сортировка. Приоритетная очередь» (PDF)</a>, стр. 12, задача 5.

Ограничение времени – 2 секунды

[Решение python](./topic_2/task_1168_priority_queue_with_removal.py) _Пройдены не все тесты!_

### [Задача №1169. Построение кучи просеиванием вверх](https://informatics.msk.ru/mod/statements/view.php?id=1234&chapterid=1169#1)
См. <a href="https://informatics.msk.ru/mod/resource/view.php?id=1230">«Двоичная куча (пирамида). Пирамидальная сортировка. Приоритетная очередь» (PDF)</a>, стр. 13, задача 6.

Ограничение времени – 1 секунда

[Решение python](./topic_2/task_1169_build_heap_shift_up.py)

### [Задача №1170. Построение кучи просеиванием вниз](https://informatics.msk.ru/mod/statements/view.php?id=1234&chapterid=1170#1)
См. <a href="https://informatics.msk.ru/mod/resource/view.php?id=1230">«Двоичная куча (пирамида). Пирамидальная сортировка. Приоритетная очередь» (PDF)</a>, стр. 13, задача 7.

Ограничение времени – 1 секунда

[Решение python](./topic_2/task_1170_build_heap_shift_down.py)

В первой реализации (на основе просеивания вверх) выполняется много просеиваний на большую глубину, и мало просеиваний на маленькую глубину. Во второй реализации, наоборот, просеиваний на большую глубину выполняется мало, а на маленькую – много. Поэтому разумно предположить (и можно доказать), что вторая реализация работает быстрее.

Можно доказать (см. например, книгу Кормена и др.), что вторая реализация (на основе просеивания вниз) всегда работает за время $ \Theta(n) $, а первая в худшем случае – за $ \Theta(n \log{n}) $. 

### [Задача №1171. Пирамидальная сортировка - подробно](https://informatics.msk.ru/mod/statements/view.php?id=1234&chapterid=1171#1)
См. <a href="https://informatics.msk.ru/mod/resource/view.php?id=1230">«Двоичная куча (пирамида). Пирамидальная сортировка. Приоритетная очередь» (PDF)</a>, стр. 13, задача 8.

Ограничение времени – 1 секунда

[Решение python](./topic_2/task_1171_detailed_pyramid_sort.py)
