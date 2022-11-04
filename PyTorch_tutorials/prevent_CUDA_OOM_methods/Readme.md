В данном разделе представлены туториалы по ускорению и оптимизации работы CUDA, в том числе с целью избежания ошибки переполнения памяти видеокарты. Туториалы подготовлены на основе очень полезных статей:

1. [How to use Autocast in PyTorch](https://wandb.ai/wandb_fc/tips/reports/How-to-use-Autocast-in-PyTorch--VmlldzoyMTk4NTky)

Данный метод позволяет автоматически преобразовывать градиенты из `float32` в `float16`, к примеру. При использовании данного метода объем занимаемой памяти уменьшается.

2. [How to Use GradScaler in PyTorch](https://wandb.ai/wandb_fc/tips/reports/How-to-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5)

В случае использования метода из п. 1, возможно обнуление очень маленьких по величине градиентов (к примеру, количество знаков после запятой для `float16` недостаточно, и градиенты обнуляются), что в свою очередь останавливает процесс обучения и обновления весов модели. `GradScaler` позволяет автоматически масштабировать градиенты, что в свою очередь препятствует их обнулению.

3. [Preventing The CUDA Out Of Memory Error In PyTorch](https://wandb.ai/wandb_fc/tips/reports/Preventing-The-CUDA-Out-Of-Memory-Error-In-PyTorch--VmlldzoxNzU3NjA1?galleryTag=general)

В данной статье представлены некоторые дополнительные методы для борьбы с CUDA ООМ (out of memory):
- контроль используемой памяти
- очистка cache и использование `gc` (garbage collector)
- настройка параметров (`batch_size`, `num_workers`)