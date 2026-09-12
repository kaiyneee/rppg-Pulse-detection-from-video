# Ограничения на использование UBFC-rPPG

Источник: `readme.txt` и `Agreement.xlsx`, приложенные к датасету.

## Запрет на публикацию изображений лиц

> DATASET_2: For subjects 21 and 27: Facial images must not be included in any
> publication, presentation, or report.

**Практически:** данные `subject21` и `subject27` использовать в выборке и в
метриках можно — запрет касается только изображений. Ни один кадр их лица не
может попасть в статью, презентацию, отчёт или слайды. Если для иллюстрации
областей интереса (ROI), карты кожи, оверлея landmark'ов или примера сигнала
нужен кадр — брать **другого** испытуемого.

В выборке этого проекта `subject27` присутствует (`subject21` не скачан).
`subject27` при этом входит в число записей с худшими метриками, то есть
именно та запись, которую было бы естественно показать как пример трудного
случая — показывать её кадром **нельзя**, только числами и графиками.

Скрипты репозитория кадры на диск не сохраняют (проверено: `imwrite`
не вызывается нигде), так что автоматически нарушить запрет нечем — риск
только в ручном скриншоте.

## Цитирование

Авторы датасета просят ссылаться так (формулировка из `readme.txt` и шапки
`ubfcrppg_data_processor.py`):

> S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, Unsupervised skin
> tissue segmentation for remote photoplethysmography, Pattern Recognition
> Letters, Elsevier, 2017.

Год именно **2017** (в `docs/research_report.md` раньше стоял 2019 — исправлено).

## Эталон

Авторы прямо указывают, что колонка ЧСС прибора не использовалась для оценки:

> Please note that heart rate values provided by the sensor were not used for
> evaluating our heart rate estimation method. Instead, our evaluation compared
> heart rate estimations derived from the remote PPG signal with estimations
> calculated from the contact PPG signal.

Поэтому эталоном служит контактная пульсовая ВОЛНА, а ЧСС прибора остаётся в
пооконных логах только для сравнения (см. `scripts/eval_ubfc_subject.py`,
`reference_bpm_from_wave`, задача 53).
