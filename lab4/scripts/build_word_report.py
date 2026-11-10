"""Create a standalone LR4 report using the formatting of the supplied LR2 example."""
import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = Path('/home/tair/NETI/MAD/lab2/Отчёт_ЛР2_Классификация.docx')


def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def number(value, digits=4):
    return f'{float(value):.{digits}f}'.replace('.', ',')


class Report:
    def __init__(self, template):
        sample = Document(template)
        self.doc = Document()
        self.table_counter = 0
        self.figure_counter = 0
        section = self.doc.sections[0]
        original = sample.sections[0]
        for attr in ('page_width', 'page_height', 'top_margin', 'bottom_margin',
                     'left_margin', 'right_margin', 'header_distance', 'footer_distance'):
            setattr(section, attr, getattr(original, attr))
        section.different_first_page_header_footer = True
        self.width = (section.page_width - section.left_margin - section.right_margin) / 360000
        for name in ('Normal', 'Heading 1', 'Heading 2', 'Heading 3'):
            source = sample.styles[name]
            target = self.doc.styles[name]
            for tag in ('w:pPr', 'w:rPr'):
                old = target.element.find(qn(tag))
                if old is not None:
                    target.element.remove(old)
                element = source.element.find(qn(tag))
                if element is not None:
                    target.element.append(deepcopy(element))
            # Copied headings inherit from the Normal style in this new document.
            if name != 'Normal':
                target.base_style = self.doc.styles['Normal']
                target.next_paragraph_style = self.doc.styles['Normal']
            target.font.name = 'Times New Roman'
            target.font.size = Pt(14)
            target.font.color.rgb = RGBColor(0, 0, 0)
        normal = self.doc.styles['Normal']
        normal.paragraph_format.widow_control = True
        for name, size, font in [('Report Code', 8.5, 'Courier New'), ('Report Caption', 12, 'Times New Roman')]:
            style = self.doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            style.base_style = normal
            style.font.name = font
            style.font.size = Pt(size)
            fmt = style.paragraph_format
            fmt.first_line_indent = Cm(0)
            fmt.space_after = Pt(0)
            fmt.space_before = Pt(0)
            fmt.line_spacing = 1
            fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT
        self.doc.styles['Report Caption'].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.doc.styles['Report Caption'].paragraph_format.space_after = Pt(6)
        # Repeating PAGE field, absent on the cover.
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer.paragraph_format.first_line_indent = Cm(0)
        footer.paragraph_format.line_spacing = 1
        run = footer.add_run()
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        for kind, text in [('begin', None), (None, ' PAGE '), ('end', None)]:
            field = OxmlElement('w:instrText' if kind is None else 'w:fldChar')
            if kind:
                field.set(qn('w:fldCharType'), kind)
            else:
                field.set(qn('xml:space'), 'preserve')
                field.text = text
            run._r.append(field)
        settings = self.doc.settings.element
        update = OxmlElement('w:updateFields')
        update.set(qn('w:val'), 'true')
        settings.append(update)
        # Keep screenshot details and original PNGs when Word updates the document.
        no_compression = OxmlElement('w:doNotAutoCompressPictures')
        settings.append(no_compression)
        self.doc.core_properties.title = 'Лабораторная работа № 4. MLOps. Вариант 2'
        self.doc.core_properties.subject = 'Сервис фильтрации SMS-спама'
        self.doc.core_properties.author = 'Бикмухаметов Таир Олегович'
        self.doc.core_properties.keywords = 'MLOps, MLflow, SMS Spam Collection, Docker, вариант 2'

    def p(self, text='', *, style=None):
        return self.doc.add_paragraph(text, style=style)

    def heading(self, text, level=1, page=False):
        p = self.doc.add_paragraph(text, style=f'Heading {level}')
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(8)
        if page:
            p.paragraph_format.page_break_before = True
        return p

    def code(self, text):
        text = text.strip('\n')
        for line in text.splitlines():
            p = self.p(line or ' ', style='Report Code')
            p.paragraph_format.keep_together = True
        self.p('').paragraph_format.line_spacing = 0.5

    def table(self, title, headers, rows, weights=None, size=11, keep_rows=False):
        self.table_counter += 1
        caption = self.p(f'Таблица {self.table_counter}. {title}')
        fmt = caption.paragraph_format
        fmt.first_line_indent = Cm(0)
        fmt.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        fmt.keep_with_next = True
        fmt.space_before = Pt(7)
        fmt.space_after = Pt(4)
        fmt.line_spacing = 1
        for run in caption.runs:
            run.font.size = Pt(12)
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.style = 'Table Grid'
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        weights = weights or [1] * len(headers)
        widths = [self.width * weight / sum(weights) for weight in weights]
        for col, width in zip(table.columns, widths):
            col.width = Cm(width)
        content = [headers] + [list(row) for row in rows]
        for index, values in enumerate(content):
            row = table.rows[0] if index == 0 else table.add_row()
            trpr = row._tr.get_or_add_trPr()
            trpr.append(OxmlElement('w:cantSplit'))
            if index == 0:
                trpr.append(OxmlElement('w:tblHeader'))
            for cell, value, width in zip(row.cells, values, widths):
                cell.width = Cm(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cell.text = str(value)
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent = Cm(0)
                    p.paragraph_format.line_spacing = 1
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)
                    if keep_rows:
                        p.paragraph_format.keep_with_next = index < len(content) - 1
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for run in p.runs:
                        run.font.name = 'Times New Roman'
                        run.font.size = Pt(size)
                        run.bold = index == 0
        self.p('').paragraph_format.line_spacing = 0.5
        return table

    def figure(self, name, caption, *, crop=None, width=None, max_height=19):
        path = ROOT / name
        with Image.open(path) as im:
            source_width, source_height = im.size
        left, top, right, bottom = crop or (0, 0, 1, 1)
        ratio = source_height * (bottom - top) / (source_width * (right - left))
        width = min(width or self.width, self.width, max_height / ratio)
        height = width * ratio
        p = self.p()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.line_spacing = 1
        p.paragraph_format.keep_with_next = True
        picture = p.add_run().add_picture(str(path), width=Cm(width), height=Cm(height))
        if crop:
            # Native Word crop: retain the original screenshot bitmap unchanged.
            fill = picture._inline.graphic.graphicData.pic.blipFill
            rect = OxmlElement('a:srcRect')
            for key, value in [('l', left), ('t', top), ('r', 1 - right), ('b', 1 - bottom)]:
                rect.set(key, str(round(value * 100000)))
            fill.insert(1, rect)
        self.figure_counter += 1
        self.p(f'Рис. {self.figure_counter} – {caption}', style='Report Caption')

    def cover(self):
        def line(text='', size=12, bold=False, before=0, after=0):
            p = self.p()
            fmt = p.paragraph_format
            fmt.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt.first_line_indent = Cm(0)
            fmt.line_spacing = 1
            fmt.space_before, fmt.space_after = Pt(before), Pt(after)
            run = p.add_run(text)
            run.font.name, run.font.size, run.bold = 'Times New Roman', Pt(size), bold
            return p
        line('Министерство науки и высшего образования РФ', after=12)
        line('Федеральное государственное бюджетное образовательное учреждение')
        line('высшего образования')
        line('«Новосибирский государственный технический университет»', after=12)
        line('Кафедра Автоматизированных систем управления')
        line('ЛАБОРАТОРНАЯ РАБОТА № 4', size=14, bold=True, before=72, after=5)
        line('«MLOps»', size=14, bold=True, after=5)
        line('по дисциплине «Методы анализа данных»')
        line('Вариант 2. Сервис фильтрации SMS-спама', after=48)
        table = self.doc.add_table(rows=1, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        for c in table.columns:
            c.width = Cm(self.width / 2)
        authors = ['Выполнил:', '', 'Бикмухаметов Таир Олегович', '', 'Группа  АВТ-314', '', 'Факультет  АВТФ']
        teacher = ['Проверил:', '', 'Ганелина Наталья Давидовна']
        for cell, lines in zip(table.rows[0].cells, [authors, teacher]):
            cell.text = '\n'.join(lines)
            for p in cell.paragraphs:
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.line_spacing = 1
                p.paragraph_format.space_after = Pt(0)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.name, r.font.size = 'Times New Roman', Pt(12)
        line('Новосибирск 2026', before=300)
        self.doc.add_page_break()


def build(template, output):
    summary, final = load('results/summary.json'), load('results/test_metrics.json')
    docker, local = load('results/docker_checks.json'), load('results/local_checks.json')
    data = load('data/manifest.json')
    with (ROOT / 'results/experiments.csv').open() as f:
        experiments = list(csv.DictReader(f))
    if docker['status'] != 'passed':
        raise RuntimeError('A successful Docker protocol is required for this report')
    report = Report(template)
    p, h, code, table, figure = report.p, report.heading, report.code, report.table, report.figure
    report.cover()

    h('1 Цель работы и постановка задачи')
    p('Цель работы – освоить основы MLOps: учёт и сравнение экспериментов в MLflow, '
      'регистрацию версий моделей, развёртывание API с использованием Docker и Docker Compose, '
      'проверку работоспособности, обновление и откат модели без повторного обучения.')
    p('Индивидуальный вариант 2 – сервис фильтрации SMS-спама. По тексту сообщения необходимо '
      'определить класс spam или ham. Начальные алгоритмы – TF-IDF + MultinomialNB и '
      'TF-IDF + LogisticRegression. В экспериментах изменяются alpha и C. Основной критерий '
      'выбора – F1 для spam; дополнительно сравниваются precision и recall этого класса.')
    p('Требуется провести не менее шести экспериментов, сохранить параметры, метрики, теги '
      'и артефакты, зарегистрировать две версии одной модели и назначить alias champion. '
      'Затем необходимо экспортировать Pipeline, реализовать API, собрать Docker-образы, '
      'проверить корректные и некорректные запросы, показать обновление и откат, '
      'проверить healthcheck в Compose и зафиксировать материалы для воспроизведения.')

    h('2 Описание набора данных и подготовка')
    p('Использован SMS Spam Collection из UCI Machine Learning Repository [1]. Авторы – '
      'Tiago Almeida и José Hidalgo; DOI: 10.24432/C5CC84. Набор распространяется '
      'по лицензии CC BY 4.0. Источник и исходные байты сохранены в проекте.')
    table('Признаки и целевая переменная', ['Поле', 'Описание'], [
        ('text', 'Исходный текст SMS; единственный входной признак.'),
        ('label / target', 'Целевой класс: ham = 0, spam = 1.'),
        ('row_id', 'Номер строки исходного файла; используется для разбиения, в модель не передаётся.'),
        ('text_hash', 'SHA-256 нормализованного текста; используется для контроля повторов.')], [1, 3])
    p('Файл SMSSpamCollection прочитан разделением каждой непустой строки по первой табуляции. '
      'Кавычки не интерпретируются как CSV-разметка. Проверены допустимые классы и непустые '
      'тексты. Для обнаружения дубликатов выполняются Unicode NFKC-нормализация, casefold '
      'и свёртка пробелов. Сохраняется первое сообщение каждой группы. Нормализация для '
      'поиска повторов не заменяет входной текст модели; классификатор получает исходную SMS.')
    table('Состав данных до и после очистки', ['Этап', 'Всего', 'ham', 'spam'], [
        ('Исходный файл', data['raw_rows'], data['raw_classes']['ham'], data['raw_classes']['spam']),
        ('После удаления повторов', data['clean_rows'], data['clean_classes']['ham'], data['clean_classes']['spam'])], [3, 1, 1, 1])
    p(f"Исключено {data['excluded_rows']} повторов. Групп с противоречащими метками: "
      f"{data['conflicting_hashes']}. Журнал исключений сохранён в data/excluded.csv. "
      'Доля spam после очистки составляет ' + number(100 * data['clean_classes']['spam'] / data['clean_rows'], 2) +
      ' %. Поэтому accuracy рассматривается как дополнительная метрика: она не отражает '
      'полноту обнаружения редкого класса сама по себе.')
    h('2.1 Разбиение и предотвращение утечки', 2)
    p('Выполнено стратифицированное разбиение приблизительно 60/20/20 с random_state = 42. '
      'Идентификаторы исходных строк сохранены в data/splits.json. Множества row_id '
      'и хешей сообщений попарно не пересекаются. Сначала удаляются повторы, '
      'затем выполняется разбиение: одинаковые сообщения не оказываются одновременно '
      'в обучающей и отложенной выборках.')
    table('Состав выборок', ['Выборка', 'Назначение', 'Всего', 'ham', 'spam'], [
        (name, purpose, data['splits'][key]['rows'], data['splits'][key]['ham'], data['splits'][key]['spam'])
        for name, key, purpose in [('Train', 'train', 'Обучение TF-IDF и классификатора'),
                                   ('Validation', 'validation', 'Выбор гиперпараметров и модели'),
                                   ('Test', 'test', 'Итоговая оценка выбранной модели')]], [1, 3, 1, 1, 1])
    p('TF-IDF обучается только на train внутри sklearn Pipeline. На validation и test '
      'применяются уже обученные преобразования. Test не используется для выбора '
      'параметров или порога. После выбора модель не переобучается на train + validation: '
      'сохраняется соответствие оценённого run, Registry-версии и экспортированного Pipeline.')
    h('2.2 Метрики и правило выбора', 2)
    p('Положительный класс – spam. Precision = TP / (TP + FP), recall = TP / (TP + FN), '
      'F1 = 2 × precision × recall / (precision + recall). Precision характеризует '
      'долю настоящего спама среди отфильтрованных сообщений, recall – долю найденного '
      'спама среди всех spam-SMS. FP означает ошибочное блокирование обычной SMS, '
      'FN – пропуск спама.')
    p('Правило выбора задано заранее: максимальный F1 для spam на validation; '
      'при равенстве – максимальная precision, затем порядок конфигураций в коде. '
      'Порог принятия решения фиксирован на 0,5. При точном равенстве вероятностей '
      'выбирается ham, как в sklearn predict.')

    h('3 Программное окружение и структура проекта', page=True)
    p('Работа выполнена в Ubuntu 24.04 под WSL 2. Docker Desktop использует Linux engine '
      'и интеграцию с дистрибутивом Ubuntu. Обучение и MLflow выполняются на хосте, '
      'API – в контейнере. Графический интерфейс MLflow доступен локально на порту 5000, '
      'API через Compose – на порту 8000.')
    packages = summary['environment']['packages']
    versions = [('Python, хост', '3.12.3'), ('Python, образ', '3.12.10'),
                ('MLflow', packages['mlflow']), ('scikit-learn', packages['scikit-learn']),
                ('NumPy', packages['numpy']), ('Pandas', packages['pandas']),
                ('SciPy', '1.15.3'), ('FastAPI', packages['fastapi']),
                ('Pydantic', packages['pydantic']), ('Uvicorn', '0.34.2'),
                ('SQLAlchemy', '2.0.44'), ('Docker Engine', '29.4.3'), ('Docker Compose', '5.1.4')]
    halves = (len(versions) + 1) // 2
    version_rows = []
    for index in range(halves):
        other = versions[index + halves] if index + halves < len(versions) else ('', '')
        version_rows.append((*versions[index], *other))
    table('Версии программного окружения', ['Компонент', 'Версия', 'Компонент', 'Версия'],
          version_rows, [1.8, 1, 1.8, 1])
    p('Прямые зависимости указаны в requirements.in, полный снимок – в requirements.txt, '
      'окружение сервиса – в requirements-runtime.txt. SQLAlchemy закреплена '
      'на 2.0.44: версия 2.1.8 несовместима с используемым MLflow.')
    table('Основные файлы и каталоги', ['Путь', 'Назначение'], [
        ('lab4_mlops.ipynb', 'Выполненный ноутбук с отчётом и результатами.'),
        ('sms_mlops/', 'Подготовка данных, обучение, экспорт, API и проверки.'),
        ('data/raw/', 'Оригинальный архив UCI, TSV и readme.'),
        ('data/splits.json', 'Точные индексы train/validation/test и контрольная сумма данных.'),
        ('mlflow-data/', 'SQLite Tracking/Registry и артефакты запусков.'),
        ('models/v1/, models/v2/', 'Pipeline и metadata.json конкретных релизов.'),
        ('results/', 'Таблица экспериментов, метрики, графики и протоколы.'),
        ('screenshots/, examples/', 'Скриншоты реальных страниц и JSON-запросы.'),
        ('Dockerfile, compose.yaml', 'Сборка образов и декларативный запуск сервиса.')], [2, 4])

    h('4 Последовательность выполнения работы', page=True)
    p('Все команды выполняются в bash из каталога lab4. Перед запуском Docker Desktop '
      'запущен, включена интеграция Settings → Resources → WSL Integration → Ubuntu. '
      'Повторная установка Docker не требуется. Для Python используется отдельное '
      'виртуальное окружение.')
    code('''cd /home/tair/NETI/AI/lab4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
docker version
docker compose version
python -m sms_mlops.train
python -m sms_mlops.verify
python -m pytest -q
python -m sms_mlops.docker_demo --require''')
    p('Обучение записывает результаты непосредственно в SQLite. Для этого не нужен '
      'предварительно запущенный HTTP-сервер MLflow. Для просмотра тех же запусков '
      'в другом терминале запущен интерфейс:')
    code('''mlflow server --backend-store-uri sqlite:///mlflow-data/mlflow.db \\
  --default-artifact-root ./mlflow-data/artifacts --no-serve-artifacts \\
  --host 127.0.0.1 --port 5000 --workers 1''')
    p('Открыт адрес http://127.0.0.1:5000. Эксперимент называется lab4-sms-spam. '
      'В интерфейсе показаны шесть завершённых запусков с гиперпараметрами '
      'и validation-метриками (рис. 1).')
    figure('screenshots/01_mlflow_experiments.png', 'Шесть успешных экспериментов в MLflow', crop=(0.215, 0.06, 0.995, 0.51))
    p('При наличии результатов train повторно использует сохранённые запуски. '
      'Ключ --force предназначен для нового цикла: он добавляет шесть run и две новые '
      'Registry-версии, обновляет экспорты и требует повторной сборки образов. '
      'Для чтения готового отчёта повторное обучение не требуется.')

    h('5 Сравнение экспериментов', page=True)
    p('Во всех экспериментах используется word TF-IDF: униграммы и биграммы, '
      'max_features = 20 000, min_df = 1, sublinear_tf = True, lowercase = True '
      'и L2-нормировка. Для MultinomialNB исследованы alpha = 0,1; 0,5; 1. '
      'Для LogisticRegression исследованы C = 0,5; 2; 10, solver = liblinear, '
      'max_iter = 2000, random_state = 42. Разбиение одинаково для всех моделей.')
    rows = []
    for i, e in enumerate(experiments, 1):
        run_id = e['run_id']
        rows.append((i, 'NB' if e['algorithm'] == 'MultinomialNB' else 'LR',
                     e['parameter'] + ' = ' + number(e['value'], 1), number(e['val_f1_spam']),
                     number(e['val_precision_spam']), number(e['val_recall_spam']),
                     number(e['val_accuracy']), run_id[:16] + '\n' + run_id[16:]))
    table('Гиперпараметры, validation-метрики и идентификаторы запусков',
          ['№', 'Тип', 'Параметр', 'F1\nspam', 'P\nspam', 'R\nspam', 'Acc.', 'run_id'],
          rows, [.5, .9, 1.3, .9, 1, .9, 1, 3.4], size=10)
    p('NB – MultinomialNB, LR – LogisticRegression; P – precision, R – recall, '
      'Acc. – accuracy. Идентификаторы run_id перенесены '
      'на две строки без изменения символов. Полная таблица, включая время обучения '
      'и размер словаря, сохранена в results/experiments.csv.')
    figure('results/figures/experiments.png', 'Сравнение F1, precision и recall для spam на validation')
    p('На рис. 3 приведён скриншот MLflow: '
      'видны шесть запусков и validation F1, precision, recall для spam. '
      'Настройки alpha и C отражены в названиях запусков и сравнительной таблице. '
      'Размер словаря во всех опытах равен 20 000. Числа на графиках MLflow '
      'округлены; точные значения приведены в таблице 6.')
    figure('screenshots/06_mlflow_validation_metrics.png',
           'Шесть запусков MLflow: validation-метрики и размер словаря')
    p('В этой сетке увеличение alpha усиливает сглаживание NB и снижает recall. '
      'При alpha = 0,5 и 1 precision равна 1, но часть спама пропускается, '
      'поэтому F1 снижается. У LogisticRegression рост C ослабляет регуляризацию '
      'и повышает recall в исследованном диапазоне. Эти зависимости наблюдаются '
      'на данном разбиении и не гарантируются для любых данных.')
    p('В каждом run сохранены параметры модели и TF-IDF, seed, размеры выборок, '
      'validation F1/precision/recall/accuracy, время обучения и размер словаря. '
      'Теги включают вариант, положительный класс, правило выбора и SHA-256 данных '
      'и разбиения. Артефакты содержат Pipeline с signature и input_example, матрицу '
      'ошибок, код, зависимости и описание разбиения. Dataset inputs фиксируют роли '
      'train и validation. Итоговые test-метрики записаны только выбранному run.')

    h('6 Обоснование выбора и итоговая оценка', page=True)
    winner, alternative = summary['champion'], summary['alternative']
    p('Выбрана модель MultinomialNB с alpha = 0,1: validation F1 = '
      + number(winner['val_f1_spam'], 6) + ', precision = ' + number(winner['val_precision_spam'], 6) +
      ', recall = ' + number(winner['val_recall_spam'], 6) + '. Лучший представитель другого '
      'семейства – LogisticRegression с C = 10, validation F1 = '
      + number(alternative['val_f1_spam'], 6) + '. Разница между лидерами невелика; '
      'по одному разбиению нельзя утверждать статистически доказанное превосходство.')
    table('Итоговые метрики выбранного Pipeline на test', ['Метрика', 'Значение'], [
        ('F1 для spam', number(final['test_f1_spam'], 6)),
        ('Precision для spam', number(final['test_precision_spam'], 6)),
        ('Recall для spam', number(final['test_recall_spam'], 6)),
        ('Accuracy', number(final['test_accuracy'], 6)), ('Число test-SMS', final['rows'])], [2, 1])
    p('Скриншоты на рис. 4 и 5 показывают время обучения всех '
      'шести моделей, validation accuracy и итоговые test-метрики. На графиках '
      'test присутствует только выбранный run nb_alpha_0.1: остальные модели '
      'сравнивались на validation, а итоговая оценка test выполнена после выбора.')
    figure('screenshots/07_mlflow_metrics_overview.png',
           'Время обучения шести моделей и test accuracy/F1 чемпиона')
    figure('screenshots/08_mlflow_test_metrics.png',
           'Test precision/recall чемпиона и validation accuracy шести моделей')
    figure('results/figures/test_confusion_matrix.png', 'Матрица ошибок выбранной модели на test', width=10, max_height=8.5)
    tn, fp = final['confusion_matrix'][0]
    fn, tp = final['confusion_matrix'][1]
    p(f'Правильно обнаружены {tp} spam-SMS, пропущены {fn}; {fp} обычных сообщения '
      f'ошибочно отнесены к spam, {tn} сообщения ham распознаны правильно. '
      'Строки матрицы соответствуют истинным классам, столбцы – предсказанным, '
      'порядок классов: ham, spam. Высокая precision не означает обнаружение всего '
      'спама: recall составляет 87,5 %. После получения test-результатов параметры '
      'и порог не изменялись.')

    h('7 Регистрация версий и alias champion', page=True)
    p('Под именем sms-spam-filter зарегистрированы два уже обученных Pipeline. '
      'Регистрация не создаёт нового обучения. Релиз v1 содержит лучшую '
      'LogisticRegression, релиз v2 – выбранный MultinomialNB. Номера Registry '
      'получены из ответа MLflow, а не выведены из тегов образов.')
    table('Версии модели в Model Registry', ['Релиз', 'Registry', 'Алгоритм', 'run_id'], [
        (release, meta['model_version'], meta['algorithm'], meta['run_id'][:16] + '\n' + meta['run_id'][16:])
        for release, meta in summary['releases'].items()], [1, 1, 2.4, 2.8], size=11)
    p('Alias champion назначен Registry-версии ' + summary['releases']['v2']['model_version'] +
      '. При экспорте alias разрешается один раз в конкретный номер, после чего '
      'загружается models:/sms-spam-filter/<номер>. Смена alias не изменяет содержимое '
      'существующего Docker-образа и не переключает работающий контейнер автоматически.')
    figure('screenshots/02_registry.png',
           'Две Registry-версии и alias champion')
    p('Экспорт хранится в models/v1 и models/v2. pipeline.joblib содержит и TF-IDF, '
      'и классификатор, metadata.json – номер версии, run_id, происхождение, '
      'порядок классов, версии окружения и контрольные суммы. Во время предсказания '
      'контейнер не обращается к MLflow.')
    for release, meta in summary['releases'].items():
        p(f'Контрольная сумма Pipeline {release} (SHA-256):')
        code(meta['pipeline_sha256'])

    h('8 API и результаты проверки запросов', page=True)
    table('Методы API', ['Метод и путь', 'Назначение'], [
        ('GET /health', 'Готовность загруженной модели, версия и релиз.'),
        ('GET /metadata', 'Происхождение экспорта, run_id и SHA-256.'),
        ('POST /predict', 'Предсказания для пакета текстов.'),
        ('GET /docs', 'Интерактивная документация Swagger UI.')], [1.2, 3])
    p('Формат входа: объект с единственным полем texts. Допускается 1–100 строгих '
      'строк, каждая длиной 1–5000 символов и не только из пробелов. '
      'Числа, логические значения, null, пустые пакеты, лишние поля и повреждённый '
      'JSON возвращают 422. Unicode и неизвестные токены допустимы, но качество '
      'на русскоязычных SMS отдельно не оценивалось.')
    code('{"texts": ["See you soon", "Claim your free prize now!"]}')
    p('Ответ содержит predictions (ham/spam), probabilities_spam, model_version, '
      'release, run_id и latency_ms. Вероятности не проходили дополнительную '
      'калибровку. Модель загружается один раз при старте. До загрузки проверяется '
      'SHA-256 экспорта и порядок классов [0, 1]. Повреждённый файл предотвращает '
      'запуск сервиса.')
    check = docker['compose_checks']
    p('Фактический ответ на сохранённый пакет examples/valid.json:')
    code(json.dumps(check['example_response'], ensure_ascii=False, indent=2))
    p('Корректность запроса означает соответствие контракту, а не обязательное '
      'совпадение каждой метки с истиной. В smoke-примере возможен пропуск спама; '
      'качество оценивается отдельно по истинным меткам test. Третий текст примера '
      'выбран из validation, где версии дают разные ответы, чтобы изменение '
      'поведения при обновлении было наблюдаемым.')
    figure('screenshots/03_swagger_valid.png', 'Корректный POST /predict через Swagger: запрос и ответ 200',
           crop=(.055, .14, .95, .61), max_height=20)
    table('Некорректные запросы', ['Случай', 'Ожидаемый HTTP-код', 'Полученный код'], [
        (label, 422, 422) for label in [
            'Нет texts; неверный тип верхнего уровня', 'Пустой пакет; null вместо пакета',
            'Пустая строка; только пробелы', 'Число, boolean или null вместо SMS',
            'Смешанные типы в пакете', 'Строка вместо массива texts',
            'Лишнее поле в объекте', '101 SMS; сообщение длиннее 5000 символов', 'Повреждённый JSON']], [3, 1, 1], keep_rows=True)
    p(f"Всего проверено {len(check['invalid_requests'])} разновидностей невалидного ввода. "
      'Дополнительно проверены один объект, пакет ровно из 100 строк, Unicode '
      'и строка предельной длины. Все проверки выполняются как локально через '
      'FastAPI TestClient, так и по HTTP в контейнере.')
    figure('screenshots/04_swagger_invalid.png', 'Пустой пакет: контролируемый ответ HTTP 422',
           crop=(.055, .34, .95, .62), max_height=13)
    code('''curl -sS http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/predict \\
  -H 'Content-Type: application/json' --data-binary @examples/valid.json
curl -i -X POST http://127.0.0.1:8000/predict \\
  -H 'Content-Type: application/json' --data-binary @examples/invalid_empty.json''')

    h('9 Сборка Docker и проверка совпадения предсказаний', page=True)
    p('Dockerfile использует python:3.12.10-slim-bookworm, дополнительно '
      'закреплённый по digest. Устанавливаются только runtime-зависимости, '
      'копируются API и выбранный экспорт. Контейнер работает от appuser '
      'с UID 10001. Uvicorn слушает 0.0.0.0:8000 внутри контейнера; '
      'публикация на хосте ограничена 127.0.0.1. .dockerignore исключает '
      'SQLite, датасет, ноутбук и виртуальное окружение.')
    code('''docker build --build-arg MODEL_RELEASE=v1 -t sms-spam:v1 .
docker build --build-arg MODEL_RELEASE=v2 -t sms-spam:v2 .
docker run --rm -d --name lab4-sms-demo \\
  -p 127.0.0.1:18000:8000 sms-spam:v1
python -m sms_mlops.verify --url http://127.0.0.1:18000 --release v1
docker stop lab4-sms-demo''')
    table('Собранные Docker-образы', ['Тег', 'Размер, MiB', 'Image ID (SHA-256)'], [
        (image['tag'], number(image['size_bytes'] / 1024**2, 2),
         image['id'].replace('sha256:', '')[:32] + '\n' + image['id'].replace('sha256:', '')[32:])
        for image in docker['images'].values()], [1.2, 1, 3.7], size=10)
    p('Размер взят из docker image inspect и не равен объёму сетевой передачи. '
      'Image ID/digest идентифицируют содержимое, тогда как тег можно '
      'перезаписать. Полный stdout/stderr сборок сохранён в '
      'results/docker_build_v1.txt, results/docker_build_v2.txt и Docker-протоколе.')
    table('Совпадение API с экспортированным Pipeline',
          ['Проверка', 'Версия', 'Строк', 'Макс. разность вероятностей', 'Результат'], [
        (f'Контейнер, шаг {i}', item['model_version'], item['parity_rows'],
         number(item['max_probability_difference'], 12), item['status'])
        for i, item in enumerate(docker['sequence'], 1)], [2, .8, .8, 2, 1])
    p('Для каждой версии проверены все 1032 test-SMS и пять дополнительных строк: '
      'всего 1037. Метки API точно совпадают с Pipeline.predict, вероятности '
      'проверяются np.testing.assert_allclose с rtol = 1e-10 и atol = 1e-12. '
      'Эта проверка подтверждает корректность доставки, а не заменяет '
      'сравнение с истинными метками.')

    h('10 Обновление и откат без повторного обучения', page=True)
    p('Выполнена последовательность sms-spam:v1 → sms-spam:v2 → sms-spam:v1. '
      'Перед каждым запуском alias champion переключается на соответствующую '
      'Registry-версию. После готовности API проверяются metadata, версия, '
      'предсказания и ошибочные запросы. При откате используется сохранённый '
      'образ v1: переобучение, повторный экспорт и пересборка не выполняются.')
    table('Фактические шаги обновления и отката',
          ['Шаг', 'Тег образа', 'Registry / champion', 'Метки трёх SMS', 'Результат'], [
        (i, item['image_tag'], item['model_version'],
         ', '.join(item['example_response']['predictions']), item['status'])
        for i, item in enumerate(docker['sequence'], 1)], [.5, 1.3, 1.2, 2, .8])
    proof = docker['no_retraining']
    p(f"Число запусков обучения до демонстрации – {proof['runs_before']}, после – "
      f"{proof['runs_after']}. SHA-256 обоих pipeline.joblib до и после совпадают. "
      'Повторное появление исходных ответов подтверждает возврат предыдущей версии. '
      'После демонстрации champion возвращён выбранной по validation версии v2; '
      'эта же версия запущена через Compose.')
    code('''docker run --rm -d --name lab4-sms-demo \\
  -p 127.0.0.1:18000:8000 sms-spam:v2
python -m sms_mlops.verify --url http://127.0.0.1:18000 --release v2
docker stop lab4-sms-demo
docker run --rm -d --name lab4-sms-demo \\
  -p 127.0.0.1:18000:8000 sms-spam:v1
python -m sms_mlops.verify --url http://127.0.0.1:18000 --release v1
docker stop lab4-sms-demo''')
    table('Наблюдаемая длительность ответа, пакет из трёх SMS',
          ['Шаг / релиз', 'Медиана HTTP, мс', 'p95 HTTP, мс', 'Медиана обработчика, мс'], [
        (f"{i} / {item['release']}", number(item['latency']['roundtrip_median_ms'], 3),
         number(item['latency']['roundtrip_p95_ms'], 3), number(item['latency']['handler_median_ms'], 3))
        for i, item in enumerate(docker['sequence'], 1)], [1, 1.5, 1.5, 2])
    p('На каждом шаге выполнено десять последовательных запросов. '
      'Время обработчика измеряет вычисление, HTTP-время включает транспорт '
      'и сериализацию. Это smoke-измерение, а не нагрузочный тест. '
      'Обновление с остановкой контейнера допускает краткую недоступность.')

    h('11 Docker Compose и healthcheck', page=True)
    p('Compose использует уже собранный sms-spam:v2. Healthcheck каждые 5 секунд '
      'запрашивает /health внутри контейнера, timeout – 3 секунды, start_period – '
      '10 секунд, retries – 5. Команда --wait ожидает готовности сервиса. '
      'restart: unless-stopped перезапускает завершившийся контейнер; само '
      'состояние unhealthy не вызывает автоматического перезапуска.')
    code('''MODEL_RELEASE=v2 docker compose up -d --no-build --wait --wait-timeout 90
docker compose ps
docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q api)"
curl -f http://127.0.0.1:8000/health''')
    p('Фактический результат проверки состояния:')
    code(docker['compose_health'])
    p('Сводка docker compose ps:')
    ps = json.loads(docker['compose_ps'])
    table('Состояние Compose-сервиса', ['Параметр', 'Значение'], [
        ('Service', ps.get('Service', 'api')), ('Image', ps.get('Image', 'sms-spam:v2')),
        ('State', ps.get('State', 'running')), ('Health', docker['compose_health']),
        ('Проверка API', docker['compose_checks']['status']),
        ('Registry version', docker['compose_checks']['model_version'])], [1, 2])
    figure('screenshots/05_health.png', 'Ответ /health контейнерного сервиса', crop=(0, 0, .75, .13))
    captured = datetime.fromisoformat(docker['created_at']).astimezone(ZoneInfo('Asia/Novosibirsk'))
    p('Начало зафиксированного контейнерного прогона: ' + captured.strftime('%d.%m.%Y %H:%M:%S') +
      ' (Новосибирск, UTC+7). По HTTP дополнительно повторены все проверки '
      'контракта и совпадения с экспортом. Остановка сервиса после работы: '
      'docker compose down.')

    h('12 Выводы по работе', page=True)
    p('Создан воспроизводимый сервис фильтрации SMS-спама: сохранены исходные '
      'данные UCI, решения об исключении повторов, индексы разбиения, seed, '
      'зависимости, MLflow-запуски, Registry и экспортированные Pipeline. '
      'Полный Pipeline сохраняет обученный TF-IDF вместе с классификатором, '
      'что предотвращает расхождение преобразований при доставке.')
    p('Сравнены шесть конфигураций двух алгоритмов. По validation F1 выбрана '
      'MultinomialNB с alpha = 0,1. На отложенном test получены F1 = '
      + number(final['test_f1_spam'], 4) + ', precision = ' + number(final['test_precision_spam'], 4) +
      ', recall = ' + number(final['test_recall_spam'], 4) + '. Анализ precision/recall '
      'показал, что при высокой точности положительных ответов сохраняются '
      'пропуски спама. Увеличение alpha снижало recall NB, увеличение C '
      'повышало recall LogisticRegression в исследованной сетке.')
    p('Зарегистрированы две версии одной модели и назначен alias champion. '
      'Реальные контейнерные проверки подтвердили совпадение API с экспортом '
      'на 1037 текстах, контролируемые ответы 422, обновление и откат без '
      'переобучения. Compose завершил запуск со статусом healthy. Размер '
      'образов и наблюдаемая длительность ответа сохранены в протоколах.')
    p('Ограничения результата: исторический преимущественно англоязычный '
      'корпус, одно случайное разбиение, возможные близкие шаблоны после '
      'удаления повторов и отсутствие отдельной калибровки вероятностей. '
      'Полученная оценка не доказывает такое же качество на современных '
      'русскоязычных SMS. Дальнейшее развитие может включать кросс-валидацию '
      'на train, отдельный современный корпус и мониторинг изменения данных.')

    h('13 Список использованных источников', page=True)
    sources = [
        'Almeida T., Hidalgo J. SMS Spam Collection. UCI Machine Learning Repository, 2011. DOI: 10.24432/C5CC84. URL: https://archive.ics.uci.edu/dataset/228/sms+spam+collection (дата обращения: 10.11.2026).',
        'Лабораторная работа 4. MLOps: методические материалы, PDF, предоставленный к заданию. Разделы 7–17 и индивидуальный вариант 2.',
        'MLflow. Model Registry; sklearn API. URL: https://mlflow.org/docs/latest/model-registry/ ; https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html (дата обращения: 10.11.2026).',
        'scikit-learn 1.6. Pipelines and composite estimators; Text feature extraction; Model evaluation. URL: https://scikit-learn.org/1.6/modules/compose.html ; https://scikit-learn.org/1.6/modules/feature_extraction.html ; https://scikit-learn.org/1.6/modules/model_evaluation.html (дата обращения: 10.11.2026).',
        'Docker. Dockerfile reference; Compose services; WSL integration. URL: https://docs.docker.com/reference/dockerfile/ ; https://docs.docker.com/reference/compose-file/services/ ; https://docs.docker.com/desktop/features/wsl/ (дата обращения: 10.11.2026).',
        'FastAPI. Request Body. URL: https://fastapi.tiangolo.com/tutorial/body/ (дата обращения: 10.11.2026).',
        'Pydantic. Validators. URL: https://docs.pydantic.dev/latest/concepts/validators/ (дата обращения: 10.11.2026).']
    for i, text in enumerate(sources, 1):
        p(f'{i}. {text}')

    h('14 Приложение. Исходный код и конфигурация', page=True)
    p('Приведены исходные файлы вычислительной части, API, контейнерного '
      'сценария и проверок, зависимости, Dockerfile, Compose и примеры запросов. '
      'Вместе с отчётом сохраняется вся папка lab4. SQLite, бинарные Pipeline, '
      'датасет, изображения и полные протоколы представлены отдельными файлами. '
      'Генераторы ноутбука и Word-отчёта находятся в scripts/.')
    files = list(sorted((ROOT / 'sms_mlops').glob('*.py')))
    files += [ROOT / 'tests/test_service.py', ROOT / 'requirements.in', ROOT / 'requirements.txt',
              ROOT / 'requirements-runtime.txt', ROOT / 'Dockerfile', ROOT / 'compose.yaml', ROOT / '.dockerignore']
    files += sorted((ROOT / 'examples').glob('*.json'))
    for path in files:
        if not path.read_text().strip():
            continue
        h(path.relative_to(ROOT).as_posix(), 2)
        code(path.read_text())
    h('Контрольные суммы для воспроизведения', 2)
    p('SHA-256 неизменённого SMSSpamCollection:')
    code(data['raw_sha256'])
    p('SHA-256 data/splits.json:')
    code(data['split_sha256'])
    p('Проверенные результаты хранятся в results/summary.json, '
      'results/experiments.csv, results/test_metrics.json, '
      'results/local_checks.json и results/docker_checks.json. '
      'При переносе проекта необходимо сохранить mlflow-data целиком; '
      'python -m sms_mlops.relocate обновляет пути к артефактам в копии SQLite '
      'с предварительным резервным копированием.')
    report.doc.save(output)
    print(output)
    print(f'Tables: {report.table_counter}; figures: {report.figure_counter}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument('--output', type=Path, default=ROOT / 'Отчёт_ЛР4_MLOps.docx')
    args = parser.parse_args()
    build(args.template, args.output)
