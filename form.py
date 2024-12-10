import os
import sys
import cv2
import qtmodern.styles
import qtmodern.windows
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QMessageBox, QVBoxLayout, QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QHBoxLayout
from PyQt5.QtGui import QPixmap, QIcon, QPen
from PyQt5.QtCore import QRectF, Qt
from PIL import Image
from ultralytics import YOLO

LOGO_PATH = os.path.join(os.curdir, "logo.jpg")

# Главная форма
class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система обнаружения повреждений автомобилей")
        self.setWindowIcon(QIcon(LOGO_PATH))
        self.setFixedSize(800, 600)

        layout = QVBoxLayout()

        imgPixmap = QPixmap(LOGO_PATH)
        imgPixmap = imgPixmap.scaled(500, 500, Qt.KeepAspectRatio)
        imgLabel = QLabel(self)
        imgLabel.setPixmap(imgPixmap)
        imgLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(imgLabel)

        label = QLabel("Система обнаружения повреждений автомобилей с помощью нейронных сетей", self)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(label)

        load_button = QPushButton("Загрузить изображение", self)
        load_button.clicked.connect(self.load_image)
        layout.addWidget(load_button)

        exit_button = QPushButton("Выход", self)
        exit_button.clicked.connect(self.close)
        layout.addWidget(exit_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            image_path = os.path.join("img", "image.png")
            os.makedirs("img", exist_ok=True)
            with open(file_path, 'rb') as src_file:
                with open(image_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())
            self.open_image_viewer(image_path)

    def open_image_viewer(self, image_path):
        self.viewer = ImageViewer(image_path, self)
        self.viewer.show()
        super().hide()


# Форма для редактирования и просмотра загруженного изображения
class ImageViewer(QMainWindow):
    def __init__(self, image_path, backForm):
        super().__init__()
        self.setWindowIcon(QIcon(LOGO_PATH))
        self.backForm = backForm
        self.image_path = image_path
        self.model = None
        self.detections = ""
        self.processed_image_path = None

        self.setWindowTitle("Просмотр изображения")
        self.setGeometry(100, 100, 800, 800)

        layout = QVBoxLayout()

        self.crop_button = QPushButton("Обрезать", self)
        self.crop_button.clicked.connect(self.crop_image)
        self.crop_button.setEnabled(False)  # Изначально кнопка неактивна
        layout.addWidget(self.crop_button)

        self.process_button = QPushButton("Обработать изображение", self)
        self.process_button.clicked.connect(self.process_image)
        layout.addWidget(self.process_button)

        self.graphics_view = GraphicsView(self.image_path, self.crop_button)  # Передаем кнопки сюда
        self.graphics_view.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.graphics_view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def crop_image(self):
        cropped_image = self.graphics_view.get_cropped_image()
        if cropped_image:
            cropped_image.save(os.path.join("img", "image.png"))  # Сохранение обрезанного изображения с тем же именем
            QMessageBox.information(self, "Сохранение", "Изображение успешно обрезано и сохранено.")
            self.graphics_view.update_view_image(os.path.join("img", "image.png"))

    def load_model(self):
        try:
            yolo_model_path = os.path.join(os.curdir, 'model', 'best.pt')
            self.model = YOLO(yolo_model_path)
        except Exception as e:
            QMessageBox.information(self, "Error", f"Error loading model: {str(e)}")

    # Метод для обработки изображения и отображения формы с результатом
    def process_image(self):
        self.load_model()
        try:
            if self.model is None:
                QMessageBox.warning(self, "Модель YOLO не загружена", "Модель YOLO не загружена. Пожалуйста проверьте путь.")
                raise ValueError("YOLO model not loaded. Please check the model path.")
            # Открытие изображения через cv2
            img = cv2.imread(self.image_path)
            if img is None:
                QMessageBox.warning(self, "Не удалось открыть изображение", f"Не удалось открыть изображение: {self.image_path}")
                raise ValueError(f"Failed to load img: {self.image_path}")
            # Нахождение объектов на изображении
            annotated_img = self.detect_objects(img)
            # Обратное преобразование изображения в PIL Image
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Сохранение обработанного изображения
            Image.fromarray(annotated_img).save(os.path.join("img", "processed_image.png"))
            self.processed_image_path = os.path.join("img", "processed_image.png")

            # Отображение формы с результатом
            QMessageBox.information(self, "Title", self.detections)
            self.results_viewer = ResultsViewer(self.processed_image_path, self.detections, self)
            self.results_viewer.show()
            self.hide()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error displaying img: {str(e)}\n")

    # Метод для определения текста обнаруженных объектов на изображении
    def update_log(self, detections, class_names):
        self.detections = ""

        label_counts = {}
        self.detections += "Detected:\n"

        if class_names is not None:
            labels = [class_names[int(cls)] for cls in detections[:, 5]]
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1

        for label, count in label_counts.items():
            if count > 1:
                self.detections += f" ●  {count}x - {label}\n"
            else:
                self.detections += f" ●  {label}\n"

    # Метод для нахождения объектов на изображении
    def detect_objects(self, img):
        results = self.model(img)
        self.update_log(results[0].boxes.data.cpu().numpy(), self.model.names)
        return results[0].plot()
    
    def closeEvent(self, a0):
        self.backForm.show()
        return super().closeEvent(a0)

# Блок для просмотра и обрезки изображения
class GraphicsView(QGraphicsView):
    def __init__(self, image_path, crop_button):
        super().__init__()
        self.image_path = image_path
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item = QGraphicsPixmapItem(QPixmap(self.image_path))
        self.scene.addItem(self.pixmap_item)

        # Масштабирование изображения под размер окна
        self.fitInView(self.pixmap_item, mode=1)

        # Переменные для выделения области
        self.start_point = None
        self.end_point = None
        self.selection_rect = None

        # Сохраняем ссылки на кнопки
        self.crop_button = crop_button

    def update_view_image(self, image_path):
        self.pixmap_item.setPixmap(QPixmap(image_path))
        if self.selection_rect:
            self.scene.removeItem(self.selection_rect)
            self.selection_rect = None  # Убираем ссылку на выделение
        self.start_point = None
        self.end_point = None
        self.crop_button.setEnabled(False)  # Деактивируем кнопку обрезки

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:  # Левый клик мыши
            self.start_point = self.mapToScene(event.pos())
            if self.selection_rect:
                self.scene.removeItem(self.selection_rect)
            self.selection_rect = None

    def mouseMoveEvent(self, event):
        if self.start_point is not None:
            if self.selection_rect:
                self.scene.removeItem(self.selection_rect)

            current_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, current_point).normalized()

            # Ограничение выделяемой области рамками изображения
            image_rect = self.pixmap_item.boundingRect()
            rect = rect.intersected(image_rect)
            pixmap_width = self.pixmap_item.pixmap().width()
            pixmap_height = self.pixmap_item.pixmap().height()
            pen = QPen(Qt.red, (pixmap_width+pixmap_height)/300)  # Красный цвет для выделенной области
            self.selection_rect = self.scene.addRect(rect, pen)

            # Проверка на активность кнопок
            if rect.width() > 0 and rect.height() > 0:
                self.crop_button.setEnabled(True)
            else:
                self.crop_button.setEnabled(False)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:  # Левый клик мыши
            self.end_point = self.mapToScene(event.pos())
            if self.selection_rect:
                rect = QRectF(self.start_point, self.end_point).normalized()

                # Ограничение выделяемой области рамками изображения
                image_rect = self.pixmap_item.boundingRect()
                rect = rect.intersected(image_rect)

                if rect.width() > 0 and rect.height() > 0:
                    self.crop_button.setEnabled(True)

    def get_cropped_image(self):
        if not self.start_point or not self.end_point:
            return None

        rect = QRectF(self.start_point, self.end_point).normalized()

        # Ограничиваем область обрезки рамками изображения
        image_rect = self.pixmap_item.boundingRect()
        rect = rect.intersected(image_rect)

        # Открываем изображение с помощью PIL для обрезки
        img = Image.open(self.image_path)

        # Преобразуем QRectF в координаты для PIL
        left = int(rect.left())
        top = int(rect.top())
        right = int(rect.right())
        bottom = int(rect.bottom())
        # Обрезаем изображение и возвращаем его
        return img.crop((left, top, right, bottom))

# Форма для отображения обработанных результатов и детектора YOLO
class ResultsViewer(QMainWindow):
    def __init__(self, image, detections, backForm=None):
        super().__init__()

        self.setWindowIcon(QIcon(LOGO_PATH))

        self.backForm = backForm

        self.setWindowTitle("Результаты обработки изображения")
        self.setGeometry(100, 100, 800, 600)  # Указываем размер и позицию окна

        layout = QVBoxLayout()
        hbox = QHBoxLayout()  # Горизонтальное размещение для кнопок

        self.back_button = QPushButton("Назад", self)
        self.back_button.clicked.connect(self.close)
        hbox.addWidget(self.back_button, alignment=Qt.AlignLeft)

        # Добавляем кнопку обратно в левую нижнюю часть
        self.processed_view = QLabel(self)
        self.processed_view.setAlignment(Qt.AlignCenter)
        self.processed_image = QPixmap(image)
        self.processed_view.setPixmap(self.processed_image)
        layout.addWidget(self.processed_view)

        self.detections_view = QLabel("Обнаруженные объекты:\n" + detections, self)
        hbox.addWidget(self.detections_view, alignment=Qt.AlignRight)
        
        layout.addLayout(hbox)  # Добавляем обе кнопки и лейбл с детекциями в вертикальный layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def close(self):
        self.backForm.show()  # Показать родительское окно при закрытии
        super().close()

    def closeEvent(self, a0):
        self.backForm.show()
        return super().closeEvent(a0)

if __name__ == "__main__":
    # Для установки логотипа
    try:
        # Включите в блок try/except, если вы также нацелены на Mac/Linux
        from PyQt5.QtWinExtras import QtWin                                         #  !!!
        myappid = 'mycompany.myproduct.subproduct.version'                          #  !!!
        QtWin.setCurrentProcessExplicitAppUserModelID(myappid)                      #  !!!    
    except ImportError:
        pass
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(LOGO_PATH))
    start_window = StartWindow()

    qtmodern.styles.dark(app)
    start_window.show()
    sys.exit(app.exec_())
