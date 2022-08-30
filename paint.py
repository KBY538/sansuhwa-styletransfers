from itertools import cycle
import sys
from PyQt5.QtCore import QSize, QBuffer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QDial, QDesktopWidget, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QIcon

from styletransfers.gaugan import spade_generate
from styletransfers.cyclegan.cyclegan_generate import CycleGANGenerator
from sansu_generator import figure_out

import io
import cv2
from PIL import Image, ImageQt

COLORS = {
          '하늘': "#87CEFA",  # 하늘
          '앞산': "#90EE90",  # 앞산
          '뒷산': "#20B2AA",  # 뒷산
          '땅': "#FFA500",  # 땅
          '바위': "#778899",  # 바위
          '풀': "#F08080",  # 풀
          '물': "#4169E1",  # 물
          '가까운 나무': "#8B4513",  # 가까운 나무
          '먼 나무': "#BC8F8F",  # 먼 나무
          }

SIZE = 500

def pil2qtgui(img):
    if img.mode == "RGB":
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
    elif  img.mode == "RGBA":
        r, g, b, a = img.split()
        img = Image.merge("RGBA", (b, g, r, a))
    elif img.mode == "L":
        img = img.convert("RGBA")

    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.size[0], img.size[1], QImage.Format_ARGB32)
    pix = QPixmap.fromImage(qimg)
    return pix

def qtgui2pil(label):
  buffer = QBuffer()
  buffer.open(QBuffer.ReadWrite)
  pix = ImageQt.fromqpixmap(label.pixmap())
  pix.save(buffer, "PNG")
  img = Image.open(io.BytesIO(buffer.data()))
  return img

class QPlatteButton(QPushButton):
  def __init__(self, color):
    super().__init__()
 
    self.setFixedSize(QSize(24, 24))
    self.color = color
    self.setStyleSheet("background-color: %s" % self.color)

class Canvas(QLabel):
  def __init__(self, size=SIZE):
    super().__init__()
    
    self.size = size
    canvas = QPixmap(size, size)
    canvas.fill(QColor(COLORS['하늘']))
    self.setPixmap(canvas)
 
    self.last_x, self.last_y = None, None
    self.pen_color = QColor(COLORS['앞산'])

    self.pen_size = 4

  def reset_canvas(self):
    canvas = QPixmap(self.size, self.size)
    canvas.fill(QColor(COLORS['하늘']))
    self.setPixmap(canvas)

  def set_pen_color(self, c):
    self.pen_color = QColor(c)
  
  def set_pen_size(self, s):
    self.pen_size = s
 
  def mouseReleaseEvent(self, *args, **kwargs):
    self.last_x, self.last_y = None, None
 
  def mouseMoveEvent(self, e):
    if self.last_x is None:
      self.last_x = e.x()
      self.last_y = e.y()
      return

    painter = QPainter(self.pixmap())
    pen = painter.pen()
    pen.setWidth(self.pen_size)
    pen.setColor(self.pen_color)
    painter.setPen(pen)
    painter.drawLine(self.last_x+3, self.last_y-30, e.x()+3, e.y()-30)
    painter.end()
    self.update()
 
    # update the origin for next time
    self.last_x = e.x()
    self.last_y = e.y()

class ImageCanvas(QLabel):
  def __init__(self, img_path, size=SIZE):
    super().__init__()

    self.size = size
    self.checkpoint_num = '460'

    if img_path == None:
      self.size = size
      canvas = QPixmap(size, size)
      canvas.fill(QColor('#D5D5D5'))
      self.setPixmap(canvas)
    else:
      img = Image.open(img_path)
      self.set_image(img)

  def convert2sansu(self, label):
    checkpoint_num = self.checkpoint_num
    label_img = qtgui2pil(label)
    output, output_path = spade_generate.generate_image(label_img, checkpoint_num)
    self.set_image(output)
    return output_path

  def convert2photo(self, input_img_path, save_path='result', save_filename='result.png'):
    if not hasattr(self, 'cyclegan_gen'):
      self.cyclegan_gen = CycleGANGenerator()
    output, _ = self.cyclegan_gen.cyclegan_generate(input_img_path, save_path, save_filename, convert_to=['A'])
    output = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    self.set_image(output)

  def set_image(self, img):
    img = img.resize((self.size, self.size))
    pixmap = pil2qtgui(img)
    self.setPixmap(pixmap)
  
  def set_checkpoint_num(self, num):
    self.checkpoint_num = str(num)

class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.initUI()

  def initUI(self):

    super().__init__()
 
    self.canvas = Canvas()
 
    widget = QWidget()
    hlayout = QHBoxLayout()
    hlayout.addStretch(1)
    
    widget.setLayout(hlayout)
    hlayout.addWidget(self.canvas)

    size_dial = QDial(self)
    size_dial.setRange(3, 50)
    size_dial.setNotchesVisible(True)
    combobox = QComboBox(self)

    for checkpoint in spade_generate.get_checkpoints():
      combobox.addItem(checkpoint[-3:])
    convert_button = QPushButton('변환')
    re_button = QPushButton('다시 그리기')
 
    palette = QVBoxLayout()
    hlayout.addLayout(palette)

    palette.addStretch()
    self.add_palette_buttons(palette)
    palette.addWidget(size_dial)
    palette.addWidget(combobox)
    palette.addWidget(convert_button)
    palette.addWidget(re_button)
    palette.addStretch()
    
    self.image_canvas = self.add_imgcanvas(hlayout, 'static/intro.png')
    self.image_canvas2 = self.add_imgcanvas(hlayout)

    save_button = QPushButton('저장')
    hlayout.addWidget(save_button)

    hlayout.addStretch(1)
    self.setCentralWidget(widget)
    combobox.activated[str].connect(self.image_canvas.set_checkpoint_num)
    size_dial.valueChanged.connect(lambda v=size_dial.value: self.canvas.set_pen_size(v))
    convert_button.pressed.connect(self.converting)
    re_button.pressed.connect(self.canvas.reset_canvas)
    save_button.pressed.connect(self.saving)

    self.setWindowTitle('산수화 만들기')
    self.setWindowIcon(QIcon('static/icon.png'))
    self.setFixedSize(1800, 580)
    self.center()
    self.show()

  def center(self):
    qr = self.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())

  def add_imgcanvas(self, layout, img_path=None):
    c = ImageCanvas(img_path)
    layout.addWidget(c)
    return c

  def add_palette_buttons(self, layout):
    for k, v in COLORS.items():
      temp_hbox = QHBoxLayout()
      l = QLabel(k)
      b = QPlatteButton(v)
      b.pressed.connect(lambda v=v: self.canvas.set_pen_color(v))
      temp_hbox.addWidget(b)
      temp_hbox.addWidget(l)
      layout.addLayout(temp_hbox)

  def converting(self):
    sansu_img = self.image_canvas.convert2sansu(self.canvas)
    self.image_canvas2.convert2photo(sansu_img)

  def saving(self):
    name = QFileDialog.getSaveFileName(self, 'Save file', '~/','JPEG (*.jpg *.jpeg);;PNG (*.png)')
    if name[0]:
      result = [qtgui2pil(self.canvas), qtgui2pil(self.image_canvas), qtgui2pil(self.image_canvas2)]
      figure_out(result).save(name[0])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())