# yangzhen
# 2021.12.04


from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg
from PyQt5 import QtWidgets as qw
import numpy as np
import copy


class MyQGraphicsView(qw.QGraphicsView):
    def __init__(self, parent):
        super(MyQGraphicsView, self).__init__(parent)
        self.setMouseTracking(True)

        self.points = []  # 选择的点
        self.allpoints = []
        self.startpos = qc.QPoint(0, 0)  # 鼠标中键按下时起始点
        self.endpos = qc.QPoint(0, 0)  # 鼠标中键弹起时终点
        self.scalenum = 1  # 缩放系数
        # self.scaleflag = 0 # 放大还是缩小(0:不动，1:放大，-1:缩小)
        self.nposx = 0  # 视图移动参数x
        self.nposy = 0  # 视图移动参数y
        # self.mindex = 1 # 当前区域编号
        self.flag = 0  # 是否进行选点的flag
        self.linkflag = 0  # 是否进行联动

    def SetLinkFlag(self, flag):
        """设置是否联动"""
        self.linkflag = flag

    def SetLinkWidget(self, mwidget):
        self.linkwidget = mwidget

    def SetLinkPara(self, para):
        """设置联动参数并作出联动操作"""
        scalenum = para[0]
        # print(scaleflag)
        nposx = para[1]
        nposy = para[2]
        # 设置缩放
        if scalenum > self.scalenum:
            self.scale(1.3, 1.3)
            self.scalenum = self.scalenum * 1.3
            # self.scalflag = 1
        elif scalenum < self.scalenum:
            self.scale(1 / 1.3, 1 / 1.3)
            self.scalenum = self.scalenum / 1.3
            # self.scalflag = -1
        # 设置移动
        if nposx != self.nposx and nposy != self.nposy:
            self.horizontalScrollBar().setValue(nposx)
            self.verticalScrollBar().setValue(nposy)
            self.nposx = nposx
            self.nposy = nposy

    def GetLinkPara(self):
        """获取联动参数"""
        para = []
        para.append(self.scalenum)
        para.append(self.nposx)
        para.append(self.nposy)
        return para

    def InitializeView(self):
        """View初始化"""
        self.points.clear()
        self.allpoints.clear()
        self.flag = 0
        self.scalenum = 1
        self.setCursor(qc.Qt.ArrowCursor)

    def SetChoosePoint(self, flag):
        """设置是否选点"""
        self.flag = flag
        if flag == 0:
            self.setCursor(qc.Qt.ArrowCursor)
        if flag == 1 or flag == 2:
            self.setCursor(qc.Qt.CrossCursor)

    def GetFlag(self):
        """获取当前是在做什么"""
        return self.flag

    def GetPoints(self):
        """获取点集"""
        return self.points

    def ClearPoints(self):
        """删除点集"""
        self.points.clear()

    def GetAllPoints(self):
        """获取所有区域点集"""
        return self.allpoints

    def ClearPoints(self):
        """删除所有区域点集合"""
        self.allpoints.clear()

    def PersentLiner(self, data, ratio):
        mdata = data.copy()
        rows, cols = data.shape[0:2]
        counts = rows * cols
        mdata = mdata.reshape(counts, 1)
        tdata = np.sort(mdata, axis=0)
        cutmin = tdata[int(counts * ratio), 0]
        cutmax = tdata[int(counts * (1 - ratio)), 0]
        ndata = 255.0 * (data.astype(np.float32) - cutmin) / float(cutmax - cutmin)
        ndata[data < cutmin] = 0
        ndata[data > cutmax] = 255
        return ndata

    def SetImage(self, data):
        """设置影像"""
        # if (data.type() == )
        dtp = data.dtype
        rows, cols, depth = data.shape
        # 如果通道数大于3个,只保留前边三个通道
        if depth > 3:
            img = data[:, :, 0:3].copy()
            img = img[:, :, ::-1]
        else:
            img = data.copy()
        rows, cols, depth = img.shape
        # 如果不是uint8的数据，需要先拉伸
        if dtp != np.uint8:
            for i in range(depth):
                img[:, :, i] = self.PersentLiner(img[:, :, i].copy(), 0.02)
            img = img.astype(np.uint8)
        # numpy矩阵转换成QImage
        if depth == 3:
            nimg = qg.QImage(img.data, cols, rows, cols * depth, \
                             qg.QImage.Format_RGB888)
        elif depth == 2:
            b1 = np.float32(img[:, :, 0])
            b2 = np.float32(img[:, :, 1])
            img = np.uint8((b1 + b2) / 2.)
            nimg = qg.QImage(img.data, cols, rows, cols * depth, \
                             qg.QImage.Format_Grayscale8)
        elif depth == 1:
            nimg = qg.QImage(img.data, cols, rows, cols * depth, \
                             qg.QImage.Format_Grayscale8)
        pix = qg.QPixmap.fromImage(nimg)
        item = qw.QGraphicsPixmapItem(pix)
        showscene = qw.QGraphicsScene()
        showscene.addItem(item)
        self.setScene(showscene)
        self.setTransform(qg.QTransform())

    def wheelEvent(self, event):
        if (event.angleDelta().y() > 0.5):
            self.scale(1.3, 1.3)
            self.scalenum = self.scalenum * 1.3
            # self.scaleflag = 1
        elif (event.angleDelta().y() < 0.5):
            self.scale(1 / 1.3, 1 / 1.3)
            # self.scaleflag = -1
            self.scalenum = self.scalenum / 1.3
        if self.linkflag == 1:
            self.linkwidget.SetLinkPara(self.GetLinkPara())

    def mousePressEvent(self, event):
        button = event.button()
        modifier = event.modifiers()
        # 按住ctrl时变更鼠标样式
        if button == qc.Qt.MiddleButton:
            self.setCursor(qc.Qt.PointingHandCursor)
            self.startpos = self.mapToScene(event.pos())
            # print(self.startpos)
        # 鼠标左键进行选点
        elif button == qc.Qt.LeftButton and self.flag == 1:
            p = self.mapToScene(event.pos())
            self.points.append([p.x(), p.y()])
            # 画点
            pen = qg.QPen()
            pen.setColor(qg.QColor(255, 0, 0))
            pen.setWidth(2)
            self.scene().addEllipse(p.x() - 1, p.y() - 1, 2, 2, pen)
            num = len(self.points)
            # 画线
            if num >= 2:
                pen.setColor(qg.QColor(241, 89, 42))
                pen.setWidth(1)
                self.scene().addLine(self.points[num - 2][0], \
                                     self.points[num - 2][1], \
                                     self.points[num - 1][0], \
                                     self.points[num - 1][1], pen)
        # 鼠标左键只选点
        elif button == qc.Qt.LeftButton and self.flag == 2:
            p = self.mapToScene(event.pos())
            self.points.append([p.x(), p.y()])
            # 画点
            pen = qg.QPen()
            pen.setColor(qg.QColor(255, 0, 0))
            pen.setWidth(2)
            self.scene().addEllipse(p.x() - 1, p.y() - 1, 4, 4, pen)
            # 添加文本
            font = qg.QFont("Roman times", 20, qg.QFont.Bold)
            text = qw.QGraphicsTextItem(str(len(self.points)))
            text.setPos(p.x() - 10, p.y() - 10)
            text.setDefaultTextColor(qg.QColor(248, 201, 0))
            text.setFont(font)
            self.scene().addItem(text)
        # 鼠标右键完成当前区域选点
        elif button == qc.Qt.RightButton and self.flag == 1:
            self.allpoints.append(copy.deepcopy(self.points))
            # 画线
            pen = qg.QPen()
            pen.setColor(qg.QColor(241, 89, 42))
            pen.setWidth(1)
            self.scene().addLine(self.points[-1][0], \
                                 self.points[-1][1], \
                                 self.points[0][0], \
                                 self.points[0][1], pen)
            # 添加文本
            tps = np.array(self.points)
            mps = np.mean(tps, axis=0)
            font = qg.QFont("Roman times", 20, qg.QFont.Bold)
            text = qw.QGraphicsTextItem(str(len(self.allpoints)))
            text.setPos(mps[0] - 20, mps[1] - 20)
            text.setDefaultTextColor(qg.QColor(248, 201, 0))
            text.setFont(font)
            self.scene().addItem(text)
            # 这个区域点压入集合
            self.points.clear()

    def mouseReleaseEvent(self, event):
        button = event.button()
        modifier = event.modifiers()
        # 鼠标中键弹起时进行视图移动
        if button == qc.Qt.MiddleButton:
            # 变更鼠标样式
            if self.flag == 0:
                self.setCursor(qc.Qt.ArrowCursor)
            elif self.flag == 1 or self.flag == 2:
                self.setCursor(qc.Qt.CrossCursor)
            # 记录当前点进行视图移动
            self.endpos = self.mapToScene(event.pos())
            # 获取滚动条当前位置
            oposx = self.horizontalScrollBar().value()
            oposy = self.verticalScrollBar().value()
            # 计算鼠标移动的距离
            offset = self.endpos - self.startpos
            # 根据移动的距离计算新的滚轮位置
            nposx = oposx - offset.x() * self.scalenum
            nposy = oposy - offset.y() * self.scalenum
            # 设置新的滚轮位置
            self.horizontalScrollBar().setValue(nposx)
            self.verticalScrollBar().setValue(nposy)
            # 记录一下备用
            self.nposx = nposx
            self.nposy = nposy
            # 进行联动
            if self.linkflag == 1:
                self.linkwidget.SetLinkPara(self.GetLinkPara())
