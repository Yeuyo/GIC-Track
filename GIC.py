from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets, QtWebEngineCore, QtWebEngineWidgets
from PyQt5.QtCore import QDateTime, Qt, QTimer, QEvent
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QHBoxLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy, QTableWidgetItem,
        QSlider, QSpinBox, QDoubleSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFileDialog, QLineEdit, QStyledItemDelegate, qApp)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QStandardItem, QFontMetrics
from PyQt5 import QtWebEngineWidgets # allow for html pages (for plotly)
import sys
import statistics
from functools import partial # allow input of args into functions in connect
import numpy as np
import pandas as pd
import sqlite3
from dfply import X, group_by, summarize, summary_functions # R's dplyr equivalent
import plotly.express as px # plotting
import plotly.graph_objects as go
import scipy.io
# from plotnine import ggplot, aes, geom_line
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
from math import inf
import matplotlib.path as mpltPath # for points in polygon
from scipy.spatial import ConvexHull as chull # to identify border points
from scipy.spatial.distance import cdist # for distance between points
import multiprocessing as mp
from pointpats import ripley, PointPattern
import matplotlib.pyplot as plt
import plotly.offline as po
import fnmatch # unix filename filtering

class PlotlySchemeHandler(QtWebEngineCore.QWebEngineUrlSchemeHandler):
    def __init__(self, app):
        super().__init__(app)
        self.m_app = app

    def requestStarted(self, request):
        fig = self.m_app.fig_by_name()
        if isinstance(fig, go.Figure):
            raw_html = '<html><head><meta charset="utf-8" />'
            raw_html += '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>'
            raw_html += "<body>"
            raw_html += po.plot(fig, include_plotlyjs=False, output_type="div")
            raw_html += "</body></html>"
            buf = QtCore.QBuffer(parent=self)
            request.destroyed.connect(buf.deleteLater)
            buf.open(QtCore.QIODevice.WriteOnly)
            buf.write(raw_html.encode())
            buf.seek(0)
            buf.close()
            request.reply(b"text/html", buf)
            return
        request.fail(QtWebEngineCore.QWebEngineUrlRequestJob.UrlNotFound)

class PlotlyApplication(QtCore.QObject):
    scheme = b"plotly"

    def __init__(self, parent=None):
        super().__init__(parent)
        scheme = QtWebEngineCore.QWebEngineUrlScheme(PlotlyApplication.scheme)
        QtWebEngineCore.QWebEngineUrlScheme.registerScheme(scheme)

    def init_handler(self, view, profile=None):
        self.view = view
        if profile is None:
            profile = QtWebEngineWidgets.QWebEngineProfile.defaultProfile()
        handler = profile.urlSchemeHandler(PlotlyApplication.scheme)
        if handler is not None:
            profile.removeUrlSchemeHandler(handler)
        self.m_handler = PlotlySchemeHandler(self)
        profile.installUrlSchemeHandler(PlotlyApplication.scheme, self.m_handler)

    def fig_by_name(self):
        return self.view.fig

class CheckableComboBox(QComboBox):
    # Yoann Quenach de Quivillic on https://gis.stackexchange.com/questions/350148/qcombobox-multiple-selection-pyqt5
    # Subclass Delegate to increase item height
    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = qApp.palette()
        palette.setBrush(QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):

        if object == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())
        return res

class DStatistic(object):
    """
    Abstract Base Class for distance statistics.

    Parameters
    ----------
    name       : string
                 Name of the function. ("G", "F", "J", "K" or "L")

    Attributes
    ----------
    d          : array
                 The distance domain sequence.

    """
    def __init__(self, name):
        self.name = name

    def plot(self, qq=False):
        """
        Plot the distance function

        Parameters
        ----------
        qq: Boolean
            If False the statistic is plotted against distance. If Frue, the
            quantile-quantile plot is generated, observed vs. CSR.
        """

        # assuming mpl
        x = self.d
        if qq:
            plt.plot(self.ev, self._stat)
            plt.plot(self.ev, self.ev)
        else:
            plt.plot(x, self._stat, label='{}'.format(self.name))
            plt.ylabel("{}(d)".format(self.name))
            plt.xlabel('d')
            plt.plot(x, self.ev, label='CSR')
            plt.title("{} distance function".format(self.name))

class K(DStatistic):
    """
    Estimates the  K function for a point pattern.

    Parameters
    ----------
    pp         : :class:`.PointPattern`
                 Point Pattern instance.
    intervals  : int
                 The length of distance domain sequence.
    dmin       : float
                 The minimum of the distance domain.
    dmax       : float
                 The maximum of the distance domain.
    d          : sequence
                 The distance domain sequence.
                 If d is specified, intervals, dmin and dmax are ignored.

    Attributes
    ----------
    d          : array
                 The distance domain sequence.
    j          : array
                 K function over d.

    """
    def __init__(self, pp, intervals=10, dmin=0.0, dmax=None, d=None):
        res = _k(pp, intervals, dmin, dmax, d)
        self.d = res[:, 0]
        self.k = self._stat = res[:, 1]
        self.ev = np.pi * self.d * self.d
        super(K, self).__init__(name="K")

def _k(pp, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    Interevent K function.

    Parameters
    ----------
    pp       : :class:`.PointPattern`
               Point Pattern instance.
    n        : int
               Number of empty space points (random points).
    intevals : int
               Number of intervals to evaluate F over.
    dmin     : float
               Lower limit of distance range.
    dmax     : float
               Upper limit of distance range. If dmax is None, dmax will be set
               to length of bounding box diagonal.
    d        : sequence
               The distance domain sequence. If d is specified, intervals, dmin
               and dmax are ignored.

    Returns
    -------
    kcdf     : array
               A 2-dimensional numpy array of 2 columns. The first column is
               the distance domain sequence for the point pattern. The second
               column is corresponding K function.

    Notes
    -----
    See :class:`.K`

    """

    if d is None:
        # use length of bounding box diagonal as max distance
        bb = pp.mbb
        dbb = np.sqrt((bb[0]-bb[2])**2 + (bb[1]-bb[3])**2)
        w = dbb/intervals
        if dmax:
            w = dmax/intervals
        d = [w*i for i in range(intervals + 2)]
#    note: changed original code
#    https://github.com/pysal/pointpats/issues/67
    den = pp.lambda_window * pp.n
#    den = pp.lambda_window * pp.n *2
    kcdf = np.asarray([(di, len(pp.tree.query_pairs(di))*2/den) for di in d])
#    kcdf = np.asarray([(di, len(pp.tree.query_pairs(di))/den) for di in d])
    return kcdf

class LoadingBarWindow(QWidget):
    def __init__(self):
        super(LoadingBarWindow, self).__init__()
        self.setWindowTitle("Progress")
        self.setStyle(QStyleFactory.create('Fusion'))
        self.createProgressBar()
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.progressBar)
        self.setLayout(mainLayout)

    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(int(curVal + (maxVal - curVal) / 100))

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)

        timer = QTimer(self)
        timer.timeout.connect(self.advanceProgressBar)
        timer.start(10)

class ErrorMessageWindow(QWidget):
    def __init__(self, errorMessage):
        super().__init__()
        layout = QGridLayout()
        self.label = QLabel(errorMessage)
        layout.addWidget(self.label)
        self.setLayout(layout)

class Controller:
    def __init__(self, model, view):
        self._model = model
        self._view = view
        self._buttonResponse()
        self._loadExistingData()

    def uploadFileButton(self, fileType):
        files = QFileDialog.getOpenFileNames(self._view, "Choose File", os.getcwd())
        if files[1] != "":
            temp = files[0][0].split("/")
            input_path = temp[0] + "/"
            for n in range(len(temp) - 2):
                input_path += temp[n+1] + "/"
            file_names = np.zeros((len(files[0]),), dtype=object)
            for n in range(len(files[0])):
                file_names[n] = files[0][n].split("/")[-1][:-4]
            if self._view.acquisitionRateFast.isChecked():
                acquisition_rate = "fast"
            elif self._view.acquisitionRateSlow.isChecked():
                acquisition_rate = "slow"
            if self._view.analysisTypePercentage.isChecked():
                analysis_type = "percentage"
            elif self._view.analysisTypeNumber.isChecked():
                analysis_type = "number"
            exposureTime = self._view.exposureTimeBox.value() / 1000
            impars = {"PixelSize": self._view.pixelSize.value(),
                      "psf_scale": self._view.psfScaling.value(),
                      "wvlnth": float(self._view.emissionWavelengthBox.value()),
                      "iNA": float(self._view.detectionObjectiveNA.value()),
                      "psfStd": self._view.psfScaling.value() * 0.55 * (self._view.emissionWavelengthBox.value()) / self._view.detectionObjectiveNA.value() / 1.17 / self._view.pixelSize.value() / 2,
                      "FrameRate": float(self._view.exposureTimeBox.value() / 1000),
                      "FrameSize": float(self._view.exposureTimeBox.value() / 1000)
                     }
            locpars = {"wn": float(self._view.detectionBox.value()),
                       "errorRate": float(self._view.localizationErrorBox.value()),
                       "dfltnLoops": float(self._view.deflationLoopsNumberBox.value()),
                       "minInt": float(self._view.minIntensity.value()),
                       "maxOptimIter": float(self._view.maxIteration.value()),
                       "termTol": float(self._view.terminationTolerance.value()),
                       "isRadiusTol": self._view.radiusTolerance.isChecked(),
                       "radiusTol": float(self._view.radiusToleranceValue.value()),
                       "posTol": float(self._view.positionTolerance.value()),
                       "optim": [float(self._view.maxIteration.value()), float(self._view.terminationTolerance.value()), float(self._view.radiusTolerance.isChecked()), float(self._view.radiusToleranceValue.value()), float(self._view.positionTolerance.value())],
                       "isThreshLocPrec": self._view.threshLocPrec.isChecked(),
                       "minLoc": float(self._view.minLoc.value()),
                       "maxLoc": inf,
                       "isThreshSNR": self._view.threshSNR.isChecked(),
                       "minSNR": float(self._view.minSNR.value()),
                       "maxSNR": inf,
                       "isThreshDensity": self._view.threshDensity.isChecked()
                      }
            if self._view.maxLoc.value() > 0:
                locpars.update({"maxLoc": float(self._view.maxLoc.value())})
            if self._view.maxSNRIter.value() > 0:
                locpars.update({"maxSNR": float(self._view.maxSNRIter.value())})
            trackpars = {"trackStart": float(self._view.trackStart.value()),
                         "trackEnd": inf,
                         "Dmax": float(self._view.diffusionConstantMaxBox.value()),
                         "searchExpFac": float(self._view.exponentialFactorSearch.value()),
                         "statWin": float(self._view.statWin.value()),
                         "maxComp": float(self._view.compMax.value()),
                         "maxOffTime": float(self._view.gapsAllowedBox.value()),
                         "intLawWeight": float(self._view.intLawWeight.value()),
                         "diffLawWeight": float(self._view.difLawWeight.value())
                        }
            if self._view.trackEnd.value() > 0:
                locpars.update({"trackEnd": float(self._view.trackEnd.value())})
            scipy.io.savemat("tifupload.mat", {"input_path": input_path,
                                               "output_path": "../Data/fast-tif/",
                                               "output_path_further_processing": "../Data/fast-raw/",
                                               "file_names": file_names,
                                               "acquisition_rate": acquisition_rate,
                                               "analysis_type": analysis_type,
                                               "bleach_rate": self._view.bleachRate.value(),
                                               "impars": impars,
                                               "locpars": locpars,
                                               "trackpars": trackpars,
                                               "traj_length": self._view.trajectoryLengthBox.value(),
                                               "min_traj": self._view.minTrajectoryNumberBox.value(),
                                               "clip_factor": self._view.clipFactorBox.value(),
                                               "tol": self._view.toleranceBox.value(),
                                               "runParallel": self._view.parallelization.isChecked(),
                                               "ExposureTime": float(self._view.exposureTimeBox.value()),
                                               "numCores": self._view.parallelizationCores.value()
                                              }
                            )
            if sys.platform == "win32":
                # Windows
                if fileType == "raw_files":
                    os.system("matlab.exe -wait -nodesktop -nosplash -r \"run([pwd, '../SPT_LocAndTrack/fastSPT_JF549_TIFF_Diffusion.m']); exit(1)\"")
                elif fileType == "post_files":
                    os.system("matlab.exe -wait -nodesktop -nosplash -r \"run([pwd, '../SPT_LocAndTrack/UploadExistingData.m']); exit(1)\"")
            elif sys.platform == "darwin":
                # MacOS
                MATLAB_Versions = [matlab for matlab in os.listdir("/Applications/") if fnmatch.fnmatch(matlab, "MATLAB_*.app")]
                if len(MATLAB_Versions) == 0:
                    # MATLAB Not Installed
                    errorMessage = ErrorMessageWindow("No Installation of MATLAB Found.")
                    errorMessage.show()
                else:
                    MATLAB_To_Run = "/Applications/" + MATLAB_Versions[-1] + "/bin/matlab"
                    if fileType == "raw_files":
                        os.system(MATLAB_To_Run + " -wait -nodesktop -nosplash -r \"run([pwd, '/SPT_LocAndTrack/fastSPT_JF549_TIFF_Diffusion.m']); exit(1)\"")
                    elif fileType == "post_files":
                        os.system(MATLAB_To_Run + " -wait -nodesktop -nosplash -r \"run([pwd, '/SPT_LocAndTrack/UploadExistingData.m']); exit(1)\"")
            dataFile, dataTraj, dataTrack, dataJD, dataAngle, dataDwellTime = self._model.processUploadedFileToDatabase(file_names, acquisition_rate, exposureTime)
            with sqlite3.connect('database.db') as conn:
                dataFile.to_sql('FileList', conn, if_exists="append")
                dataTraj.to_sql('TrajectoryList', conn, if_exists="append")
                dataTrack.to_sql('TrackList', conn, if_exists="append")
                dataJD.to_sql('JDList', conn, if_exists="append")
                dataAngle.to_sql('AngleList', conn, if_exists="append")
                if len(dataDwellTime) > 0:
                    dataDwellTime.to_sql('DwellTimeData', conn, if_exists="append") # TODO: Make a new table for dwell time data
            self._loadExistingData()

    def comboMutationUpdate(self):
        data = self._model.updateMutationFilelist(self._view.comboAcquisitionRate.currentData())
        self._view.comboMutation.model().clear()
        self._view.comboMutation.addItems(set(data['mutation']))
        self._view.comboMutation.updateText()
        self._view.comboFileList.model().clear()
        self._view.comboFileList.addItems(set(data['filename']))
        self._view.comboFileList.updateText()

    def comboFileListUpdate(self):
        data = self._model.updateFilelist(self._view.comboAcquisitionRate.currentData(), self._view.comboMutation.currentData())
        self._view.comboFileList.model().clear()
        self._view.comboFileList.addItems(set(data['filename']))
        self._view.comboFileList.updateText()

    def sidebarFileList(self):
        data = self._model.getSelectedFiles(self._view.comboAcquisitionRate.currentData(), self._view.comboMutation.currentData(), self._view.comboFileList.currentData(), "FileList")
        return data
        # use group by filename to plot diffusion graph?
        # do what with the data? self.bigData = data?
        # don't need this? when in each tab, check for the selected condition?

    def plotTrajectory(self, *args):
        trajNumber = int(self._view.trajNumberBox.text())
        jumpNumber = int(self._view.jumpNumberDrawBox.text())
        trajLength = int(self._view.minTrajLength.text())
        selectionFile = self._view.comboFileList.currentData()
        if selectionFile == []:
            selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
        if len(selectionFile) > 1:
            self._view.trajectory_browser.setHtml("")
        else:
            data = self._model.getTrackFiles(selectionFile)
            df = self._model.plotTrajectory_data(trajLength, trajNumber, jumpNumber, data)

            # if self._view.trajTabTrajGroupButton.isChecked():
            #     figure = px.line(df, x = "x", y = "y", color = "trajID")
            # elif self._view.trajTabSpeedGroupButton.isChecked():
            #     1
            figure = px.line(df, x = "x", y = "y", color = "trajID", labels = {"x": "X (\u03BCm)", "y": "Y (\u03BCm)"})
            figure.layout.update(showlegend = False)
            self._view.trajectory_browser.setHtml(figure.to_html(include_plotlyjs='cdn'))

    def trajectoryData(self):
        selectionFile = self._view.comboFileList.currentData()
        if selectionFile == []:
            selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
        data = self._model.getTrajectoryDataFiles(selectionFile)
        boxData = data >> group_by(X.filename, X.mutation) >> summarize(TrajNumber = summary_functions.n_distinct(X.trajID))
        boxFigure = px.box(boxData, y = "TrajNumber", color = "mutation", points = "all")
        self._view.trajectoryNumberBox_browser.setHtml(boxFigure.to_html(include_plotlyjs='cdn'))
    def diffusionPlotUpdate(self):
        selectionFile = self._view.comboFileList.currentData()
        if selectionFile == []:
            selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
        if self._view.diffusionErrorVariation.isChecked() == True:
            errorView = 0
        elif self._view.diffusionErrorSTD.isChecked() == True:
            errorView = 1
        else:
            errorView = 2
        data = self._model.getTrajectoryFiles(selectionFile)
        if self._view.boundaryComputation.currentText() == "Formula":
            boundaryValue = -0.5
        else:
            boundaryValue = self._view.boundaryRawValue.value()
        plotData, pieData = self._model.produceDiffusionData(data, self._view.diffusionBinSize.value(), self._view.diffusionLowerLimit.value(), self._view.diffusionUpperLimit.value(), errorView, boundaryValue)       
        pieFigure = self._model.produceDiffusionPieFigures(pieData)
        if errorView == 0:
            figure = px.line(plotData, x = "x", y = "y", error_y = "error", error_y_minus = "error_minus", color = "mutation", labels = {"x": "Log10(D(um^2/s))", "y": "Normalized Frequency"})
        else:
            figure = px.line(plotData, x = "x", y = "y", error_y = "error", color = "mutation", labels = {"x": "Log10(D(um^2/s))", "y": "Normalized Frequency"})
        boundaryLine = pd.DataFrame({"x": [boundaryValue, boundaryValue], "y": [0, max(plotData["y"] + plotData["error"])]})
        figure.add_trace(px.line(boundaryLine, x = "x", y = "y", line_dash_sequence = ["longdashdot"], color_discrete_sequence = ["#7F7F7F"]).data[0])
        figure.update_layout(font = dict(size = 18))
        self._view.diffusion_browser.setHtml(figure.to_html(include_plotlyjs='cdn'))
        pieFigure[0].update_layout(font = dict(size = 16))
        self._view.diffusionFraction1_browser.setHtml(pieFigure[0].to_html(include_plotlyjs='cdn'))
        if pieFigure[1] != None:
            pieFigure[1].update_layout(font = dict(size = 16))
            self._view.diffusionFraction2_browser.setHtml(pieFigure[1].to_html(include_plotlyjs='cdn'))
        else:
            self._view.diffusionFraction2_browser.setHtml("")
        if pieFigure[2] != None:
            pieFigure[2].update_layout(font = dict(size = 16))
            self._view.diffusionFraction3_browser.setHtml(pieFigure[2].to_html(include_plotlyjs='cdn'))
        else:
            self._view.diffusionFraction3_browser.setHtml("")

    # def fractionExport(self):
    #     selectionFile = self._view.comboFileList.currentData()
    #     if selectionFile == []:
    #         selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
    #     data = self._model.getAngleTrackFiles(selectionFile)
    #     if self._view.boundaryComputation.currentText() == "Formula":
    #         boundaryValue = -0.5
    #     else:
    #         boundaryValue = self._view.boundaryRawValue.value()
    #     polarHist = self._model.produceAngleHist(data, self._view.diffusionLowerLimit.value(), self._view.diffusionUpperLimit.value(), boundaryValue) 
    #     testData = pd.DataFrame({"name": [0, 1, 2, 3, 4, 5], "number": [5, 6, 5, 4, 3, 4]})
    #     polarHist = px.bar_polar(testData, r = "number", theta = "name", color = "name")
    #     self._view.diffusionFraction3_browser.setHtml(polarHist.to_html(include_plotlyjs='cdn'))
    #     1

    def jumpDistancePlotUpdate(self):
        self._view.diffusionTrack2Par_browser.setHtml("")
        self._view.diffusionTrack3Par_browser.setHtml("")
        url = QtCore.QUrl()
        selectionFile = self._view.comboFileList.currentData()
        if selectionFile == []:
            selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
        data = self._model.getJumpDistanceData(selectionFile)
        twoParBoxData, threeParBoxData = self._model.getJumpDistanceBoxData(selectionFile)

        # Check if there's more than one condition being selected
        mutList = list(set(data['mutation']))
        if len(mutList) > 1:
            # Multi conditions comparison
            dataLineChart = data.loc[data["jump_distance"] <= self._view.jumpDistanceConsidered.value(),]
            dataMulti = dataLineChart >> group_by(X.mutation, X.jump_distance) >> summarize(sharedFrequency_mean = summary_functions.mean(X.sharedFrequency))
            twoParMultiFigure = px.line(dataMulti, x = "jump_distance", y = "sharedFrequency_mean", color = "mutation")
            self._view.diffusionTrack2Par_browser.setHtml(twoParMultiFigure.to_html(include_plotlyjs='cdn'))

            dataMultiTrajSum = dataMulti >> group_by(X.mutation) >> summarize(totalFrequency = (X.sharedFrequency_mean.sum()))
            dataMulti['frequencyRatio'] = 0
            for n in range(len(list(set(data["mutation"])))):
                dataMulti.loc[dataMulti['mutation'] == dataMultiTrajSum['mutation'][n], 'frequencyRatio'] = dataMulti.loc[dataMulti['mutation'] == dataMultiTrajSum['mutation'][n], 'sharedFrequency_mean'] / dataMultiTrajSum.iloc[n, 1]
            threeParMultiFigure = px.line(dataMulti, x = "jump_distance", y = "frequencyRatio", color = "mutation")
            self._view.diffusionTrack3Par_browser.setHtml(threeParMultiFigure.to_html(include_plotlyjs='cdn'))
        else:
            # Single condition
            data = data.loc[data["jump_distance"] <= self._view.jumpDistanceConsidered.value(),]
            twoParFigure = px.bar(data, x = "jump_distance", y = "sharedFrequency", color = "mutation", barmode = "group", labels = {"jump_distance": "Jump Distance (um)", "sharedFrequency": "Frequency"})
            for n in range(len(list(set(data["mutation"])))):
                # twoParFigure.add_trace(go.Scatter(data, x = "jump_distance", y = "twoParFrequency", color = "mutation", name = "test").data[n])
                twoParFigure.add_trace(px.line(data, x = "jump_distance", y = "twoParFrequency", color = "mutation", color_discrete_sequence = ["#EF553B"]).data[n])
                # twoParFigure.add_trace(px.line(data, x = "jump_distance", y = "twoParD1Values", color = "mutation", line_dash = "mutation", line_dash_sequence = ["dot"], color_discrete_sequence = ["#00CC96"]).data[n])
                # twoParFigure.add_trace(px.line(data, x = "jump_distance", y = "twoParD2Values", color = "mutation", line_dash = "mutation", line_dash_sequence = ["dash"], color_discrete_sequence = ["#AB63FA"]).data[n])
                twoParFigure.add_trace(px.line(data, x = "jump_distance", y = "twoParD1Values", color = "mutation", color_discrete_sequence = ["#00CC96"]).data[n])
                twoParFigure.add_trace(px.line(data, x = "jump_distance", y = "twoParD2Values", color = "mutation", color_discrete_sequence = ["#AB63FA"]).data[n])
            twoParFigure.update_xaxes(range = [0, self._view.jumpDistanceConsidered.value()])
            # twoParFigure.update_traces(name = "test", selector = dict(type="bar"))
            twoParFigure.update_traces(name = "Total", selector = dict(line_color="#EF553B"))
            twoParFigure.update_traces(name = "Diffusing", selector = dict(line_color="#00CC96"))
            twoParFigure.update_traces(name = "Bound", selector = dict(line_color="#AB63FA"))
            # self._view.diffusionTrack2Par_browser.setHtml(twoParFigure.to_html(include_plotlyjs='cdn'))
            self._view.fig = twoParFigure # TODO: Temporary fix by setting a figure to view scene so it can be captured by the url generator.
            url.setScheme(PlotlyApplication.scheme.decode())
            # self._view.diffusionTrack2Par_browser.load(url)
            self._view.diffusionTrack2Par_browser.setHtml(twoParFigure.to_html(include_plotlyjs='cdn'))
            
            threeParFigure = px.bar(data, x = "jump_distance", y = "sharedFrequency", color = "mutation", barmode = "group", labels = {"jump_distance": "Jump Distance (um)", "sharedFrequency": "Frequency"})
            for n in range(len(list(set(data["mutation"])))):
                threeParFigure.add_trace(px.line(data, x = "jump_distance", y = "threeParFrequency", color = "mutation", color_discrete_sequence = ["#EF553B"]).data[n])
                # threeParFigure.add_trace(px.line(data, x = "jump_distance", y = "threeParD1Values", color = "mutation", line_dash = "mutation", line_dash_sequence = ["dot"], color_discrete_sequence = ["#00CC96"]).data[n])
                # threeParFigure.add_trace(px.line(data, x = "jump_distance", y = "threeParD2Values", color = "mutation", line_dash = "mutation", line_dash_sequence = ["dash"], color_discrete_sequence = ["#AB63FA"]).data[n])
                # threeParFigure.add_trace(px.line(data, x = "jump_distance", y = "threeParD3Values", color = "mutation", line_dash = "mutation", line_dash_sequence = ["longdashdot"], color_discrete_sequence = ["#FFA15A"]).data[n])
                threeParFigure.add_trace(px.line(data, x = "jump_distance", y = "threeParD1Values", color = "mutation", color_discrete_sequence = ["#00CC96"]).data[n])
                threeParFigure.add_trace(px.line(data, x = "jump_distance", y = "threeParD2Values", color = "mutation", color_discrete_sequence = ["#AB63FA"]).data[n])
                threeParFigure.add_trace(px.line(data, x = "jump_distance", y = "threeParD3Values", color = "mutation", color_discrete_sequence = ["#FFA15A"]).data[n])
            threeParFigure.update_xaxes(range = [0, self._view.jumpDistanceConsidered.value()])
            threeParFigure.update_traces(name = "Total", selector = dict(line_color="#EF553B"))
            threeParFigure.update_traces(name = "Diffusing", selector = dict(line_color="#00CC96"))
            threeParFigure.update_traces(name = "Mixed", selector = dict(line_color="#AB63FA"))
            threeParFigure.update_traces(name = "Bound", selector = dict(line_color="#FFA15A"))
            # self._view.diffusionTrack3Par_browser.setHtml(threeParFigure.to_html(include_plotlyjs='cdn'))
            self._view.fig = threeParFigure # TODO: Temporary fix by setting a figure to view scene so it can be captured by the url generator.
            url.setScheme(PlotlyApplication.scheme.decode())
            self._view.diffusionTrack3Par_browser.load(url)
            
        # twoParFigure.update_layout(showlegend=False)
        twoParBox = px.box(twoParBoxData, x = "fraction", y = "values", color = "mutation", points = "all", labels = {"fraction": "States", "values": "Fraction"})
        self._view.diffusionTrack2ParBox_browser.setHtml(twoParBox.to_html(include_plotlyjs='cdn'))
        self._view.twoParNText.setText(str(round(sum(list(set(data["twoParN"]))), 2)) + " +/- " + str(sum(list(set(data["twoPardN"])))))
        self._view.twoParD1Text.setText(str(round(np.mean(list(set(data["twoParD1"]))), 2)) + " +/- " + str(round(np.mean(list(set(data["twoPardD1"]))), 2)) + " um/s")
        self._view.twoParD2Text.setText(str(round(np.mean(list(set(data["twoParD2"]))), 2)) + " +/- " + str(round(np.mean(list(set(data["twoPardD2"]))), 2)) + " um/s")
        self._view.twoParf1Text.setText(str(round(np.mean(list(set(data["twoParf1"]))), 2)) + " +/- " + str(round(np.mean(list(set(data["twoPardf1"]))), 2)))
        self._view.twoParSSRText.setText(str(round(np.mean(list(set(data["twoParSSR"]))), 2)))
     
        threeParBox = px.box(threeParBoxData, x = "fraction", y = "values", color = "mutation", points = "all", labels = {"fraction": "States", "values": "Fraction"})
        self._view.diffusionTrack3ParBox_browser.setHtml(threeParBox.to_html(include_plotlyjs='cdn'))
        self._view.threeParNText.setText(str(round(sum(list(set(data["threeParN"]))), 2)) + " +/- " + str(round(sum(list(set(data["threePardN"]))), 2)))
        self._view.threeParD1Text.setText(str(round(np.mean(list(set(data["threeParD1"]))), 2)) + " +/- " + str(round(np.mean(list(set(data["threePardD1"]))), 2)) + " um/s")
        self._view.threeParD2Text.setText(str(round(np.mean(list(set(data["threeParD2"]))), 2)) + " +/- " + str(round(np.mean(list(set(data["threePardD2"]))), 2)) + " um/s")
        self._view.threeParD3Text.setText(str(round(np.mean(list(set(data["threeParD3"]))), 2)) + " +/- " + str(round(np.mean(list(set(data["threePardD3"]))), 2)) + " um/s")
        self._view.threeParf1Text.setText(str(round(np.mean(list(set(data["threeParf1"]))), 2)) + " +/- " + str(round(np.mean(list(set(data["threePardf1"]))), 2)))
        self._view.threeParf2Text.setText(str(round(np.mean(list(set(data["threeParf2"]))), 2)) + " +/- " + str(round(np.mean(list(set(data["threePardf2"]))), 2)))
        self._view.threeParSSRText.setText(str(round(np.mean(list(set(data["threeParSSR"]))), 2)))

    def jumpDistanceDataSave(self):
        selectionFile = self._view.comboFileList.currentData()
        if selectionFile == []:
            selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
        data = self._model.getJumpDistanceData(selectionFile)
        twoParBoxData, threeParBoxData = self._model.getJumpDistanceBoxData(selectionFile)
        twoParBoxData.to_csv("Two_Parameter_Fit.csv")
        threeParBoxData.to_csv("Three_Parameter_Fit.csv")

    def anglePlot(self):
        selectionFile = self._view.comboFileList.currentData()
        if selectionFile == []:
            selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
        selectionAngle = self._view.angleSelection.currentData()
        if selectionAngle == []:
            selectionAngle = [self._view.angleSelection.itemText(i) for i in range(self._view.angleSelection.count())]
        angleRatio = self._view.angleRatio
        data = self._model.getAngleData(selectionFile)
        boundaryValue = self._view.boundaryValueAngle.value()
        mutHist, stateHist, boundHist, diffuHist, trendPlot, boxPlot = self._model.produceAnglePlots(data, boundaryValue, selectionAngle)
        self._view.trackAngleMut_browser.setHtml(mutHist.to_html(include_plotlyjs='cdn'))
        self._view.trackAngleState_browser.setHtml(stateHist.to_html(include_plotlyjs='cdn'))
        self._view.trackAngleBound_browser.setHtml(boundHist.to_html(include_plotlyjs='cdn'))
        self._view.trackAngleDiffu_browser.setHtml(diffuHist.to_html(include_plotlyjs='cdn'))
        self._view.trackAngleBox_browser.setHtml(boxPlot.to_html(include_plotlyjs='cdn'))

        # self._view.fig = trendPlot # TODO: Temporary fix by setting a figure to view scene so it can be captured by the url generator.
        # url = QtCore.QUrl()
        # url.setScheme(PlotlyApplication.scheme.decode())
        # self._view.trackAngleBox_browser.load(url)       

    def dOTMapUpdate(self):
        selectionFile = self._view.comboFileList.currentData()
        if selectionFile == []:
            selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
        dOTRegions = [float(i) for i in self._view.dOTRegionArea.text().split(",")]
        data = self._model.getDOTFiles(selectionFile)
        tableData, boxData, figure = self._model.getDOTData(data, dOTRegions, self._view.dOTMinTrajLength.value())

        # Set table
        self._view.dOTTable.setColumnCount(5)
        self._view.dOTTable.setHorizontalHeaderLabels(["Region", "Number of Data", "Max Length", "Mean Length", "Median Length"])
        self._view.dOTTable.setRowCount(len(dOTRegions) + 1)
        for n in range(len(tableData)):
            self._view.dOTTable.setItem(n, 0, QTableWidgetItem(str(tableData["Region"][n])))
            self._view.dOTTable.setItem(n, 1, QTableWidgetItem(str(tableData["number"][n])))
            self._view.dOTTable.setItem(n, 2, QTableWidgetItem(str(tableData["max"][n])))
            self._view.dOTTable.setItem(n, 3, QTableWidgetItem(str(tableData["mean"][n])))
            self._view.dOTTable.setItem(n, 4, QTableWidgetItem(str(tableData["median"][n])))

        # Plot the final figure
        if len(selectionFile) > 1:
            self._view.dOTMapBrowser.setHtml("")
        else:
            self._view.dOTMapBrowser.setHtml(figure.to_html(include_plotlyjs='cdn'))

        # Plot the box plot
        if self._view.dOTTrajChoiceMax.isChecked() == True:
            dOTTrajChoice = "max"
        elif self._view.dOTTrajChoiceMean.isChecked() == True:
            dOTTrajChoice = "mean"
        else:
            dOTTrajChoice = "median"
        if self._view.dOTDataPointChoice.isChecked() == True:
            dOTTrajPoints = "all"
        else:
            dOTTrajPoints = "outliers"
        boxFigure = px.box(boxData, x = "Region", y = dOTTrajChoice, color = "mutation", points = dOTTrajPoints, labels = {"mean": "Mean Trajectory LEngth (s)", "max": "Max Trajectory LEngth (s)", "median": "Median Trajectory LEngth (s)"})
        self._view.dOTBoxPlotBrowser.setHtml(boxFigure.to_html(include_plotlyjs='cdn'))

    def heatMapUpdate(self):
        selectionFile = self._view.comboFileList.currentData()
        if selectionFile == []:
            selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
        data = self._model.getTrackFiles(selectionFile)
        trajData = self._model.getHeatMapTraj(selectionFile, 0)
        trajData["duration"] = trajData["endTime"]-trajData["startTime"]

        trajCumData = trajData >> group_by(X.mutation, X.startTime) >> summarize(Frequency = summary_functions.n(X.startTime) / summary_functions.n_distinct(X.filename))
        trajCumData["Cummulative Frequency"] = trajCumData.groupby("mutation")["Frequency"].transform(pd.Series.cumsum)
        trajLiveData = data >> group_by(X.mutation, X.Frame) >> summarize(Frequency = summary_functions.n(X.Frame) / summary_functions.n_distinct(X.filename))

        figurePlot = px.density_heatmap(trajData, x = "meanX", y = "meanY", nbinsx = 16, nbinsy = 16)
        figureCumTrajs = px.line(trajCumData, x = "startTime", y = "Cummulative Frequency", color = "mutation", labels = {"startTime": "Time (s)"})
        figureLiveTrajs = px.line(trajLiveData, x = "Frame", y = "Frequency", color = "mutation", labels = {"Frame": "Time (s)"})
        figureLifetime = px.histogram(trajData, x = "duration", color = "mutation", labels = {"duration": "Burst lifetime (s)"}).update_layout(yaxis_title = "Frequency")
        
        mutations = list(set(trajCumData["mutation"]))
        ripleyData = pd.DataFrame()
        for n in range(len(mutations)):
            # points = trajCumData.loc[trajCumData["mutation"] == mutations[n], ["startTime", "Cummulative Frequency"]].to_numpy()
            points = trajData.loc[trajData["mutation"] == mutations[n], ["meanX", "meanY"]].to_numpy()
            pp = PointPattern(points)
            kp = K(pp)
            # ripleyData = pd.concat([ripleyData, pd.DataFrame({"mutation": mutations[n], "k": kp.k, "ev": kp.ev, "d": kp.d})], axis = 0)
            ripleyData = pd.concat([ripleyData, pd.DataFrame({"mutation": mutations[n] + " - k", "k": kp.k, "d": kp.d})], axis = 0)
            ripleyData = pd.concat([ripleyData, pd.DataFrame({"mutation": mutations[n] + " - ev", "k": kp.ev, "d": kp.d})], axis = 0)
        figureRipley = px.line(ripleyData, x = "d", y = "k", color = "mutation")


        self._view.heatMapPlot.setHtml(figurePlot.to_html(include_plotlyjs='cdn'))

        self._view.heatMapCummulativeTrajs.setHtml(figureCumTrajs.to_html(include_plotlyjs='cdn'))
        self._view.heatMapLiveTrajs.setHtml(figureLiveTrajs.to_html(include_plotlyjs='cdn'))
        self._view.heatMapBurstLifetime.setHtml(figureLifetime.to_html(include_plotlyjs='cdn'))
        self._view.heatMapRipley.setHtml(figureRipley.to_html(include_plotlyjs='cdn'))

    def dwellTimeUpdate(self):
        selectionFile = self._view.comboFileList.currentData()
        if selectionFile == []:
            selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
        boxFigure, pieFigure = self._model.produceDwellTimeFigures(selectionFile)
        if boxFigure != [None]:
            self._view.dwellBox_browser.setHtml(boxFigure.to_html(include_plotlyjs='cdn'))
            self._view.dwellPie1_browser.setHtml(pieFigure[0].to_html(include_plotlyjs='cdn'))
            if pieFigure[1] != None:
                self._view.dwellPie2_browser.setHtml(pieFigure[1].to_html(include_plotlyjs='cdn'))
            else:
                self._view.dwellPie2_browser.setHtml("")
            if pieFigure[2] != None:
                self._view.dwellPie3_browser.setHtml(pieFigure[2].to_html(include_plotlyjs='cdn'))
            else:
                self._view.dwellPie3_browser.setHtml("")
        else:
            self._view.dwellBox_browser.setHtml("")
            self._view.dwellPie1_browser.setHtml("")
            self._view.dwellPie2_browser.setHtml("")
            self._view.dwellPie3_browser.setHtml("")

    def tabsUpdate(self):
        tabIndex = self._view.tabs.currentIndex()
        # self._view.tabs.tabText(self._view.tabs.currentIndex()) <- use string instead of index?
        if tabIndex == 1:
            if self._view.trajectory_tab.currentIndex() == 0: # Trajectory Plot
                self.plotTrajectory()
            elif self._view.trajectory_tab.currentIndex() == 1: # Trajectory Data
                self.trajectoryData()
        elif tabIndex == 2:
            if self._view.diffusionTabs.currentIndex() == 0: # Diffusion Plot
                self.diffusionPlotUpdate()
            elif self._view.diffusionTabs.currentIndex() == 1:
                self.jumpDistancePlotUpdate()
        elif tabIndex == 3: # Angle
            self.anglePlot()
        elif tabIndex == 4: # Distribution of Tracks
            self.dOTMapUpdate()
        elif tabIndex == 5: # Heat map
            self.heatMapUpdate()
        elif tabIndex == 6: # Dwell time
            self.dwellTimeUpdate()

    def deleteFiles(self):
        selectionFile = self._view.comboFileList.currentData()
        if selectionFile == []:
            selectionFile = [self._view.comboFileList.itemText(i) for i in range(self._view.comboFileList.count())]
        with sqlite3.connect('database.db') as conn:
            if len(selectionFile) > 1:
                data = pd.read_sql_query(f"SELECT TrackList.trajID FROM TrackList INNER JOIN TrajectoryList ON TrackList.trajID = TrajectoryList.trajID AND TrajectoryList.filename IN {tuple(selectionFile)}", conn)
                trajIDs = tuple(pd.unique(data.trajID))
                conn.execute(f"DELETE FROM TrackList WHERE TrackList.trajID IN {trajIDs}")
                conn.execute(f"DELETE FROM TrajectoryList WHERE TrajectoryList.filename IN {tuple(selectionFile)}")
                conn.execute(f"DELETE FROM JDList WHERE JDList.filename IN {tuple(selectionFile)}")
                conn.execute(f"DELETE FROM DwellTimeData WHERE DwellTimeData.filename IN {tuple(selectionFile)}")
                conn.execute(f"DELETE FROM FileList WHERE FileList.filename IN {tuple(selectionFile)}")
            else:
                data = pd.read_sql_query(f"SELECT TrackList.trajID FROM TrackList INNER JOIN TrajectoryList ON TrackList.trajID = TrajectoryList.trajID AND TrajectoryList.filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
                trajIDs = tuple(pd.unique(data.trajID))
                conn.execute(f"DELETE FROM TrackList WHERE TrackList.trajID IN {trajIDs}")
                conn.execute("DELETE FROM TrajectoryList WHERE TrajectoryList.filename = ?", (selectionFile[0],))
                conn.execute("DELETE FROM JDList WHERE JDList.filename = ?", (selectionFile[0],))
                conn.execute("DELETE FROM DwellTimeData WHERE DwellTimeData.filename = ?", (selectionFile[0],))
                conn.execute("DELETE FROM FileList WHERE FileList.filename = ?", (selectionFile[0],))
        self._loadExistingData()
        return

    def _buttonResponse(self):
        self._view.comboAcquisitionRate.model().dataChanged.connect(self.comboMutationUpdate)
        self._view.comboMutation.model().dataChanged.connect(self.comboFileListUpdate)
        # self._view.comboFileList.model().dataChanged.connect(self.sidebarFileList)
        self._view.tabs.currentChanged.connect(self.tabsUpdate)
        self._view.trajectory_tab.currentChanged.connect(self.tabsUpdate)
        self._view.diffusionTabs.currentChanged.connect(self.tabsUpdate)
        
        self._view.comboAcquisitionRate.model().dataChanged.connect(self.tabsUpdate)
        self._view.comboMutation.model().dataChanged.connect(self.tabsUpdate)
        self._view.comboFileList.model().dataChanged.connect(self.tabsUpdate)
        self._view.deleteFile.pressed.connect(self.deleteFiles)

        # Trajectory Plot
        self._view.trajNumberBox.textChanged.connect(self.plotTrajectory) # or valueChanged
        self._view.jumpNumberDrawBox.textChanged.connect(self.plotTrajectory)
        self._view.minTrajLength.textChanged.connect(self.plotTrajectory)
        self._view.jumpDistanceToCSV.clicked.connect(self.jumpDistanceDataSave)
        # self._view.fractionExportButton.clicked.connect(self.fractionExport)
        # self._view.trajTabTrajGroupButton.clicked.connect(self.plotTrajectory)

        # Diffusion Plot Interactive
        self._view.diffusionBinSize.valueChanged.connect(self.diffusionPlotUpdate)
        self._view.diffusionLowerLimit.valueChanged.connect(self.diffusionPlotUpdate)
        self._view.diffusionUpperLimit.valueChanged.connect(self.diffusionPlotUpdate)
        self._view.diffusionErrorVariation.clicked.connect(self.diffusionPlotUpdate)
        self._view.diffusionErrorSTD.clicked.connect(self.diffusionPlotUpdate)
        self._view.diffusionErrorSEM.clicked.connect(self.diffusionPlotUpdate)
        self._view.boundaryComputation.currentTextChanged.connect(self.diffusionPlotUpdate)
        self._view.boundaryRawValue.valueChanged.connect(self.diffusionPlotUpdate)
        
        # Track Diffusion Plot Interactive
        self._view.jumpDistanceConsidered.valueChanged.connect(self.jumpDistancePlotUpdate)

        # Angle Plot Interactive
        self._view.boundaryValueAngle.valueChanged.connect(self.anglePlot)

        # Distribution of Tracks Interactive
        self._view.dOTTrajChoiceMax.clicked.connect(self.dOTMapUpdate)
        self._view.dOTTrajChoiceMean.clicked.connect(self.dOTMapUpdate)
        self._view.dOTTrajChoiceMedian.clicked.connect(self.dOTMapUpdate)
        self._view.dOTDataPointChoice.toggled.connect(self.dOTMapUpdate)
        self._view.dOTMinTrajLength.valueChanged.connect(self.dOTMapUpdate)
        self._view.dOTButton.clicked.connect(self.dOTMapUpdate)

        # Upload tab
        self._view.uploadFileButton.pressed.connect(partial(self.uploadFileButton, "raw_files"))
        self._view.uploadPostFileButton.pressed.connect(partial(self.uploadFileButton, "post_files"))
              
    def _loadExistingData(self):
        # Remove previous data to prevent repeats after data upload
        self._view.comboAcquisitionRate.model().clear()
        self._view.comboMutation.model().clear()
        self._view.comboFileList.model().clear()
        self._view.angleSelection.model().clear()

        # Get data from database
        with sqlite3.connect('database.db') as conn:
            try:
                df = pd.read_sql_query("select * from FileList", conn)
                dataExist = True
            except:
                dataExist = False
                
        if dataExist == True:
            # Adding data from database to selection
            acquisitionRate = set(df['acquisition_rate'])
            self._view.comboAcquisitionRate.addItems(acquisitionRate)
            mutation = set(df['mutation'])
            self._view.comboMutation.addItems(mutation)
            filename = df['filename']
            self._view.comboFileList.addItems(filename)
            angleList = pd.DataFrame({"0 - 10": "",
                                      "10 - 20": "",
                                      "20 - 30": "",
                                      "30 - 40": "",
                                      "40 - 50": "",
                                      "50 - 60": "",
                                      "60 - 70": "",
                                      "70 - 80": "",
                                      "80 - 90": "",
                                      "90 - 100": "",
                                      "100 - 110": "",
                                      "110 - 120": "",
                                      "120 - 130": "",
                                      "130 - 140": "",
                                      "140 - 150": "",
                                      "150 - 160": "",
                                      "160 - 170": "",
                                      "170 - 180": ""},
                                      index = [0])
            self._view.angleSelection.addItems(angleList)

            # Clearing the names in the box
            self._view.comboAcquisitionRate.updateText()
            self._view.comboMutation.updateText()
            self._view.comboFileList.updateText()
            self._view.angleSelection.updateText()

class Model:
    def __init__(self):
        1

    @staticmethod
    def processUploadedFileToDatabaseParallel(file_name, acquisition_rate, exposureTime):
        dataDir = os.path.realpath(os.getcwd()) + "/Data/fast-tif/"
        
        mutation = file_name.split("_")[0]
        data = scipy.io.loadmat(dataDir + file_name + "_dataTrack.mat")
        dataTrack = pd.DataFrame({"trajID": [file_name + f"_{m}" for m in data['dataTrack'][:, 0].astype(int)],
                                  "Frame": data['dataTrack'][:, 1].astype(float),
                                  "x": data['dataTrack'][:, 2].astype(float),
                                  "y": data['dataTrack'][:, 3].astype(float),
                                  "msd": data['dataTrack'][:, 4].astype(float)
                                 }
                                )
        trajIDs = list(set(dataTrack["trajID"].to_numpy().astype(str)))
        distances = pd.DataFrame(data = [])
        for m in range(len(trajIDs)):
            dataSubset = dataTrack.loc[dataTrack["trajID"] == trajIDs[m]]
            points = dataSubset.to_numpy()[:, (2,3)].astype(float)
            hull = chull(points)
            hullpoints = points[hull.vertices, :]
            hdist = cdist(hullpoints, hullpoints, metric='euclidean')
            distances = pd.concat([distances, pd.DataFrame({"meanX": dataSubset['x'].mean(), "meanY": dataSubset['y'].mean(), "maxDistance": hdist.max(), "meanDistance": hdist[0,:][1:].mean(), "medianDistance": np.median(hdist[0,:][1:])}, index = [m])], axis = 0)
        data = scipy.io.loadmat(dataDir + file_name + "_dataTraj.mat")
        dataTraj = pd.DataFrame({"filename": [file_name] * len(data['dataTraj'][:, 0]),
                                 "trajID": [file_name + f"_{m}" for m in data['dataTraj'][:, 1].astype(int)],
                                 "traj_length": data['dataTraj'][:, 2].astype(int),
                                 "msd": data['dataTraj'][:, 3].astype(float),
                                 "D": data['dataTraj'][:, 4].astype(float),
                                 "startTime": data['dataTraj'][:, 5].astype(float),
                                 "endTime": data['dataTraj'][:, 6].astype(float),
                                 "meanX": distances["meanX"],
                                 "meanY": distances["meanY"],
                                 "maxDistance": distances["maxDistance"],
                                 "meanDistance": distances["meanDistance"],
                                 "medianDistance": distances["medianDistance"]
                                }
                               )
        data = scipy.io.loadmat(dataDir + file_name + "_CMPFitPar.mat")
        twoCMPData = scipy.io.loadmat(dataDir + file_name + "_2CMPFit.mat")
        threeCMPData = scipy.io.loadmat(dataDir + file_name + "_3CMPFit.mat")
        dataJD = pd.DataFrame({"filename": [file_name] * len(data['FJH'][:, 0]),
                               "jump_distance": data["FJH"][:, 0],
                               "sharedFrequency": data["FJH"][:, 1],
                               "twoParFrequency": data["FJH"][:, 3],
                               "threeParFrequency": data["FJH"][:, 4],
                               "twoParD1Values": twoCMPData["twoCMPFit"][:, 0],
                               "twoParD2Values": twoCMPData["twoCMPFit"][:, 1],
                               "threeParD1Values": threeCMPData["threeCMPFit"][:, 0],
                               "threeParD2Values": threeCMPData["threeCMPFit"][:, 1],
                               "threeParD3Values": threeCMPData["threeCMPFit"][:, 2]
                              }
                             )
        data = scipy.io.loadmat(dataDir + file_name + "_FitPar.mat")
        dataFile = pd.DataFrame(data = {"filename": file_name,
                                        "mutation": mutation,
                                        "acquisition_rate": acquisition_rate,
                                        "exposure_time": exposureTime,
                                        "twoParN": data["FitPar"][0, 3],
                                        "twoPardN": data["FitPar"][1, 3],
                                        "twoParD1": data["FitPar"][0, 4],
                                        "twoPardD1": data["FitPar"][1, 4],
                                        "twoParD2": data["FitPar"][0, 5],
                                        "twoPardD2": data["FitPar"][1, 5],
                                        "twoParf1": data["FitPar"][0, 6],
                                        "twoPardf1": data["FitPar"][1, 6],
                                        "twoParSSR": data["FitPar"][0, 7],
                                        "threeParN": data["FitPar"][0, 8],
                                        "threePardN": data["FitPar"][1, 8],
                                        "threeParD1": data["FitPar"][0, 9],
                                        "threePardD1": data["FitPar"][1, 9],
                                        "threeParD2": data["FitPar"][0, 10],
                                        "threePardD2": data["FitPar"][1, 10],
                                        "threeParD3": data["FitPar"][0, 11],
                                        "threePardD3": data["FitPar"][1, 11],
                                        "threeParf1": data["FitPar"][0, 12],
                                        "threePardf1": data["FitPar"][1, 12],
                                        "threeParf2": data["FitPar"][0, 13],
                                        "threePardf2": data["FitPar"][1, 13],
                                        "threeParSSR": data["FitPar"][0, 14]
                                       }, index = [0]
                               )
        data = scipy.io.loadmat(dataDir + file_name + "_dataAngle.mat")
        dataAngle = pd.DataFrame(data = {"filename": file_name,
                                         "trajID": [file_name + f"_{m}" for m in data['dataAngle'][:, 0].astype(int)],
                                         "A1": [m for m in data['dataAngle'][:, 1].astype(int)],
                                         "A2": [m for m in data['dataAngle'][:, 2].astype(int)],
                                         "A3": [m for m in data['dataAngle'][:, 3].astype(int)],
                                         "A4": [m for m in data['dataAngle'][:, 4].astype(int)],
                                         "A5": [m for m in data['dataAngle'][:, 5].astype(int)],
                                         "A6": [m for m in data['dataAngle'][:, 6].astype(int)],
                                         "A7": [m for m in data['dataAngle'][:, 7].astype(int)],
                                         "A8": [m for m in data['dataAngle'][:, 8].astype(int)],
                                         "A9": [m for m in data['dataAngle'][:, 9].astype(int)],
                                         "A10": [m for m in data['dataAngle'][:, 10].astype(int)],
                                         "A11": [m for m in data['dataAngle'][:, 11].astype(int)],
                                         "A12": [m for m in data['dataAngle'][:, 12].astype(int)],
                                         "A13": [m for m in data['dataAngle'][:, 13].astype(int)],
                                         "A14": [m for m in data['dataAngle'][:, 14].astype(int)],
                                         "A15": [m for m in data['dataAngle'][:, 15].astype(int)],
                                         "A16": [m for m in data['dataAngle'][:, 16].astype(int)],
                                         "A17": [m for m in data['dataAngle'][:, 17].astype(int)],
                                         "A18": [m for m in data['dataAngle'][:, 18].astype(int)]
                                        })
        if acquisition_rate == "fast":
            return dataFile, dataTraj, dataTrack, dataJD, dataAngle
        else:
            if os.path.exists(dataDir + file_name + "_dwellTime.mat"):
                data = scipy.io.loadmat(dataDir + file_name + "_dwellTime.mat")
                dataDwellTime = pd.DataFrame(data = {"filename": file_name,
                                                    "R1": data["dwellTimeData"][0][0], # TODO: Make a new table for dwell time data
                                                    "R2": data["dwellTimeData"][0][1],
                                                    "F": data["dwellTimeData"][0][2]
                                                    }, index = [0]
                                            )
            else:
                dataDwellTime = pd.DataFrame()
            return dataFile, dataTraj, dataTrack, dataJD, dataAngle, dataDwellTime

    def processUploadedFileToDatabase(self, file_names, acquisition_rate, exposureTime):
        mpPool = mp.Pool(mp.cpu_count())
        data = [mpPool.apply(self.processUploadedFileToDatabaseParallel, args = (file_names[n], acquisition_rate, exposureTime)) for n in range(len(file_names))]
        mpPool.close()

        dataFile = pd.DataFrame(data = [])
        dataTraj = pd.DataFrame(data = [])
        dataTrack = pd.DataFrame(data = [])
        dataJD = pd.DataFrame(data = [])
        dataAngle = pd.DataFrame(data = [])
        dataDwellTime = pd.DataFrame(data = [])
        if acquisition_rate == "fast":
            for n in range(len(data)):
                dataFile = pd.concat([dataFile, data[n][0]], axis = 0)
                dataTraj = pd.concat([dataTraj, data[n][1]], axis = 0)
                dataTrack = pd.concat([dataTrack, data[n][2]], axis = 0)
                dataJD = pd.concat([dataJD, data[n][3]], axis = 0)
                dataAngle = pd.concat([dataAngle, data[n][4]], axis = 0)
        else:
            for n in range(len(data)):
                dataFile = pd.concat([dataFile, data[n][0]], axis = 0)
                dataTraj = pd.concat([dataTraj, data[n][1]], axis = 0)
                dataTrack = pd.concat([dataTrack, data[n][2]], axis = 0)
                dataJD = pd.concat([dataJD, data[n][3]], axis = 0)
                dataAngle = pd.concat([dataAngle, data[n][4]], axis = 0)
                dataDwellTime = pd.concat([dataDwellTime, data[n][5]], axis = 0)
        return dataFile, dataTraj, dataTrack, dataJD, dataAngle, dataDwellTime

    def updateMutationFilelist(self, selectionRate):
        if selectionRate == []:
            with sqlite3.connect('database.db') as conn:
                data = pd.read_sql_query("select * from FileList", conn)
        else:
            with sqlite3.connect('database.db') as conn:
                if len(selectionRate) > 1:
                    data = pd.read_sql_query(f"select * from FileList where acquisition_rate IN {tuple(selectionRate)}", conn)
                else:
                    data = pd.read_sql_query("select * from FileList where acquisition_rate = :selectionRate", conn, params = {"selectionRate": selectionRate[0]})
        return data

    def updateFilelist(self, selectionRate, selectionMutation):
        if selectionRate == []:
            if selectionMutation == []:
                with sqlite3.connect('database.db') as conn:
                    data = pd.read_sql_query("select * from FileList", conn)
            else:
                # data = pd.DataFrame(data = [])
                with sqlite3.connect('database.db') as conn:
                    if len(selectionMutation) > 1:
                        data = pd.read_sql_query(f"select * from FileList where mutation IN {tuple(selectionMutation)}", conn)
                    else:
                        data = pd.read_sql_query("select * from FileList where mutation = :selectionMutation", conn, params = {"selectionMutation": selectionMutation[0]})
                    # for n in range(len(selectionMutation)):
                    #     data = data.append(pd.read_sql_query("select * from FileList where mutation = :selectionMutation", conn, params = {"selectionMutation": selectionMutation[n]}))
        else:
            # data = pd.DataFrame(data = [])
            if selectionMutation == []:
                with sqlite3.connect('database.db') as conn:
                    if len(selectionRate) > 1:
                        data = pd.read_sql_query(f"select * from FileList where acquisition_rate IN {tuple(selectionRate)}", conn)
                    else:
                        data = pd.read_sql_query("select * from FileList where acquisition_rate = :selectionRate", conn, params = {"selectionRate": selectionRate[0]})
                    # for n in range(len(selectionRate)):
                    #     data = data.append(pd.read_sql_query("select * from FileList where acquisition_rate = :selectionRate", conn, params = {"selectionRate": selectionRate[n]}))
            else:
                # data = pd.DataFrame(data = [])
                with sqlite3.connect('database.db') as conn:
                    if len(selectionRate) > 1:
                        if len(selectionMutation) > 1:
                            data = pd.read_sql_query(f"select * from FileList where acquisition_rate IN {tuple(selectionRate)} AND mutation IN {tuple(selectionMutation)}", conn)
                        else:
                            data = pd.read_sql_query(f"select * from FileList where acquisition_rate IN {tuple(selectionRate)} AND mutation = :selectionMutation", conn, params = {"selectionMutation": selectionMutation[0]})
                    else:
                        if len(selectionMutation) > 1:
                            data = pd.read_sql_query(f"select * from FileList where acquisition_rate = :selectionRate AND mutation IN {tuple(selectionMutation)}", conn, params = {"selectionRate": selectionRate[0]})
                        else:
                            data = pd.read_sql_query("select * from FileList where acquisition_rate = :selectionRate AND mutation = :selectionMutation", conn, params = {"selectionRate": selectionRate[0], "selectionMutation": selectionMutation[0]})
                    # for n in range(len(selectionRate)):
                    #     for m in range(len(selectionMutation)):
                    #         data = data.append(pd.read_sql_query("select * from FileList where acquisition_rate = :selectionRate AND mutation = :selectionMutation", conn, params = {"selectionRate": selectionRate[n], "selectionMutation": selectionMutation[m]}))
        return data

    def getSelectedFiles(self, selectionRate, selectionMutation, selectionFile, table):
        if selectionRate == []:
            if selectionMutation == []:
                if selectionFile == []:
                    # All data
                    with sqlite3.connect('database.db') as conn:
                        data = pd.read_sql_query(f"select * from {table}", conn)
                else:
                    # All rate, all condition, some files
                    with sqlite3.connect('database.db') as conn:
                        if len(selectionFile) > 1:
                            data = pd.read_sql_query(f"select * from {table} where filename IN {tuple(selectionFile)}", conn)
                        else:
                            data = pd.read_sql_query(f"select * from {table} where filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
            else:
                if selectionFile == []:
                    # All rate, some condition, all files
                    with sqlite3.connect('database.db') as conn:
                        if len(selectionMutation) > 1:
                            data = pd.read_sql_query(f"select * from {table} where mutation IN {tuple(selectionMutation)}", conn)
                        else:
                            data = pd.read_sql_query(f"select * from {table} where mutation = :selectionMutation", conn, params = {"selectionMutation": selectionMutation[0]})
                else:
                    # All rate, some condition, some files
                    with sqlite3.connect('database.db') as conn:
                        if len(selectionMutation) > 1:
                            if len(selectionFile) > 1:
                                data = pd.read_sql_query(f"select * from {table} where mutation IN {tuple(selectionMutation)} AND filename IN {tuple(selectionFile)}", conn)
                            else:
                                data = pd.read_sql_query(f"select * from {table} where mutation IN {tuple(selectionMutation)} AND filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
                        else:
                            if len(selectionFile) > 1:
                                data = pd.read_sql_query(f"select * from {table} where mutation = :selectionMutation AND filename IN {tuple(selectionFile)}", conn, params = {"selectionMutation": selectionMutation[0]})
                            else:
                                data = pd.read_sql_query(f"select * from {table} where mutation = :selectionMutation AND filename = :selectionFile", conn, params = {"selectionMutation": selectionMutation[0], "selectionFile": selectionFile[0]})
        else:
            if selectionMutation == []:
                if selectionFile == []:
                    # Some rate, all condition, all files
                    with sqlite3.connect('database.db') as conn:
                        if len(selectionRate) > 1:
                            data = pd.read_sql_query(f"select * from {table} where acquisition_rate IN {tuple(selectionRate)}", conn)
                        else:
                            data = pd.read_sql_query(f"select * from {table} where acquisition_rate = :selectionRate", conn, params = {"selectionRate": selectionRate[0]})
                else:
                    # Some rate, all condition, some files
                    with sqlite3.connect('database.db') as conn:
                        if len(selectionRate) > 1:
                            if len(selectionFile) > 1:
                                data = pd.read_sql_query(f"select * from {table} where acquisition_rate IN {tuple(selectionRate)} AND filename IN {tuple(selectionFile)}", conn)
                            else:
                                data = pd.read_sql_query(f"select * from {table} where acquisition_rate IN {tuple(selectionRate)} AND filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
                        else:
                            if len(selectionFile) > 1:
                                data = pd.read_sql_query(f"select * from {table} where acquisition_rate = :selectionRate AND filename IN {tuple(selectionFile)}", conn, params = {"selectionRate": selectionRate[0]})
                            else:
                                data = pd.read_sql_query(f"select * from {table} where acquisition_rate = :selectionRate AND filename = :selectionFile", conn, params = {"selectionRate": selectionRate[0], "selectionFile": selectionFile[0]})
            else:
                if selectionFile == []:
                    # Some rate, some condition, all files
                    with sqlite3.connect('database.db') as conn:
                        if len(selectionRate) > 1:
                            if len(selectionMutation) > 1:
                                data = pd.read_sql_query(f"select * from {table} where acquisition_rate IN {tuple(selectionRate)} AND mutation = {tuple(selectionMutation)}", conn)
                            else:
                                data = pd.read_sql_query(f"select * from {table} where acquisition_rate IN {tuple(selectionRate)} AND mutation = :selectionMutation", conn, params = {"selectionMutation": selectionMutation[0]})
                        else:
                            if len(selectionMutation) > 1:
                                data = pd.read_sql_query(f"select * from {table} where acquisition_rate = :selectionRate AND mutation IN {tuple(selectionMutation)}", conn, params = {"selectionRate": selectionRate[0]})
                            else:
                                data = pd.read_sql_query(f"select * from {table} where acquisition_rate = :selectionRate AND mutation = :selectionMutation", conn, params = {"selectionRate": selectionRate[0], "selectionMutation": selectionMutation[0]})
                else:
                    # Some rate, some condition, some files
                    with sqlite3.connect('database.db') as conn:
                        if len(selectionRate) > 1:
                            if len(selectionMutation) > 1:
                                if len(selectionFile) > 1:
                                    data = pd.read_sql_query(f"select * from {table} where acquisition_rate IN {tuple(selectionRate)} AND mutation IN {tuple(selectionMutation)} AND filename IN {tuple(selectionFile)}", conn)
                                else:
                                    data = pd.read_sql_query(f"select * from {table} where acquisition_rate IN {tuple(selectionRate)} AND mutation IN {tuple(selectionMutation)} AND filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
                            else:
                                if len(selectionFile) > 1:
                                    data = pd.read_sql_query(f"select * from {table} where acquisition_rate IN {tuple(selectionRate)} AND mutation = :selectionMutation AND filename IN {tuple(selectionFile)}", conn, params = {"selectionMutation": selectionMutation[0]})
                                else:
                                    data = pd.read_sql_query(f"select * from {table} where acquisition_rate IN {tuple(selectionRate)} AND mutation = :selectionMutation AND filename = :selectionFile", conn, params = {"selectionMutation": selectionMutation[0], "selectionFile": selectionFile[0]})
                        else:
                            if len(selectionMutation) > 1:
                                if len(selectionFile) > 1:
                                    data = pd.read_sql_query(f"select * from {table} where acquisition_rate = :selectionRate AND mutation IN {tuple(selectionMutation)} AND filename IN {tuple(selectionFile)}", conn, params = {"selectionRate": selectionRate[0]})
                                else:
                                    data = pd.read_sql_query(f"select * from {table} where acquisition_rate = :selectionRate AND mutation IN {tuple(selectionMutation)} AND filename = :selectionFile", conn, params = {"selectionRate": selectionRate[0], "selectionFile": selectionFile[0]})
                            else:
                                if len(selectionFile) > 1:
                                    data = pd.read_sql_query(f"select * from {table} where acquisition_rate = :selectionRate AND mutation = :selectionMutation AND filename IN {tuple(selectionFile)}", conn, params = {"selectionRate": selectionRate[0], "selectionMutation": selectionMutation[0]})
                                else:
                                    data = pd.read_sql_query(f"select * from {table} where acquisition_rate = :selectionRate AND mutation = :selectionMutation AND filename = :selectionFile", conn, params = {"selectionRate": selectionRate[0], "selectionMutation": selectionMutation[0], "selectionFile": selectionFile[0]})
        return data

    def getTrajectoryFiles(self, selectionFile):
        with sqlite3.connect('database.db') as conn:
            if len(selectionFile) > 1:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.traj_length, TrajectoryList.msd, TrajectoryList.D from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename WHERE FileList.filename IN {tuple(selectionFile)}", conn)
            else:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.traj_length, TrajectoryList.msd, TrajectoryList.D from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename WHERE FileList.filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
        return data

    def getTrajectoryDataFiles(self, selectionFile):
        with sqlite3.connect('database.db') as conn:
            if len(selectionFile) > 1:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.trajID, TrajectoryList.traj_length from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename WHERE FileList.filename IN {tuple(selectionFile)}", conn)
            else:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.trajID, TrajectoryList.traj_length from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename WHERE FileList.filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
        return data

    def getAngleTrackFiles(self, selectionFile):
        with sqlite3.connect('database.db') as conn:
            if len(selectionFile) > 1:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.trajID, TrajectoryList.traj_length, TrajectoryList.D, TrackList.Frame, TrackList.x, TrackList.y from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename INNER JOIN TrackList ON TrajectoryList.trajID = TrackList.trajID WHERE FileList.filename IN {tuple(selectionFile)}", conn)
            else:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.trajID, TrajectoryList.traj_length, TrajectoryList.D, TrackList.Frame, TrackList.x, TrackList.y from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename INNER JOIN TrackList ON TrajectoryList.trajID = TrackList.trajID WHERE FileList.filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
        return data

    def produceAnglePlots(self, data, boundaryValue, selectionAngle):
        mutData = data >> group_by(X.mutation) >> summarize(A1 = X.A1.sum() / summary_functions.n_distinct(X.filename),
                                                            A2 = X.A2.sum() / summary_functions.n_distinct(X.filename),
                                                            A3 = X.A3.sum() / summary_functions.n_distinct(X.filename),
                                                            A4 = X.A4.sum() / summary_functions.n_distinct(X.filename),
                                                            A5 = X.A5.sum() / summary_functions.n_distinct(X.filename),
                                                            A6 = X.A6.sum() / summary_functions.n_distinct(X.filename),
                                                            A7 = X.A7.sum() / summary_functions.n_distinct(X.filename),
                                                            A8 = X.A8.sum() / summary_functions.n_distinct(X.filename),
                                                            A9 = X.A9.sum() / summary_functions.n_distinct(X.filename),
                                                            A10 = X.A10.sum() / summary_functions.n_distinct(X.filename),
                                                            A11 = X.A11.sum() / summary_functions.n_distinct(X.filename),
                                                            A12 = X.A12.sum() / summary_functions.n_distinct(X.filename),
                                                            A13 = X.A13.sum() / summary_functions.n_distinct(X.filename),
                                                            A14 = X.A14.sum() / summary_functions.n_distinct(X.filename),
                                                            A15 = X.A15.sum() / summary_functions.n_distinct(X.filename),
                                                            A16 = X.A16.sum() / summary_functions.n_distinct(X.filename),
                                                            A17 = X.A17.sum() / summary_functions.n_distinct(X.filename),
                                                            A18 = X.A18.sum() / summary_functions.n_distinct(X.filename)
                                                           )
        mutData = mutData.rename(columns = {"A1": "0 - 10",
                                            "A2": "10 - 20",
                                            "A3": "20 - 30",
                                            "A4": "30 - 40",
                                            "A5": "40 - 50",
                                            "A6": "50 - 60",
                                            "A7": "60 - 70",
                                            "A8": "70 - 80",
                                            "A9": "80 - 90",
                                            "A10": "90 - 100",
                                            "A11": "100 - 110",
                                            "A12": "110 - 120",
                                            "A13": "120 - 130",
                                            "A14": "130 - 140",
                                            "A15": "140 - 150",
                                            "A16": "150 - 160",
                                            "A17": "160 - 170",
                                            "A18": "170 - 180"})
        mutData["Base"] = 0
        mutData.loc[:, 1:19] = mutData.iloc[:, 1:19].divide(mutData.sum(1, numeric_only = True), axis = 0)
        mutData = mutData.melt(id_vars = ["mutation", "Base"], var_name = "Theta", value_name = "Counts")
        mutHist = px.bar_polar(mutData, r = "Counts", theta = "Theta", color = "mutation", base = "Base", start_angle = 0, direction = "counterclockwise", title = "Condition")
        mutHist.update_traces(opacity = 0.6)

        boundData = data.loc[data.D <= boundaryValue,]
        diffuData = data.loc[data.D > boundaryValue,]
        boundTData = boundData >> summarize(A1 = X.A1.sum() / summary_functions.n_distinct(X.filename),
                                            A2 = X.A2.sum() / summary_functions.n_distinct(X.filename),
                                            A3 = X.A3.sum() / summary_functions.n_distinct(X.filename),
                                            A4 = X.A4.sum() / summary_functions.n_distinct(X.filename),
                                            A5 = X.A5.sum() / summary_functions.n_distinct(X.filename),
                                            A6 = X.A6.sum() / summary_functions.n_distinct(X.filename),
                                            A7 = X.A7.sum() / summary_functions.n_distinct(X.filename),
                                            A8 = X.A8.sum() / summary_functions.n_distinct(X.filename),
                                            A9 = X.A9.sum() / summary_functions.n_distinct(X.filename),
                                            A10 = X.A10.sum() / summary_functions.n_distinct(X.filename),
                                            A11 = X.A11.sum() / summary_functions.n_distinct(X.filename),
                                            A12 = X.A12.sum() / summary_functions.n_distinct(X.filename),
                                            A13 = X.A13.sum() / summary_functions.n_distinct(X.filename),
                                            A14 = X.A14.sum() / summary_functions.n_distinct(X.filename),
                                            A15 = X.A15.sum() / summary_functions.n_distinct(X.filename),
                                            A16 = X.A16.sum() / summary_functions.n_distinct(X.filename),
                                            A17 = X.A17.sum() / summary_functions.n_distinct(X.filename),
                                            A18 = X.A18.sum() / summary_functions.n_distinct(X.filename)
                                           )
        diffuTData = diffuData >> summarize(A1 = X.A1.sum() / summary_functions.n_distinct(X.filename),
                                            A2 = X.A2.sum() / summary_functions.n_distinct(X.filename),
                                            A3 = X.A3.sum() / summary_functions.n_distinct(X.filename),
                                            A4 = X.A4.sum() / summary_functions.n_distinct(X.filename),
                                            A5 = X.A5.sum() / summary_functions.n_distinct(X.filename),
                                            A6 = X.A6.sum() / summary_functions.n_distinct(X.filename),
                                            A7 = X.A7.sum() / summary_functions.n_distinct(X.filename),
                                            A8 = X.A8.sum() / summary_functions.n_distinct(X.filename),
                                            A9 = X.A9.sum() / summary_functions.n_distinct(X.filename),
                                            A10 = X.A10.sum() / summary_functions.n_distinct(X.filename),
                                            A11 = X.A11.sum() / summary_functions.n_distinct(X.filename),
                                            A12 = X.A12.sum() / summary_functions.n_distinct(X.filename),
                                            A13 = X.A13.sum() / summary_functions.n_distinct(X.filename),
                                            A14 = X.A14.sum() / summary_functions.n_distinct(X.filename),
                                            A15 = X.A15.sum() / summary_functions.n_distinct(X.filename),
                                            A16 = X.A16.sum() / summary_functions.n_distinct(X.filename),
                                            A17 = X.A17.sum() / summary_functions.n_distinct(X.filename),
                                            A18 = X.A18.sum() / summary_functions.n_distinct(X.filename)
                                           )
        boundTData["State"] = "Bound"
        boundTData["Base"] = 0
        diffuTData["State"] = "Diffusing"
        diffuTData["Base"] = 0
        stateData = pd.concat([boundTData, diffuTData]) # boundData.append(diffuData)
        stateData = stateData.rename(columns = {"A1": "0 - 10",
                                                "A2": "10 - 20",
                                                "A3": "20 - 30",
                                                "A4": "30 - 40",
                                                "A5": "40 - 50",
                                                "A6": "50 - 60",
                                                "A7": "60 - 70",
                                                "A8": "70 - 80",
                                                "A9": "80 - 90",
                                                "A10": "90 - 100",
                                                "A11": "100 - 110",
                                                "A12": "110 - 120",
                                                "A13": "120 - 130",
                                                "A14": "130 - 140",
                                                "A15": "140 - 150",
                                                "A16": "150 - 160",
                                                "A17": "160 - 170",
                                                "A18": "170 - 180"})
        stateData.loc[:, 0:18] = stateData.iloc[:, 0:18].divide(stateData.sum(1, numeric_only = True), axis = 0)
        stateData = stateData.melt(id_vars = ["State", "Base"], var_name = "Theta", value_name = "Counts")
        stateHist = px.bar_polar(stateData, r = "Counts", theta = "Theta", color = "State", base = "Base", start_angle = 0, direction = "counterclockwise", title = "State")
        stateHist.update_traces(opacity = 0.6)

        boundData = boundData >> group_by(X.mutation) >> summarize(A1 = X.A1.sum() / summary_functions.n_distinct(X.filename),
                                                                   A2 = X.A2.sum() / summary_functions.n_distinct(X.filename),
                                                                   A3 = X.A3.sum() / summary_functions.n_distinct(X.filename),
                                                                   A4 = X.A4.sum() / summary_functions.n_distinct(X.filename),
                                                                   A5 = X.A5.sum() / summary_functions.n_distinct(X.filename),
                                                                   A6 = X.A6.sum() / summary_functions.n_distinct(X.filename),
                                                                   A7 = X.A7.sum() / summary_functions.n_distinct(X.filename),
                                                                   A8 = X.A8.sum() / summary_functions.n_distinct(X.filename),
                                                                   A9 = X.A9.sum() / summary_functions.n_distinct(X.filename),
                                                                   A10 = X.A10.sum() / summary_functions.n_distinct(X.filename),
                                                                   A11 = X.A11.sum() / summary_functions.n_distinct(X.filename),
                                                                   A12 = X.A12.sum() / summary_functions.n_distinct(X.filename),
                                                                   A13 = X.A13.sum() / summary_functions.n_distinct(X.filename),
                                                                   A14 = X.A14.sum() / summary_functions.n_distinct(X.filename),
                                                                   A15 = X.A15.sum() / summary_functions.n_distinct(X.filename),
                                                                   A16 = X.A16.sum() / summary_functions.n_distinct(X.filename),
                                                                   A17 = X.A17.sum() / summary_functions.n_distinct(X.filename),
                                                                   A18 = X.A18.sum() / summary_functions.n_distinct(X.filename)
                                                                  )
        diffuData = diffuData >> group_by(X.mutation) >> summarize(A1 = X.A1.sum() / summary_functions.n_distinct(X.filename),
                                                                   A2 = X.A2.sum() / summary_functions.n_distinct(X.filename),
                                                                   A3 = X.A3.sum() / summary_functions.n_distinct(X.filename),
                                                                   A4 = X.A4.sum() / summary_functions.n_distinct(X.filename),
                                                                   A5 = X.A5.sum() / summary_functions.n_distinct(X.filename),
                                                                   A6 = X.A6.sum() / summary_functions.n_distinct(X.filename),
                                                                   A7 = X.A7.sum() / summary_functions.n_distinct(X.filename),
                                                                   A8 = X.A8.sum() / summary_functions.n_distinct(X.filename),
                                                                   A9 = X.A9.sum() / summary_functions.n_distinct(X.filename),
                                                                   A10 = X.A10.sum() / summary_functions.n_distinct(X.filename),
                                                                   A11 = X.A11.sum() / summary_functions.n_distinct(X.filename),
                                                                   A12 = X.A12.sum() / summary_functions.n_distinct(X.filename),
                                                                   A13 = X.A13.sum() / summary_functions.n_distinct(X.filename),
                                                                   A14 = X.A14.sum() / summary_functions.n_distinct(X.filename),
                                                                   A15 = X.A15.sum() / summary_functions.n_distinct(X.filename),
                                                                   A16 = X.A16.sum() / summary_functions.n_distinct(X.filename),
                                                                   A17 = X.A17.sum() / summary_functions.n_distinct(X.filename),
                                                                   A18 = X.A18.sum() / summary_functions.n_distinct(X.filename)
                                                                  )
        boundData = boundData.rename(columns = {"A1": "0 - 10",
                                                "A2": "10 - 20",
                                                "A3": "20 - 30",
                                                "A4": "30 - 40",
                                                "A5": "40 - 50",
                                                "A6": "50 - 60",
                                                "A7": "60 - 70",
                                                "A8": "70 - 80",
                                                "A9": "80 - 90",
                                                "A10": "90 - 100",
                                                "A11": "100 - 110",
                                                "A12": "110 - 120",
                                                "A13": "120 - 130",
                                                "A14": "130 - 140",
                                                "A15": "140 - 150",
                                                "A16": "150 - 160",
                                                "A17": "160 - 170",
                                                "A18": "170 - 180"})
        boundData["Base"] = 0
        boundData.loc[:, 1:19] = boundData.iloc[:, 1:19].divide(boundData.sum(1, numeric_only = True), axis = 0)
        boundData = boundData.melt(id_vars = ["mutation", "Base"], var_name = "Theta", value_name = "Counts")
        boundHist = px.bar_polar(boundData, r = "Counts", theta = "Theta", color = "mutation", base = "Base", start_angle = 0, direction = "counterclockwise", title = "Bound")
        boundHist.update_traces(opacity = 0.6)

        diffuData = diffuData.rename(columns = {"A1": "0 - 10",
                                                "A2": "10 - 20",
                                                "A3": "20 - 30",
                                                "A4": "30 - 40",
                                                "A5": "40 - 50",
                                                "A6": "50 - 60",
                                                "A7": "60 - 70",
                                                "A8": "70 - 80",
                                                "A9": "80 - 90",
                                                "A10": "90 - 100",
                                                "A11": "100 - 110",
                                                "A12": "110 - 120",
                                                "A13": "120 - 130",
                                                "A14": "130 - 140",
                                                "A15": "140 - 150",
                                                "A16": "150 - 160",
                                                "A17": "160 - 170",
                                                "A18": "170 - 180"})
        diffuData["Base"] = 0
        diffuData.loc[:, 1:19] = diffuData.iloc[:, 1:19].divide(diffuData.sum(1, numeric_only = True), axis = 0)
        diffuData = diffuData.melt(id_vars = ["mutation", "Base"], var_name = "Theta", value_name = "Counts")
        diffuHist = px.bar_polar(diffuData, r = "Counts", theta = "Theta", color = "mutation", base = "Base", start_angle = 0, direction = "counterclockwise", title = "Diffusing")
        diffuHist.update_traces(opacity = 0.6)

        data["Ratio"] = np.log2((data["A1"] + data["A2"] + data["A3"]).div(data["A16"] + data["A17"] + data["A18"]))
        trendPlot = px.scatter(data, x = "D", y = "Ratio", color = "mutation", labels = {"Ratio": "Asymmetry Coefficient", "D": "Log10(D(um^2/s))"})
        trendPlot.update_traces(opacity = 0.6)

        dataSubset = data.loc[data.loc[:, 'D'] <= boundaryValue, ]
        forwardBound = dataSubset.loc[dataSubset.loc[:, "Ratio"] > 0,]
        backwardBound = dataSubset.loc[dataSubset.loc[:, "Ratio"] < 0,]
        dataSubset = data.loc[data.loc[:, 'D'] > boundaryValue, ]
        forwardDiffu = dataSubset.loc[dataSubset.loc[:, "Ratio"] > 0,]
        backwardDiffu = dataSubset.loc[dataSubset.loc[:, "Ratio"] < 0,]

        forwardBound["State"] = "Forward Bound"
        backwardBound["State"] = "Backward Bound"
        forwardDiffu["State"] = "Forward Diffusing"
        backwardDiffu["State"] = "Backward Diffusing"
        boxData = pd.concat([forwardBound, backwardBound, forwardDiffu, backwardDiffu])
        boxPlot = px.box(boxData, x = "State", y = "Ratio", color = "mutation", points = "all", labels = {"Ratio": "Asymmetry Ratio"}) 
        return mutHist, stateHist, boundHist, diffuHist, trendPlot, boxPlot

    def getJumpDistanceData(self, selectionFile):
        with sqlite3.connect('database.db') as conn:
            if len(selectionFile) > 1:
                data = pd.read_sql_query(f"select FileList.*, JDList.jump_distance, JDList.sharedFrequency, JDList.twoParFrequency, JDList.threeParFrequency, JDList.twoParD1Values, JDList.twoParD2Values, JDList.threeParD1Values, JDList.threeParD2Values, JDList.threeParD3Values from FileList INNER JOIN JDList ON FileList.filename = JDList.filename WHERE FileList.filename IN {tuple(selectionFile)}", conn)
            else:
                data = pd.read_sql_query(f"select FileList.*, JDList.jump_distance, JDList.sharedFrequency, JDList.twoParFrequency, JDList.threeParFrequency, JDList.twoParD1Values, JDList.twoParD2Values, JDList.threeParD1Values, JDList.threeParD2Values, JDList.threeParD3Values from FileList INNER JOIN JDList ON FileList.filename = JDList.filename WHERE FileList.filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
        return data

    def getJumpDistanceBoxData(self, selectionFile):
        with sqlite3.connect('database.db') as conn:
            if len(selectionFile) > 1:
                data = pd.read_sql_query(f"select FileList.* from FileList WHERE FileList.filename IN {tuple(selectionFile)}", conn)
            else:
                data = pd.read_sql_query(f"select FileList.* from FileList WHERE FileList.filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
        twoParBoxData = pd.DataFrame()
        threeParBoxData = pd.DataFrame()
        for n in range(len(data.index)):
            twoParTemp = pd.DataFrame({"filename": data["filename"][n],
                                       "mutation": data["mutation"][n],
                                       "fraction": ["Bound", "Diffusing"],
                                       "values": [data["twoParf1"][n], 1 - data["twoParf1"][n]]
                                       }
                                     )
            threeParTemp = pd.DataFrame({"filename": data["filename"][n],
                                         "mutation": data["mutation"][n],
                                         "fraction": ["Bound", "Mixed", "Diffusing"],
                                         "values": [data["threeParf1"][n], data["threeParf2"][n], 1 - data["threeParf1"][n] - data["threeParf2"][n]]
                                         }
                                       )
            twoParBoxData = pd.concat([twoParBoxData, twoParTemp], axis = 0)
            threeParBoxData = pd.concat([threeParBoxData, threeParTemp], axis = 0)
        return twoParBoxData, threeParBoxData

    def getAngleData(self, selectionFile):
        with sqlite3.connect('database.db') as conn:
            if len(selectionFile) > 1:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.trajID, TrajectoryList.D, TrajectoryList.traj_length, AngleList.A1, AngleList.A2, AngleList.A3, AngleList.A4, AngleList.A5, AngleList.A6, AngleList.A7, AngleList.A8, AngleList.A9, AngleList.A10, AngleList.A11, AngleList.A12, AngleList.A13, AngleList.A14, AngleList.A15, AngleList.A16, AngleList.A17, AngleList.A18 from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename INNER JOIN AngleList ON TrajectoryList.trajID = AngleList.trajID WHERE FileList.filename IN {tuple(selectionFile)}", conn)
            else:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.trajID, TrajectoryList.D, TrajectoryList.traj_length, AngleList.A1, AngleList.A2, AngleList.A3, AngleList.A4, AngleList.A5, AngleList.A6, AngleList.A7, AngleList.A8, AngleList.A9, AngleList.A10, AngleList.A11, AngleList.A12, AngleList.A13, AngleList.A14, AngleList.A15, AngleList.A16, AngleList.A17, AngleList.A18  from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename INNER JOIN AngleList ON TrajectoryList.trajID = AngleList.trajID WHERE FileList.filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
        return data

    def getDOTFiles(self, selectionFile):
        with sqlite3.connect('database.db') as conn:
            if len(selectionFile) > 1:
                # data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, FileList.exposure_time, TrajectoryList.traj_length, TrajectoryList.meanX, TrajectoryList.meanY, TrajectoryList.maxDistance, TrajectoryList.meanDistance, TrajectoryList.medianDistance from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename WHERE FileList.filename IN {tuple(selectionFile)} AND TrajectoryList.traj_length > :minTrajLength", conn, params = {"minTrajLength": minTrajLength})
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, FileList.exposure_time, TrajectoryList.traj_length, TrajectoryList.meanX, TrajectoryList.meanY, TrajectoryList.maxDistance, TrajectoryList.meanDistance, TrajectoryList.medianDistance from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename WHERE FileList.filename IN {tuple(selectionFile)}", conn)
            else:
                # data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, FileList.exposure_time, TrajectoryList.traj_length, TrajectoryList.meanX, TrajectoryList.meanY, TrajectoryList.maxDistance, TrajectoryList.meanDistance, TrajectoryList.medianDistance from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename WHERE FileList.filename = :selectionFile AND TrajectoryList.traj_length > :minTrajLength", conn, params = {"selectionFile": selectionFile[0], "minTrajLength": minTrajLength})
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, FileList.exposure_time, TrajectoryList.traj_length, TrajectoryList.meanX, TrajectoryList.meanY, TrajectoryList.maxDistance, TrajectoryList.meanDistance, TrajectoryList.medianDistance from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename WHERE FileList.filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]}) 
        # data = data.loc[data.exposure_time * data.traj_length > minTrajLength,]
        return data

    def getDOTData(self, data, dOTRegions, minTrajLength):
        coord = data[["meanX", "meanY"]].to_numpy().astype(float)
        temp = chull(coord) # TODO: Do for each cell or condition?
        boundaryRegionPoints = np.append(temp.vertices, temp.vertices[0]) # list(set(temp.vertices)) # list(set([i for row in temp.simplices for i in row]))
        boundaryRegion = np.array(coord[boundaryRegionPoints, :]) #outside most points
        centerPoint = np.mean(boundaryRegion, axis = 0)
        data = data.assign(traj_duration = data["exposure_time"] * data["traj_length"])

        # In theory, I could just replace the data and coord with the selectedData and selectedCoord, they're already used in defining the region
        selectedData = data.loc[data.traj_duration > minTrajLength,]
        selectedCoord = selectedData[["meanX", "meanY"]].to_numpy().astype(float)
        figure = px.scatter(selectedData, x = "meanX", y = "meanY", color = "traj_duration", size = "maxDistance", labels = {"meanX": "X (\u03BCm)", "meanY": "Y (\u03BCm)"})

        # First row of data
        newBoundaryRegion = ((boundaryRegion - centerPoint) * dOTRegions[0]) + centerPoint
        boundaryRegionDF = pd.DataFrame({"x": newBoundaryRegion[:, 0], "y": newBoundaryRegion[:, 1]})
        figure.add_trace(px.line(boundaryRegionDF, x = "x", y = "y").data[0])
        polygon = mpltPath.Path(newBoundaryRegion)
        pointsWithinPolygon = polygon.contains_points(selectedCoord)
        # Prepare dataframe
        plotData = selectedData.loc[pointsWithinPolygon]
        plotData["Region"] = [str(dOTRegions[0])] * len(plotData)
        for n in range(1, len(dOTRegions)):
            # Drawing the border
            nextBoundaryRegion = ((boundaryRegion - centerPoint) * dOTRegions[n]) + centerPoint
            nextBoundaryRegionDF = pd.DataFrame({"x": nextBoundaryRegion[:, 0], "y": nextBoundaryRegion[:, 1]})
            figure.add_trace(px.line(nextBoundaryRegionDF, x = "x", y = "y").data[0])
            # Computing the values
            nextPolygon = mpltPath.Path(nextBoundaryRegion)
            pointsWithinNextPolygon = nextPolygon.contains_points(selectedCoord)

            temp = selectedData.iloc[pointsWithinNextPolygon].merge(selectedData.iloc[pointsWithinPolygon], how = "outer", indicator = True).loc[lambda x : x['_merge']=='left_only']
            temp["Region"] = [str(dOTRegions[n])] * len(temp)
            plotData = pd.concat([plotData, temp], axis = 0)

            # Prepare for next loop
            pointsWithinPolygon = pointsWithinNextPolygon
        # Final row of table
        boundaryRegionDF = pd.DataFrame({"x": boundaryRegion[:, 0], "y": boundaryRegion[:, 1]})
        figure.add_trace(px.line(boundaryRegionDF, x = "x", y = "y").data[0])

        temp = selectedData.merge(selectedData.iloc[pointsWithinPolygon], how = "outer", indicator = True).loc[lambda x : x['_merge']=='left_only']
        temp["Region"] = ["1.0"] * len(temp)
        plotData = pd.concat([plotData, temp], axis = 0)

        tableData = plotData >> group_by(X.Region) >> summarize(number = summary_functions.n(X.traj_duration), max = X.traj_duration.max(), mean = X.traj_duration.mean(), median = X.traj_duration.median())
        boxData = (plotData >> group_by(X.filename, X.mutation, X.Region) >> summarize(max = X.traj_duration.max(), mean = X.traj_duration.mean(), median = X.traj_duration.median()))
        boxData = boxData.sort_values(by=['Region'])
        return tableData, boxData, figure

    def getHeatMapTraj(self, selectionFile, minTrajLength):
        with sqlite3.connect('database.db') as conn:
            if len(selectionFile) > 1:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.traj_length, TrajectoryList.startTime, TrajectoryList.endTime, TrajectoryList.meanX, TrajectoryList.meanY from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename WHERE FileList.filename IN {tuple(selectionFile)} AND TrajectoryList.traj_length > :minTrajLength", conn, params = {"minTrajLength": minTrajLength})
            else:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.traj_length, TrajectoryList.startTime, TrajectoryList.endTime, TrajectoryList.meanX, TrajectoryList.meanY from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename WHERE FileList.filename = :selectionFile AND TrajectoryList.traj_length > :minTrajLength", conn, params = {"selectionFile": selectionFile[0], "minTrajLength": minTrajLength})
        return data

    def getTrackFiles(self, selectionFile):
        with sqlite3.connect('database.db') as conn:
            if len(selectionFile) > 1:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.trajID, TrajectoryList.traj_length, TrackList.Frame, TrackList.x, TrackList.y from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename INNER JOIN TrackList ON TrajectoryList.trajID = TrackList.trajID WHERE FileList.filename IN {tuple(selectionFile)}", conn)
            else:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, TrajectoryList.trajID, TrajectoryList.traj_length, TrackList.Frame, TrackList.x, TrackList.y from FileList INNER JOIN TrajectoryList ON FileList.filename = TrajectoryList.filename INNER JOIN TrackList ON TrajectoryList.trajID = TrackList.trajID WHERE FileList.filename = :selectionFile", conn, params = {"selectionFile": selectionFile[0]})
        return data

    def plotTrajectory_data(self, trajLength, trajNumber, jumpNumber, data):
        mutations = set(data["mutation"])
        mutations = list(mutations)
        df = pd.DataFrame()
        for n in range(len(mutations)):
            dataSubset = data.loc[data["mutation"] == mutations[n]]
            files = set(dataSubset["filename"])
            files = list(files)
            for m in range(len(files)):
                dataSubSubset = dataSubset.loc[dataSubset["filename"] == files[m]]
                trajs = set(dataSubSubset["trajID"])
                trajs = list(trajs)
                if trajNumber > len(trajs):
                    trajToPlot = len(trajs)
                else:
                    trajToPlot = trajNumber
                for i in range(trajToPlot):
                    dataOfInterest = dataSubSubset.loc[dataSubSubset["trajID"] == trajs[i]]
                    if len(dataOfInterest) > trajLength:
                        if jumpNumber > len(dataOfInterest):
                            temp = dataOfInterest.head(len(dataOfInterest))
                        else:
                            temp = dataOfInterest.head(jumpNumber)
                        df = pd.concat([df, temp], axis = 0)
        return df

    def produceDiffusionData(self, data, binSize, lowerLimit, upperLimit, errorView, boundaryValue):
        mutations = set(data["mutation"])
        mutations = list(mutations)
        plotData = pd.DataFrame(data = [])
        data = data.loc[data.D > lowerLimit,]
        data = data.loc[data.D < upperLimit,]
        pieData = data >> group_by(X.mutation) >> summarize(BoundFraction = (X.D < boundaryValue).value_counts()[1] / summary_functions.n(X.D), UnboundFraction = (X.D >= boundaryValue).value_counts()[1] / summary_functions.n(X.D)) 
        for n in range(len(mutations)):
            dataSubset = data.loc[data["mutation"] == mutations[n]]
            files = set(dataSubset["filename"])
            files = list(files)
            errorData = pd.DataFrame(data = [])
            for m in range(len(files)):
                dataSubSubset = dataSubset.loc[dataSubset["filename"] == files[m]]
                a, binEdges = np.histogram(dataSubSubset['D'], bins = binSize, range = (lowerLimit, upperLimit))
                errorData = pd.concat([pd.DataFrame({f"{files[m]}": a / len(dataSubSubset['D'])}), errorData], axis = 1)
            a, binEdges = np.histogram(dataSubset['D'], bins = binSize, range = (lowerLimit, upperLimit))
            binCenters = 0.5 * (binEdges[:-1] + binEdges[1:])
            if errorView == 0: # data variation
                temp = pd.DataFrame({"x": binCenters, "y": a / len(dataSubset['D']), "error": errorData.max(axis = 1) - errorData.min(axis = 1), "error_minus": (a / len(dataSubset['D'])) - errorData.min(axis = 1), "mutation": [mutations[n]] * binSize})
            elif errorView == 1: # STD
                temp = pd.DataFrame({"x": binCenters, "y": a / len(dataSubset['D']), "error": errorData.std(axis = 1), "mutation": [mutations[n]] * binSize})
            else: # Standard Error of Mean
                temp = pd.DataFrame({"x": binCenters, "y": a / len(dataSubset['D']), "error": errorData.sem(axis = 1), "mutation": [mutations[n]] * binSize})
            plotData = pd.concat([plotData, temp], axis = 0)
        return plotData, pieData

    def produceDiffusionPieFigures(self, pieData):
        if pieData.shape[0] >= 3:
            pieToDraw = 3
        elif pieData.shape[0] < 3:
            pieToDraw = pieData.shape[0]
        pieFigure = [None] * 3
        for n in range(pieToDraw):
            figureData = pd.DataFrame({"Condition": ["Bound", "Unbound"], 
                                       "Fraction": [pieData.iloc[n, 1], 1.0 - pieData.iloc[n, 1]]
                                      }
                                     )
            pieFigure[n] = px.pie(figureData, values = "Fraction", names = "Condition", title = pieData["mutation"][n])
        return pieFigure

    def produceDwellTimeFigures(self, selectionFile):
        # filter out to only slow acquisition data first
        with sqlite3.connect('database.db') as conn: #TODO : Filter out filename with slow acquisition time first and then search the database with the slow acquisition filename only
            if len(selectionFile) > 1:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, DwellTimeData.R1, DwellTimeData.R2, DwellTimeData.F from FileList INNER JOIN DwellTimeData ON FileList.filename = DwellTimeData.filename WHERE FileList.filename IN {tuple(selectionFile)} AND acquisition_rate = 'slow'", conn)
            else:
                data = pd.read_sql_query(f"select FileList.filename, FileList.mutation, DwellTimeData.R1, DwellTimeData.R2, DwellTimeData.F from FileList INNER JOIN DwellTimeData ON FileList.filename = DwellTimeData.filename WHERE FileList.filename = :selectionFile AND acquisition_rate = 'slow'", conn, params = {"selectionFile": selectionFile[0]})
        if len(data.index) > 0:
            dwellData = pd.DataFrame()
            for n in range(len(data.index)):
                dwellDataTemp = pd.DataFrame({"filename": data["filename"][n],
                                            "mutation": data["mutation"][n],
                                            "R": ["Long", "Short"],
                                            "rTime": [data["R1"][n], data["R2"][n]],
                                            "Fraction": [data["F"][n], 1 - data["F"][n]]
                                           }
                                          )
                dwellData = pd.concat([dwellData, dwellDataTemp], axis = 0)
            boxFigure = px.box(dwellData, x = "R", y = "rTime", color = "mutation", points = "all", labels = {"R": "Dwell-Time Types", "rTime": "Dwell-Time (s)"})
            pieData = data >> group_by(X.mutation) >> summarize(Long = X.F.mean(), Short = (1 - X.F.mean())) 
            if pieData.shape[0] >= 3:
                pieToDraw = 3
            elif pieData.shape[0] < 3:
                pieToDraw = pieData.shape[0]
            pieFigure = [None] * 3
            for n in range(pieToDraw):
                figureData = pd.DataFrame({"Condition": ["Long", "Short"], 
                                           "Fraction": [pieData["Long"][n], pieData["Short"][n]]
                                          }
                                         )
                pieFigure[n] = px.pie(figureData, values = "Fraction", names = "Condition", title = pieData["mutation"][n])
            # pieFigure = px.pie(dwellData, values = "Fraction", names = "R") # TODO: Add more pie charts to show the different mutations and group the mutations together
        else:
            boxFigure = [None]
            pieFigure = [None]
        return boxFigure, pieFigure

        
class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)

        # Input logo images
        logoCI = QLabel(self)
        pixmap = QPixmap("Centenary_Institute_logo.png")
        logoCI.setPixmap(pixmap.scaled(200, 120))
        logoGIC = QLabel(self)
        pixmap = QPixmap("GIC.png")
        logoGIC.setPixmap(pixmap.scaled(200, 120))

        # Input Dashboard Title
        dashboardTitle = QLabel("Transcription Factor Analysis Dashboard")
        dashboardTitle.setFont(QFont("Arial", 24))

        self.create_tabs()
        self.createTopLeftGroupBox()
        self.createBottomLeftTabWidget()

        topLayout = QHBoxLayout()
        topLayout.addStretch()
        topLayout.addWidget(logoCI)
        topLayout.addWidget(dashboardTitle)
        topLayout.addWidget(logoGIC)
        topLayout.addStretch()

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 3) # y, x, y-span, x-span
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.tabs, 1, 1, 4, 3)
        mainLayout.addWidget(self.bottomLeftTabWidget, 3, 0)
        mainLayout.setRowStretch(1, 1) # stretching which row, by how much
        mainLayout.setRowStretch(2, 1)
        mainLayout.setRowStretch(3, 1)
        mainLayout.setRowStretch(4, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 2)
        mainLayout.setColumnStretch(2, 2)
        self.setLayout(mainLayout)

        self.setWindowTitle("Genome Imaging Centre Dashboard")
        self.changeStyle('Fusion')

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        QApplication.setPalette(QApplication.style().standardPalette())

    def closeEvent(self, event):
        with open('Notes.txt', 'w') as f:
            f.write(self.textEdit.toPlainText())

    def create_tabs(self):
        # Create tabs
        self.tabs = QTabWidget()
        self.home_tab = QWidget()
        self.trajectory_tab = QTabWidget()
        self.diffusionTabs = QTabWidget()
        self.trackAngleTab = QTabWidget()
        self.trajectoryCharacteristicsTabs = QTabWidget()
        self.distributionOfTrackTab = QTabWidget()
        self.heatMapTab = QTabWidget()
        self.dwellTab = QTabWidget()
        self.uploadTab = QWidget()

        # Add tabs to tabs
        self.tabs.addTab(self.home_tab, "&Home")
        self.tabs.addTab(self.trajectory_tab, "&Trajectory")
        self.tabs.addTab(self.diffusionTabs, "&Diffusion Plots")
        self.tabs.addTab(self.trackAngleTab, "&Angle Plots")
        ## TODO:
        #self.tabs.addTab(self.trajectoryCharacteristicsTabs, "Tra&jectory Characteristics")
        self.tabs.addTab(self.distributionOfTrackTab, "D&istribution of Tracks")
        self.tabs.addTab(self.heatMapTab, "&Heat Map")
        self.tabs.addTab(self.dwellTab, "Dwe&ll Time")
        self.tabs.addTab(self.uploadTab, "&Upload")

        # Create the first tab
        self.home_tab.layout = QGridLayout(self)
        self.home_tab.setLayout(self.home_tab.layout)

        welcomeText = QLabel()
        welcomeText.setText("Welcome to the Transcription Factor Analysis dashboard!")
        welcomeText.setFont(QFont("Arial", 20))
        aboutLabel = QLabel()
        aboutLabel.setText("About")
        aboutLabel.setFont(QFont("Arial", 16))
        aboutText = QLabel()
        aboutText.setText("This app is designed for the spatial and temporal analysis of fluorescently tagged transcription factors within cell nuclei.")
        aboutText.setFont(QFont("Arial", 12))
        aboutText.setWordWrap(True)
        gsLabel = QLabel()
        gsLabel.setText("Getting started")
        gsLabel.setFont(QFont("Arial", 16))
        gsText = QLabel()
        gsText.setText("Please head to the app's GitHub repository and download the PDF documentation to get started and analyse your data.")
        gsText.setFont(QFont("Arial", 12))
        gsText.setWordWrap(True)
        gicLabel = QLabel()
        gicLabel.setText("Genome Imaging Centre")
        gicLabel.setFont(QFont("Arial", 16))
        gicText = QLabel()
        gicText.setText("This dashboard was built by the Genome Imaging Centre, a Core Research Facility of the Centenary Institute.")
        gicText.setFont(QFont("Arial", 12))
        gicText.setWordWrap(True)
        citaLabel = QLabel()
        citaLabel.setText("Citation")
        citaLabel.setFont(QFont("Arial", 16))
        citaText = QLabel()
        citaText.setText("Previously published algorithms, analysis, and scripts that are utilized in the dashboard can be found below: \n" +
                          "\n" +
                          "Localization and Tracking: \n" +
                          "D. M. McSwiggen et al. (2019) Evidence for DNA-mediated nuclear compartmentalization distinct from phase separation. eLife. 8:e47098. \n" +
                          "A. Sergé et al. (2008) Dynamic multiple-target tracing to probe spatiotemporal cartography of cell membranes. Nature methods, 5(8):687. \n" +
                          "\n" +
                          "MSD-Based Diffusion Plot: \n" +
                          "J. Chen et al. (2014) Single-molecule dynamics of enhanceosome assembly in embryonic stem cells. Cell. 156(6):1274 - 1285. \n" +
                          "Jump Distance Plot: \n" +
                          "D. Mazza et al. (2013) Monitoring dynamic binding of chromatin proteins in vivo by single-molecule tracking. Methods Mol Biol. 1042:117-37. \n" +
                          "\n" +
                          "Angle Plots: \n" +
                          "I. Izeddin et. al. (2014), Single-molecule tracking in live cells reveals distinct target-search strategies of transcription factors in the nucleus. eLife. 3:e02230. \n" +
                          "\n" +
                          "Heat Map: \n" +
                          "J. O. Andrews et al. (2018) qSR: a quantitative super-resolution analysis tool reveals the cell-cycle dependent organization of RNA Polymerase I in live human cells. Sci Rep. 7424 (2018). +\n" +
                          "\n" +
                          "Dwell Time: \n" +
                          "A.J. McCann et al. (2021) A dominant-negative SOX18 mutant disrupts multiple regulatory layers essential to transcription factor activity. Nucleic Acids Res. 49(19):10931-10955." +
                          "Developed by Zhe Liu in Janelia Research Campus")
        citaText.setFont(QFont("Arial", 12))
        citaText.setWordWrap(True)

        self.home_tab.layout.addWidget(welcomeText)
        self.home_tab.layout.addWidget(aboutLabel)
        self.home_tab.layout.addWidget(aboutText)
        self.home_tab.layout.addWidget(gsLabel)
        self.home_tab.layout.addWidget(gsText)
        self.home_tab.layout.addWidget(gicLabel)
        self.home_tab.layout.addWidget(gicText)
        self.home_tab.layout.addWidget(citaLabel)
        self.home_tab.layout.addWidget(citaText)

        # Trajectory tab
        self.trajectoryPlotTab = QWidget()
        self.trajectoryDataTab = QWidget()

        self.trajectory_tab.addTab(self.trajectoryPlotTab, "&Plot")
        self.trajectory_tab.addTab(self.trajectoryDataTab, "Detai&ls")

        # Trajectory plot sub tab
        self.trajectoryPlotTab.layout = QGridLayout(self)
        self.trajectoryPlotTab.setLayout(self.trajectoryPlotTab.layout)

        self.trajectory_browser = QtWebEngineWidgets.QWebEngineView(self)

        trajNumber = QLabel()
        trajNumber.setText("Number of Trajectory:")
        self.trajNumberBox = QSpinBox()
        self.trajNumberBox.setMaximum(99999)
        self.trajNumberBox.setValue(100)
        self.trajNumberBox.setMaximumWidth(200)
        jumpNumberDraw = QLabel()
        jumpNumberDraw.setText("Jumps To Draw:")
        self.jumpNumberDrawBox = QSpinBox()
        self.jumpNumberDrawBox.setValue(5)
        minTrajLength = QLabel()
        minTrajLength.setText("Minimum Trajectory Length Considered:")
        self.minTrajLength = QSpinBox()
        self.minTrajLength.setValue(5)

        trajectoryGrouping = QLabel()
        trajectoryGrouping.setText("Group Trajectory By:")
        self.trajTabTrajGroupButton = QRadioButton("Trajectory")
        self.trajTabSpeedGroupButton = QRadioButton("Speed")
        self.trajTabTrajGroupButton.setChecked(True)

        self.trajectoryPlotTab.layout.addWidget(trajNumber, 0, 0, 1, 1)
        self.trajectoryPlotTab.layout.addWidget(self.trajNumberBox, 1, 0, 1, 1)
        self.trajectoryPlotTab.layout.addWidget(jumpNumberDraw, 2, 0, 1, 1)
        self.trajectoryPlotTab.layout.addWidget(self.jumpNumberDrawBox, 3, 0, 1, 1)
        self.trajectoryPlotTab.layout.addWidget(minTrajLength, 4, 0, 1, 1)
        self.trajectoryPlotTab.layout.addWidget(self.minTrajLength, 5, 0, 1, 1)
        # self.trajectory_tab.layout.addWidget(trajectoryGrouping, 6, 0)
        # self.trajectory_tab.layout.addWidget(self.trajTabTrajGroupButton, 7, 0)
        # self.trajectory_tab.layout.addWidget(self.trajTabSpeedGroupButton, 8, 0)
        self.trajectoryPlotTab.layout.addWidget(self.trajectory_browser, 0, 1, 10, 3)

        self.trajectoryPlotTab.layout.setColumnStretch(0, 5)
        self.trajectoryPlotTab.layout.setColumnStretch(1, 1)
        self.trajectoryPlotTab.layout.setColumnStretch(2, 2)

        # Trajectory data tab
        self.trajectoryDataTab.layout = QGridLayout(self)
        self.trajectoryDataTab.setLayout(self.trajectoryDataTab.layout)

        self.trajectoryNumberBox_browser = QtWebEngineWidgets.QWebEngineView(self)

        self.trajectoryDataTab.layout.addWidget(self.trajectoryNumberBox_browser, 0, 0, 1, 1)

        # Diffusion tab
        self.diffusionTrajectoryTab = QWidget()
        self.diffusionTrackTab = QWidget()

        self.diffusionTabs.addTab(self.diffusionTrajectoryTab, "&Trajectory")
        self.diffusionTabs.addTab(self.diffusionTrackTab, "Tra&ck")

        # Trajectory diffusion sub tab
        self.diffusionTrajectoryTab.layout = QGridLayout(self)
        self.diffusionTrajectoryTab.setLayout(self.diffusionTrajectoryTab.layout)

        self.diffusionTrajectoryTabLeftSideBar = QVBoxLayout(self.diffusionTrajectoryTab)
        self.diffusionTrajectoryTabPlots = QGridLayout(self.diffusionTrajectoryTab)

        self.diffusionTrajectoryTab.layout.addLayout(self.diffusionTrajectoryTabLeftSideBar, 0, 0, 1, 1)
        self.diffusionTrajectoryTab.layout.addLayout(self.diffusionTrajectoryTabPlots, 0, 1, 1, 5)

        diffusionBinSize = QLabel()
        diffusionBinSize.setText("Number of Bins:")
        self.diffusionBinSize = QSpinBox()
        self.diffusionBinSize.setValue(20)

        diffusionLowerLimit = QLabel()
        diffusionLowerLimit.setText("Lower Limit For Plot:")
        self.diffusionLowerLimit = QSpinBox()
        self.diffusionLowerLimit.setMinimum(-10)
        self.diffusionLowerLimit.setValue(-4)
        diffusionUpperLimit = QLabel()
        diffusionUpperLimit.setText("Upper Limit For Plot:")
        self.diffusionUpperLimit = QSpinBox()
        self.diffusionUpperLimit.setValue(2)

        boundaryValue = QLabel()
        boundaryValue.setText("Boundary Computation:")
        self.boundaryComputation = QComboBox()
        self.boundaryComputation.addItems(["Formula", "Raw Value"])
        self.boundaryRawValue = QDoubleSpinBox()
        self.boundaryRawValue.setMinimum(-99.99)
        self.boundaryRawValue.setValue(-0.5)

        diffusionErrorBar = QLabel()
        diffusionErrorBar.setText("Errorbar Type:")
        self.diffusionErrorVariation = QRadioButton("Data Variation")
        self.diffusionErrorSTD = QRadioButton("Standard Deviation")
        self.diffusionErrorSEM = QRadioButton("Standard Error of Mean")
        self.diffusionErrorSTD.setChecked(True)

        self.fractionExportButton = QPushButton("Boundary Export")

        self.diffusion_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.diffusionFraction1_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.diffusionFraction2_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.diffusionFraction3_browser = QtWebEngineWidgets.QWebEngineView(self)

        self.diffusionTrajectoryTabLeftSideBar.addWidget(diffusionBinSize)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(self.diffusionBinSize)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(diffusionLowerLimit)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(self.diffusionLowerLimit)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(diffusionUpperLimit)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(self.diffusionUpperLimit)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(boundaryValue)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(self.boundaryComputation)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(self.boundaryRawValue)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(diffusionErrorBar)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(self.diffusionErrorVariation)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(self.diffusionErrorSTD)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(self.diffusionErrorSEM)
        self.diffusionTrajectoryTabLeftSideBar.addWidget(self.fractionExportButton)
        self.diffusionTrajectoryTabPlots.addWidget(self.diffusion_browser, 0, 0, 3, 3)
        self.diffusionTrajectoryTabPlots.addWidget(self.diffusionFraction1_browser, 0, 4, 1, 1)
        self.diffusionTrajectoryTabPlots.addWidget(self.diffusionFraction2_browser, 1, 4, 1, 1)
        self.diffusionTrajectoryTabPlots.addWidget(self.diffusionFraction3_browser, 2, 4, 1, 1)

        # Track diffusion sub tab
        self.diffusionTrackTab.layout = QGridLayout(self)
        self.diffusionTrackTab.setLayout(self.diffusionTrackTab.layout)
        self.jumpDistanceToCSV = QPushButton("Export Data")

        twoParLabel = QLabel()
        twoParLabel.setText("2 Parameters Fit:")
        twoParLabel.setFont(QFont("Arial", 16))
        self.diffusionTrack2Par_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.diffusionTrack2ParBox_browser = QtWebEngineWidgets.QWebEngineView(self)
        twoParNLabel = QLabel()
        twoParNLabel.setText("n =")
        self.twoParNText = QLabel()
        twoParD1Label = QLabel()
        twoParD1Label.setText("D1 =")
        self.twoParD1Text = QLabel()
        twoParD2Label = QLabel()
        twoParD2Label.setText("D2 =")
        self.twoParD2Text = QLabel()
        twoParf1Label = QLabel()
        twoParf1Label.setText("f1 =")
        self.twoParf1Text = QLabel()
        twoParSSRLabel = QLabel()
        twoParSSRLabel.setText("SSR =")
        self.twoParSSRText = QLabel()
        emptyLabel = QLabel()

        threeParLabel = QLabel()
        threeParLabel.setText("3 Parameters Fit:")
        threeParLabel.setFont(QFont("Arial", 16))
        self.diffusionTrack3Par_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.diffusionTrack3ParBox_browser = QtWebEngineWidgets.QWebEngineView(self)
        threeParNLabel = QLabel()
        threeParNLabel.setText("n =")
        self.threeParNText = QLabel()
        threeParD1Label = QLabel()
        threeParD1Label.setText("D1 =")
        self.threeParD1Text = QLabel()
        threeParD2Label = QLabel()
        threeParD2Label.setText("D2 =")
        self.threeParD2Text = QLabel()
        threeParD3Label = QLabel()
        threeParD3Label.setText("D3 =")
        self.threeParD3Text = QLabel()
        threeParf1Label = QLabel()
        threeParf1Label.setText("f1 =")
        self.threeParf1Text = QLabel()
        threeParf2Label = QLabel()
        threeParf2Label.setText("f2 =")
        self.threeParf2Text = QLabel()
        threeParSSRLabel = QLabel()
        threeParSSRLabel.setText("SSR =")
        self.threeParSSRText = QLabel()

        jumpDistanceLabel = QLabel()
        jumpDistanceLabel.setText("Jump Distance To Plot:")
        self.jumpDistanceConsidered = QDoubleSpinBox()
        self.jumpDistanceConsidered.setValue(0.5)

        self.diffusionTrackTab.layout.addWidget(twoParLabel, 0, 0)
        self.diffusionTrackTab.layout.addWidget(self.jumpDistanceToCSV, 0, 2)
        self.diffusionTrackTab.layout.addWidget(self.diffusionTrack2Par_browser, 1, 0, 14, 2)
        self.diffusionTrackTab.layout.addWidget(self.diffusionTrack2ParBox_browser, 1, 2, 10, 4)
        self.diffusionTrackTab.layout.addWidget(twoParNLabel, 11, 2)
        self.diffusionTrackTab.layout.addWidget(self.twoParNText,11, 3)
        self.diffusionTrackTab.layout.addWidget(twoParD1Label, 12, 2)
        self.diffusionTrackTab.layout.addWidget(self.twoParD1Text, 12, 3)
        self.diffusionTrackTab.layout.addWidget(twoParD2Label, 13, 2)
        self.diffusionTrackTab.layout.addWidget(self.twoParD2Text, 13, 3)
        self.diffusionTrackTab.layout.addWidget(twoParf1Label, 12, 4)
        self.diffusionTrackTab.layout.addWidget(self.twoParf1Text, 12, 5)
        self.diffusionTrackTab.layout.addWidget(twoParSSRLabel, 13, 4)
        self.diffusionTrackTab.layout.addWidget(self.twoParSSRText, 13, 5)
        self.diffusionTrackTab.layout.addWidget(emptyLabel, 14, 2)
        self.diffusionTrackTab.layout.addWidget(emptyLabel, 14, 3)

        self.diffusionTrackTab.layout.addWidget(threeParLabel, 15, 0)
        self.diffusionTrackTab.layout.addWidget(self.diffusionTrack3Par_browser, 16, 0, 14, 2)
        self.diffusionTrackTab.layout.addWidget(self.diffusionTrack3ParBox_browser, 16, 2, 10, 4)
        self.diffusionTrackTab.layout.addWidget(threeParNLabel, 26, 2)
        self.diffusionTrackTab.layout.addWidget(self.threeParNText, 26, 3)
        self.diffusionTrackTab.layout.addWidget(threeParD1Label, 27, 2)
        self.diffusionTrackTab.layout.addWidget(self.threeParD1Text, 27, 3)
        self.diffusionTrackTab.layout.addWidget(threeParD2Label, 28, 2)
        self.diffusionTrackTab.layout.addWidget(self.threeParD2Text, 28, 3)
        self.diffusionTrackTab.layout.addWidget(threeParD3Label, 29, 2)
        self.diffusionTrackTab.layout.addWidget(self.threeParD3Text, 29, 3)
        self.diffusionTrackTab.layout.addWidget(threeParf1Label, 27, 4)
        self.diffusionTrackTab.layout.addWidget(self.threeParf1Text, 27, 5)
        self.diffusionTrackTab.layout.addWidget(threeParf2Label, 28, 4)
        self.diffusionTrackTab.layout.addWidget(self.threeParf2Text, 28, 5)
        self.diffusionTrackTab.layout.addWidget(threeParSSRLabel, 29, 4)
        self.diffusionTrackTab.layout.addWidget(self.threeParSSRText, 29, 5)

        self.diffusionTrackTab.layout.addWidget(jumpDistanceLabel, 30, 2, 1, 2)
        self.diffusionTrackTab.layout.addWidget(self.jumpDistanceConsidered, 30, 4, 1, 2)

        # Angle tab
        self.trackAngleTab.layout = QGridLayout(self)
        self.trackAngleTab.setLayout(self.trackAngleTab.layout)
        self.trackAngleMut_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.trackAngleState_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.trackAngleBound_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.trackAngleDiffu_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.trackAngleBox_browser = QtWebEngineWidgets.QWebEngineView(self)

        self.angleGroupBox = QGroupBox("Angle Parameters")

        angleSelectionText = QLabel()
        angleSelectionText.setText("Angles of Interest:")
        self.angleSelection = CheckableComboBox()
        angleRatioText = QLabel()
        angleRatioText.setText("Ratio:")
        self.angleRatio = QDoubleSpinBox()
        self.angleRatio.setMaximum(100)
        self.angleRatio.setMinimum(0)
        self.angleRatio.setValue(50)

        angleGroupBoxLayout = QVBoxLayout()
        angleGroupBoxLayout.addWidget(angleSelectionText)
        angleGroupBoxLayout.addWidget(self.angleSelection)
        angleGroupBoxLayout.addWidget(angleRatioText)
        angleGroupBoxLayout.addWidget(self.angleRatio)
        self.angleGroupBox.setLayout(angleGroupBoxLayout)  

        boundaryValueText = QLabel()
        boundaryValueText.setText("Boundary Computation:")
        self.boundaryValueAngle = QDoubleSpinBox()
        self.boundaryValueAngle.setMinimum(-99.99)
        self.boundaryValueAngle.setValue(-0.5)

        self.trackAngleTab.layout.addWidget(self.trackAngleMut_browser, 0, 0, 1, 1)
        self.trackAngleTab.layout.addWidget(self.trackAngleState_browser, 1, 0, 1, 1)
        self.trackAngleTab.layout.addWidget(self.trackAngleBound_browser, 0, 1, 1, 1)
        self.trackAngleTab.layout.addWidget(self.trackAngleDiffu_browser, 1, 1, 1, 1)
        self.trackAngleTab.layout.addWidget(self.trackAngleBox_browser, 0, 2, 1, 1)
        self.trackAngleTab.layout.addWidget(self.angleGroupBox, 1, 2, 1, 1)
        self.trackAngleTab.layout.addWidget(boundaryValueText, 2, 0, 1, 1)
        self.trackAngleTab.layout.addWidget(self.boundaryValueAngle, 2, 1, 1, 2)

        # Trajectory characteristics tab
        self.trajCharLifetimeTab = QWidget()
        self.trajCharAveDistanceTab = QWidget()
        self.trajCharTotDistanceTab = QWidget()

        self.trajectoryCharacteristicsTabs.addTab(self.trajCharLifetimeTab, "&Lifetime")
        self.trajectoryCharacteristicsTabs.addTab(self.trajCharAveDistanceTab, "A&verage Distance")
        self.trajectoryCharacteristicsTabs.addTab(self.trajCharTotDistanceTab, "&Total Distance")

        # Lifetime trajectory characterteristics sub tab
        self.trajCharLifetimeTab.layout = QGridLayout(self)
        self.trajCharLifetimeTab.setLayout(self.trajCharLifetimeTab.layout)

        # Average distance trajectory characterteristics sub tab
        self.trajCharAveDistanceTab.layout = QGridLayout(self)
        self.trajCharAveDistanceTab.setLayout(self.trajCharAveDistanceTab.layout)

        # Total distance trajectory characterteristics sub tab
        self.trajCharTotDistanceTab.layout = QGridLayout(self)
        self.trajCharTotDistanceTab.setLayout(self.trajCharTotDistanceTab.layout)

        # Distribution of tracks tab
        self.distributionOfTrackTab.layout = QGridLayout(self)
        self.distributionOfTrackTab.setLayout(self.distributionOfTrackTab.layout)

        self.dOTBoxPlotBrowser = QtWebEngineWidgets.QWebEngineView(self)
        dOTTrajChoice = QLabel()
        dOTTrajChoice.setText("Trajectory Length:")
        self.dOTTrajChoiceMax = QRadioButton("Max")
        self.dOTTrajChoiceMean = QRadioButton("Mean")
        self.dOTTrajChoiceMedian = QRadioButton("Median")
        self.dOTTrajChoiceMean.setChecked(True)
        self.dOTDataPointChoice = QPushButton("Show Data Points")
        self.dOTDataPointChoice.setCheckable(True)
        self.dOTDataPointChoice.setChecked(True)

        self.dOTMapBrowser = QtWebEngineWidgets.QWebEngineView(self)

        self.dOTTable = QTableWidget(self)

        dOTMinTrajLength = QLabel()
        dOTMinTrajLength.setText("Minimum Seconds of Trajectory Length:")
        self.dOTMinTrajLength = QDoubleSpinBox()
        self.dOTMinTrajLength.setMinimum(0.00)
        self.dOTMinTrajLength.setMaximum(999.99)
        self.dOTMinTrajLength.setValue(0)
        dOTRegionArea = QLabel()
        dOTRegionArea.setText("Regions Area:")
        self.dOTRegionArea = QLineEdit()
        self.dOTRegionArea.setText("0.4, 0.8")
        self.dOTButton = QPushButton("Update")

        self.distributionOfTrackTab.layout.addWidget(self.dOTMapBrowser, 0, 0, 6, 1)
        self.distributionOfTrackTab.layout.addWidget(self.dOTBoxPlotBrowser, 0, 1, 1, 2)
        self.distributionOfTrackTab.layout.addWidget(self.dOTTable, 1, 1, 5, 1)
        self.distributionOfTrackTab.layout.addWidget(dOTTrajChoice, 1, 2)
        self.distributionOfTrackTab.layout.addWidget(self.dOTTrajChoiceMax, 2, 2)
        self.distributionOfTrackTab.layout.addWidget(self.dOTTrajChoiceMean, 3, 2)
        self.distributionOfTrackTab.layout.addWidget(self.dOTTrajChoiceMedian, 4, 2)
        self.distributionOfTrackTab.layout.addWidget(self.dOTDataPointChoice, 5, 2)
        self.distributionOfTrackTab.layout.addWidget(dOTMinTrajLength, 6, 0)
        self.distributionOfTrackTab.layout.addWidget(self.dOTMinTrajLength, 7, 0)
        self.distributionOfTrackTab.layout.addWidget(dOTRegionArea, 6, 1)
        self.distributionOfTrackTab.layout.addWidget(self.dOTRegionArea, 7, 1)
        self.distributionOfTrackTab.layout.addWidget(self.dOTButton, 6, 2, 2, 1)
        self.distributionOfTrackTab.layout.setRowStretch(0, 5)
        self.distributionOfTrackTab.layout.setColumnStretch(0, 3)
        self.distributionOfTrackTab.layout.setColumnStretch(1, 3)
        self.distributionOfTrackTab.layout.setColumnStretch(2, 1)

        # Heat map tab
        self.heatMapTab.layout = QGridLayout(self)
        self.heatMapTab.setLayout(self.heatMapTab.layout)

        self.heatMapPlot = QtWebEngineWidgets.QWebEngineView(self)
        self.heatMapCummulativeTrajs = QtWebEngineWidgets.QWebEngineView(self)
        self.heatMapLiveTrajs = QtWebEngineWidgets.QWebEngineView(self)
        self.heatMapBurstLifetime = QtWebEngineWidgets.QWebEngineView(self)
        self.heatMapRipley = QtWebEngineWidgets.QWebEngineView(self)

        self.heatMapTab.layout.addWidget(self.heatMapPlot, 0, 0, 2, 1)
        self.heatMapTab.layout.addWidget(self.heatMapCummulativeTrajs, 0, 1)
        self.heatMapTab.layout.addWidget(self.heatMapLiveTrajs, 1, 1)
        self.heatMapTab.layout.addWidget(self.heatMapBurstLifetime, 2, 0)
        self.heatMapTab.layout.addWidget(self.heatMapRipley, 2, 1)

        # Dwell time tab
        self.dwellTab.layout = QGridLayout(self)
        self.dwellTab.setLayout(self.dwellTab.layout)

        self.dwellBox_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.dwellPie1_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.dwellPie2_browser = QtWebEngineWidgets.QWebEngineView(self)
        self.dwellPie3_browser = QtWebEngineWidgets.QWebEngineView(self)

        self.dwellTab.layout.addWidget(self.dwellBox_browser, 0, 0, 3, 2)
        self.dwellTab.layout.addWidget(self.dwellPie1_browser, 0, 2, 1, 1)
        self.dwellTab.layout.addWidget(self.dwellPie2_browser, 1, 2, 1, 1)
        self.dwellTab.layout.addWidget(self.dwellPie3_browser, 2, 2, 1, 1)

        # Upload tab
        self.uploadTab.layout = QGridLayout(self)
        self.uploadTab.setLayout(self.uploadTab.layout)

        acquisitionRate = QLabel()
        acquisitionRate.setText("Acquisition Rate:")
        acquisitionRate.setToolTip("Select whether the analysis will be carried out using a pre-defined number of tracks from the trajectory or a percentage of the total trajectory.")
        self.acquisitionRateFast = QRadioButton("Fast")
        self.acquisitionRateSlow = QRadioButton("Slow")
        self.acquisitionRateFast.setChecked(True)

        parallelization = QLabel()
        parallelization.setText("Make Use of Multi-Core/Parallelization:")
        parallelization.setToolTip("Make use of multi-cores architecture, needs to ensure you have the MATLAB multicore processing toolbox.")
        self.parallelization = QPushButton("Yes")
        self.parallelization.setCheckable(True)
        self.parallelization.setChecked(True)

        parallelizationCores = QLabel()
        parallelizationCores.setText("Number of Cores to Use:")
        parallelizationCores.setToolTip("If parallelization is on, define the number of cores (not thread) to use.")
        self.parallelizationCores = QSpinBox()
        self.parallelizationCores.setValue(16)
        # self.parallelization.clicked.connect(parallelizationCores.setDisabled)
        # self.parallelization.clicked.connect(self.parallelizationCores.setDisabled)

        bleachRate = QLabel()
        bleachRate.setText("Bleach Rate:")
        self.bleachRate = QDoubleSpinBox()
        self.bleachRate.setMinimum(0.00)
        self.bleachRate.setMaximum(100.00)
        self.bleachRate.setValue(0.00)

        self.createParametersGroupBox()
        self.createImagingGroupBox()
        self.createLocalizationGroupBox()
        self.createTrackingGroupBox()

        self.uploadFileButton = QPushButton("Upload &Files")
        self.uploadFileButton.setDefault(True)
        self.uploadPostFileButton = QPushButton("Upload Post-Processe&d Files")
        self.uploadPostFileButton.setDefault(True)

        self.uploadTab.layout.addWidget(acquisitionRate, 0, 0)
        self.uploadTab.layout.addWidget(self.acquisitionRateFast, 1, 0)
        self.uploadTab.layout.addWidget(self.acquisitionRateSlow, 2, 0)
        self.uploadTab.layout.addWidget(parallelization, 0, 1)
        self.uploadTab.layout.addWidget(self.parallelization, 1, 1)
        self.uploadTab.layout.addWidget(parallelizationCores, 0, 2)
        self.uploadTab.layout.addWidget(self.parallelizationCores, 1, 2)
        self.uploadTab.layout.addWidget(bleachRate, 2, 1)
        self.uploadTab.layout.addWidget(self.bleachRate, 2, 2)
        self.uploadTab.layout.addWidget(self.parametersGroupBox, 3, 0, 2, 1)
        self.uploadTab.layout.addWidget(self.localizationGroupBox, 3, 1, 2, 1)
        self.uploadTab.layout.addWidget(self.trackingGroupBox, 3, 2, 1, 1)
        self.uploadTab.layout.addWidget(self.imagingGroupBox, 4, 2, 1, 1)

        self.uploadTab.layout.addWidget(self.uploadFileButton, 5, 0, 1, 2)
        self.uploadTab.layout.addWidget(self.uploadPostFileButton, 5, 2, 1, 1)

    def createParametersGroupBox(self):
        self.parametersGroupBox = QGroupBox("Generic Parameters")
        self.parametersGroupBox.layout = QVBoxLayout()

        analysisType = QLabel()
        analysisType.setText("Analysis Type:")
        analysisType.setToolTip("Select whether the analysis will be carried out using a pre-defined number of tracks from the trajectory or a percentage of the total trajectory.")
        self.analysisTypePercentage = QRadioButton("Percentage")
        self.analysisTypeNumber = QRadioButton("Number")
        self.analysisTypeNumber.setChecked(True)

        clipFactor = QLabel()
        clipFactor.setText("Clip Factor:")
        clipFactor.setToolTip("Decide the number of tracks in a trajectory to be used for the 'number' analysis. Otherwise, choose the percentage of tracks to be used, with 1 being all of the tracks in a trajectory.")
        self.clipFactorBox = QSpinBox()
        self.clipFactorBox.setMaximum(100)
        self.clipFactorBox.setValue(4)

        trajectoryLength = QLabel()
        trajectoryLength.setText("Trajectory Length:")
        trajectoryLength.setToolTip("Length of trajectory to keep (trajectory appear with less than this number of frame will be discarded).")
        self.trajectoryLengthBox = QSpinBox()
        self.trajectoryLengthBox.setValue(7)

        minTrajectoryNumber = QLabel()
        minTrajectoryNumber.setText("Minimum Number of Trajectory:")
        minTrajectoryNumber.setToolTip("Minimum trajectories in a file to be accepted into the analysis.")
        self.minTrajectoryNumberBox = QSpinBox()
        self.minTrajectoryNumberBox.setMaximum(9999)
        self.minTrajectoryNumberBox.setValue(1000)

        tolerance = QLabel()
        tolerance.setText("Tolerance:")
        tolerance.setToolTip("Number of decimals to be kept during analysis.")
        self.toleranceBox = QSpinBox()
        self.toleranceBox.setValue(12)

        localizationError = QLabel()
        localizationError.setText("Localization Error:")
        localizationError.setToolTip("Localization Error: -6 <- 10^-6.")
        self.localizationErrorBox = QDoubleSpinBox()
        self.localizationErrorBox.setMinimum(-99.99)
        self.localizationErrorBox.setValue(-6.5)

        emissionWavelength = QLabel()
        emissionWavelength.setText("Emission Wavelength:")
        emissionWavelength.setToolTip("Wavelength in nm consider emission max and filter cutoff.")
        self.emissionWavelengthBox = QSpinBox()
        self.emissionWavelengthBox.setMaximum(9999)
        self.emissionWavelengthBox.setValue(580)

        exposureTime = QLabel()
        exposureTime.setText("Exposure Time:")
        exposureTime.setToolTip("Exposure time in milliseconds.")
        self.exposureTimeBox = QDoubleSpinBox()
        self.exposureTimeBox.setMaximum(1000)
        self.exposureTimeBox.setValue(20)

        deflationLoopsNumber = QLabel()
        deflationLoopsNumber.setText("Number of Deflation Loops:")
        deflationLoopsNumber.setToolTip("Generally keep this to 0 if you need deflation loops, you are imaging at too high a density.")
        self.deflationLoopsNumberBox = QSpinBox()
        self.deflationLoopsNumberBox.setValue(0)

        diffusionConstantMax = QLabel()
        diffusionConstantMax.setText("Maximum Expected Diffusion Constant::")
        diffusionConstantMax.setToolTip("The maximal expected diffusion constant caused by Brownian motion in um^2/s.")
        self.diffusionConstantMaxBox = QDoubleSpinBox()
        self.diffusionConstantMaxBox.setMaximum(10)
        self.diffusionConstantMaxBox.setMinimum(0)
        self.diffusionConstantMaxBox.setValue(3)

        gapsAllowed = QLabel()
        gapsAllowed.setText("Number of Gaps Allowed:")
        gapsAllowed.setToolTip("The number of gaps allowed in trajectories (1 being trajectories must exist in both frame n and frame n+1).")
        self.gapsAllowedBox = QSpinBox()
        self.gapsAllowedBox.setValue(1)

        self.parametersGroupBox.layout.addWidget(analysisType)
        self.parametersGroupBox.layout.addWidget(self.analysisTypePercentage)
        self.parametersGroupBox.layout.addWidget(self.analysisTypeNumber)
        self.parametersGroupBox.layout.addWidget(clipFactor)
        self.parametersGroupBox.layout.addWidget(self.clipFactorBox)
        self.parametersGroupBox.layout.addWidget(trajectoryLength)
        self.parametersGroupBox.layout.addWidget(self.trajectoryLengthBox)
        self.parametersGroupBox.layout.addWidget(minTrajectoryNumber)
        self.parametersGroupBox.layout.addWidget(self.minTrajectoryNumberBox)
        self.parametersGroupBox.layout.addWidget(tolerance)
        self.parametersGroupBox.layout.addWidget(self.toleranceBox)
        self.parametersGroupBox.layout.addWidget(localizationError)
        self.parametersGroupBox.layout.addWidget(self.localizationErrorBox)
        self.parametersGroupBox.layout.addWidget(emissionWavelength)
        self.parametersGroupBox.layout.addWidget(self.emissionWavelengthBox)
        self.parametersGroupBox.layout.addWidget(exposureTime)
        self.parametersGroupBox.layout.addWidget(self.exposureTimeBox)
        self.parametersGroupBox.layout.addWidget(deflationLoopsNumber)
        self.parametersGroupBox.layout.addWidget(self.deflationLoopsNumberBox)
        self.parametersGroupBox.layout.addWidget(diffusionConstantMax)
        self.parametersGroupBox.layout.addWidget(self.diffusionConstantMaxBox)
        self.parametersGroupBox.layout.addWidget(gapsAllowed)
        self.parametersGroupBox.layout.addWidget(self.gapsAllowedBox)
        self.parametersGroupBox.setLayout(self.parametersGroupBox.layout)

    def createImagingGroupBox(self):
        self.imagingGroupBox = QGroupBox("Imaging Parameters")
        self.imagingGroupBox.layout = QVBoxLayout()
        
        pixelSize = QLabel()
        pixelSize.setText("Pixel Size:")
        pixelSize.setToolTip("um per pixel.")
        self.pixelSize = QDoubleSpinBox()
        self.pixelSize.setValue(0.13)

        psfScaling = QLabel()
        psfScaling.setText("PSF Scaling:")
        psfScaling.setToolTip("PSF scaling.")
        self.psfScaling = QDoubleSpinBox()
        self.psfScaling.setValue(1.35)

        detectionObjectiveNA = QLabel()
        detectionObjectiveNA.setText("NA of Detection Objective:")
        detectionObjectiveNA.setToolTip("NA of detection objective.")
        self.detectionObjectiveNA = QDoubleSpinBox()
        self.detectionObjectiveNA.setValue(1.49)

        self.imagingGroupBox.layout.addWidget(pixelSize)
        self.imagingGroupBox.layout.addWidget(self.pixelSize)
        self.imagingGroupBox.layout.addWidget(psfScaling)
        self.imagingGroupBox.layout.addWidget(self.psfScaling)
        self.imagingGroupBox.layout.addWidget(detectionObjectiveNA)
        self.imagingGroupBox.layout.addWidget(self.detectionObjectiveNA)
        self.imagingGroupBox.setLayout(self.imagingGroupBox.layout)

    def createLocalizationGroupBox(self):
        self.localizationGroupBox = QGroupBox("Localization Parameters")
        self.localizationGroupBox.layout = QVBoxLayout()

        detectionBox = QLabel()
        detectionBox.setText("Detection Box:")
        detectionBox.setToolTip("In pixels.")
        self.detectionBox = QSpinBox()
        self.detectionBox.setValue(9)

        minIntensity = QLabel()
        minIntensity.setText("Minimum Intensity:")
        minIntensity.setToolTip("Minimum intensity in counts.")
        self.minIntensity = QSpinBox()

        maxIteration = QLabel()
        maxIteration.setText("Maximum Number of Iterations:")
        maxIteration.setToolTip("Maximum number of iterations.")
        self.maxIteration = QSpinBox()
        self.maxIteration.setValue(50)

        terminationTolerance = QLabel()
        terminationTolerance.setText("Termination Tolerance:")
        terminationTolerance.setToolTip("Termination tolerance.")
        self.terminationTolerance = QSpinBox()
        self.terminationTolerance.setMinimum(-99)
        self.terminationTolerance.setValue(-2)

        self.radiusTolerance = QPushButton("Radius Tolerance")
        self.radiusTolerance.setCheckable(True)
        radiusTolerance = QLabel()
        radiusTolerance.setText("Radius Tolerance:")
        radiusTolerance.setToolTip("Radius tolerance in percent.")
        radiusTolerance.setEnabled(False)
        self.radiusToleranceValue = QSpinBox()
        self.radiusToleranceValue.setValue(50)
        self.radiusToleranceValue.setEnabled(False)
        positionTolerance = QLabel()
        positionTolerance.setText("Position Tolerance:")
        positionTolerance.setToolTip("Maximum position refinement.")
        positionTolerance.setEnabled(False)
        self.positionTolerance = QDoubleSpinBox()
        self.positionTolerance.setValue(1.5)
        self.positionTolerance.setEnabled(False)
        self.radiusTolerance.toggled.connect(radiusTolerance.setEnabled)
        self.radiusTolerance.toggled.connect(self.radiusToleranceValue.setEnabled)
        self.radiusTolerance.toggled.connect(positionTolerance.setEnabled)
        self.radiusTolerance.toggled.connect(self.positionTolerance.setEnabled)

        self.threshLocPrec = QPushButton("Thresh Loc Prec")
        self.threshLocPrec.setCheckable(True)
        minLoc = QLabel()
        minLoc.setText("Minimum Loc:")
        minLoc.setToolTip("Minimum Loc.")
        minLoc.setEnabled(False)
        self.minLoc = QSpinBox()
        self.minLoc.setEnabled(False)
        maxLoc = QLabel()
        maxLoc.setText("Maximum Loc:")
        maxLoc.setToolTip("Maximum Loc, leave zero for infinity.")
        maxLoc.setEnabled(False)
        self.maxLoc = QSpinBox()
        self.maxLoc.setEnabled(False)
        self.threshLocPrec.toggled.connect(minLoc.setEnabled)
        self.threshLocPrec.toggled.connect(self.minLoc.setEnabled)
        self.threshLocPrec.toggled.connect(maxLoc.setEnabled)
        self.threshLocPrec.toggled.connect(self.maxLoc.setEnabled)

        self.threshSNR = QPushButton("Thresh SNR")
        self.threshSNR.setCheckable(True)
        minSNR = QLabel()
        minSNR.setText("Minimum SNR:")
        minSNR.setToolTip("Minimum SNR.")
        minSNR.setEnabled(False)
        self.minSNR = QSpinBox()
        self.minSNR.setEnabled(False)
        maxSNRIter = QLabel()
        maxSNRIter.setText("Max Number of Iterations for Thresh SNR:")
        maxSNRIter.setToolTip("Maximum SNR, leave zero for infinity.")
        maxSNRIter.setEnabled(False)
        self.maxSNRIter = QSpinBox()
        self.maxSNRIter.setEnabled(False)
        self.threshSNR.toggled.connect(minSNR.setEnabled)
        self.threshSNR.toggled.connect(self.minSNR.setEnabled)
        self.threshSNR.toggled.connect(maxSNRIter.setEnabled)
        self.threshSNR.toggled.connect(self.maxSNRIter.setEnabled)

        self.threshDensity = QPushButton("Thresh Density")
        self.threshDensity.setCheckable(True)

        self.localizationGroupBox.layout.addWidget(detectionBox)
        self.localizationGroupBox.layout.addWidget(self.detectionBox)
        self.localizationGroupBox.layout.addWidget(minIntensity)
        self.localizationGroupBox.layout.addWidget(self.minIntensity)
        self.localizationGroupBox.layout.addWidget(maxIteration)
        self.localizationGroupBox.layout.addWidget(self.maxIteration)
        self.localizationGroupBox.layout.addWidget(terminationTolerance)
        self.localizationGroupBox.layout.addWidget(self.terminationTolerance)
        self.localizationGroupBox.layout.addWidget(self.radiusTolerance)
        self.localizationGroupBox.layout.addWidget(radiusTolerance)
        self.localizationGroupBox.layout.addWidget(self.radiusToleranceValue)
        self.localizationGroupBox.layout.addWidget(positionTolerance)
        self.localizationGroupBox.layout.addWidget(self.positionTolerance)
        self.localizationGroupBox.layout.addWidget(self.threshLocPrec)
        self.localizationGroupBox.layout.addWidget(minLoc)
        self.localizationGroupBox.layout.addWidget(self.minLoc)
        self.localizationGroupBox.layout.addWidget(maxLoc)
        self.localizationGroupBox.layout.addWidget(self.maxLoc)
        self.localizationGroupBox.layout.addWidget(self.threshSNR)
        self.localizationGroupBox.layout.addWidget(minSNR)
        self.localizationGroupBox.layout.addWidget(self.minSNR)
        self.localizationGroupBox.layout.addWidget(maxSNRIter)
        self.localizationGroupBox.layout.addWidget(self.maxSNRIter)
        self.localizationGroupBox.layout.addWidget(self.threshDensity)
        self.localizationGroupBox.setLayout(self.localizationGroupBox.layout)

    def createTrackingGroupBox(self):
        self.trackingGroupBox = QGroupBox("Tracking Parameters")
        self.trackingGroupBox.layout = QVBoxLayout()

        trackStart = QLabel()
        trackStart.setText("Track Start:")
        trackStart.setToolTip("Track start.")
        self.trackStart = QSpinBox()
        self.trackStart.setValue(1)

        trackEnd = QLabel()
        trackEnd.setText("Track End:")
        trackEnd.setToolTip("Track end, leave zero for infinity.")
        self.trackEnd = QSpinBox()

        exponentialFactorSearch = QLabel()
        exponentialFactorSearch.setText("Search Exponential Factor:")
        exponentialFactorSearch.setToolTip("Search exponential factor.")
        self.exponentialFactorSearch = QDoubleSpinBox()
        self.exponentialFactorSearch.setValue(1.2)

        statWin = QLabel()
        statWin.setText("Stat Win:")
        statWin.setToolTip("Stat win.")
        self.statWin = QSpinBox()
        self.statWin.setValue(10)

        compMax = QLabel()
        compMax.setText("Maximum Comp:")
        compMax.setToolTip("Maximum comp.")
        self.compMax = QSpinBox()
        self.compMax.setValue(5)

        intLawWeight = QLabel()
        intLawWeight.setText("Int Law Weight:")
        intLawWeight.setToolTip("Int law weight.")
        self.intLawWeight = QDoubleSpinBox()
        self.intLawWeight.setValue(0.9)

        difLawWeight = QLabel()
        difLawWeight.setText("Diff Law Weight:")
        difLawWeight.setToolTip("Diff law weight.")
        self.difLawWeight = QDoubleSpinBox()
        self.difLawWeight.setValue(0.5)

        self.trackingGroupBox.layout.addWidget(trackStart)
        self.trackingGroupBox.layout.addWidget(self.trackStart)
        self.trackingGroupBox.layout.addWidget(trackEnd)
        self.trackingGroupBox.layout.addWidget(self.trackEnd)
        self.trackingGroupBox.layout.addWidget(exponentialFactorSearch)
        self.trackingGroupBox.layout.addWidget(self.exponentialFactorSearch)
        self.trackingGroupBox.layout.addWidget(statWin)
        self.trackingGroupBox.layout.addWidget(self.statWin)
        self.trackingGroupBox.layout.addWidget(compMax)
        self.trackingGroupBox.layout.addWidget(self.compMax)
        self.trackingGroupBox.layout.addWidget(intLawWeight)
        self.trackingGroupBox.layout.addWidget(self.intLawWeight)
        self.trackingGroupBox.layout.addWidget(difLawWeight)
        self.trackingGroupBox.layout.addWidget(self.difLawWeight)
        self.trackingGroupBox.setLayout(self.trackingGroupBox.layout)

    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Group 1")

        acquisitionRateSelection = QLabel()
        acquisitionRateSelection.setText("Acquisition Rate:")
        self.comboAcquisitionRate = CheckableComboBox()
        mutationSelection = QLabel()
        mutationSelection.setText("Mutation(s):")
        self.comboMutation = CheckableComboBox()
        fileSelection = QLabel()
        fileSelection.setText("File(s):")
        self.comboFileList = CheckableComboBox()
        self.deleteFile = QPushButton("Delete Selected Files")

        layout = QVBoxLayout()
        layout.addWidget(acquisitionRateSelection)
        layout.addWidget(self.comboAcquisitionRate)
        layout.addWidget(mutationSelection)
        layout.addWidget(self.comboMutation)
        layout.addWidget(fileSelection)
        layout.addWidget(self.comboFileList)
        layout.addStretch(1)
        layout.addWidget(self.deleteFile)
        self.topLeftGroupBox.setLayout(layout)    

    def createBottomLeftTabWidget(self):
        self.bottomLeftTabWidget = QTabWidget()
        self.bottomLeftTabWidget.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Ignored)

        tab1 = QWidget()
        tableWidget = QTableWidget(10, 10)

        tab1hbox = QHBoxLayout()
        tab1hbox.setContentsMargins(5, 5, 5, 5)
        tab1hbox.addWidget(tableWidget)
        tab1.setLayout(tab1hbox)

        tab2 = QWidget()
        self.textEdit = QTextEdit()

        if os.path.exists("Notes.txt"):
            with open('Notes.txt') as f:
                lines = f.readlines()
                if lines != []:
                    self.textEdit.setPlainText("".join(lines))
        else:
            with open('Notes.txt', 'w') as f:
                f.write("")       

        tab2hbox = QHBoxLayout()
        tab2hbox.setContentsMargins(5, 5, 5, 5)
        tab2hbox.addWidget(self.textEdit)
        tab2.setLayout(tab2hbox)

        self.bottomLeftTabWidget.addTab(tab1, "&Table")
        self.bottomLeftTabWidget.addTab(tab2, "Text &Edit")

if __name__ == '__main__':
    app = QApplication([])
    plotly_app = PlotlyApplication()
    gallery = WidgetGallery()
    plotly_app.init_handler(gallery)
    gallery.show()
    controller = Controller(model=Model(), view=gallery)
    sys.exit(app.exec())