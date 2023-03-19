from description.mu import *
from description.func import *

#===========================================================НАЧАЛО РАБОТЫ=================================================================================================
class Start(QMainWindow):

    def __init__(self):
        super(Start, self).__init__()
        uic.loadUi('description\\ui\\start.ui', self)
        self.show()

        self.cwd = os.getcwd()
        global cwdGLOBAL
        cwdGLOBAL = self.cwd

        self.pb_close.clicked.connect(self.close)
        self.pb_startEvent.clicked.connect(self.startEvent)

    def startEvent(self):
        if self.rb_VK.isChecked():
            global a
            a = Autorisation()
        elif self.rb_meta.isChecked():
            st = time()
            metadata(self.cwd, cwd_host, tempPrint)
            os.chdir(cwdGLOBAL)
            t = time() - st

            msg = QMessageBox()
            msg.setWindowTitle("Результат")
            msg.setText(f"Успешно!\nЗанятое время работы - {t} секунд.")
            msg.setIcon(QMessageBox.Information)
            buttonAceptar = msg.addButton("Открыть полученный каталог", QMessageBox.YesRole)
            buttonCancelar = msg.addButton("Закрыть", QMessageBox.RejectRole)
            msg.setDefaultButton(buttonAceptar)
            msg.exec_()
            if msg.clickedButton() == buttonAceptar:
                os.startfile(self.cwd + "\\OutputReport\\FakePhotos")

class Autorisation (QWidget):
    def __init__(self):
        super(Autorisation, self).__init__()
        uic.loadUi('description\\ui\\autorisation.ui', self)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.le_pass.setEchoMode(QLineEdit.Password)
        self.show()




        self.pb_close.clicked.connect(self.close)
        self.pb_autorisation.clicked.connect(self.auto)


    def auto(self):
        conn = sqlite3.connect(cwd_host)
        cursor = conn.cursor()
        query_str = f"SELECT token FROM adminZGT  WHERE login = \"{self.le_login.text()}\" AND pas = \"{self.le_pass.text()}\";"
        cursor.execute(query_str)
        records = cursor.fetchall()
        if len(records) != 0:
            global i
            global token
            token = records[0][0]
            i = InputLink()
            self.close()
        else:
            msg = QMessageBox()
            msg.setWindowTitle("Внимание")
            msg.setText("Неверный логин и/или пароль!\nПовторите попытку.")
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        conn.close()

class InputLink(QWidget):
    def __init__(self):
        super(InputLink, self).__init__()
        uic.loadUi('description\\ui\\inputLink.ui', self)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.show()

        self.cwd = os.getcwd()

        self.pb_search.clicked.connect(self.search)




    def search(self):
        link = self.le_link.text()
        st = time()
        VK_API(self.cwd, file_cfg, file_weights, token, link)
        t = time() - st
        os.chdir(cwdGLOBAL)
        msg = QMessageBox()
        msg.setWindowTitle("Результат")
        msg.setText(f"Успешно!\nЗанятое время работы - {t} секунд.")
        msg.setIcon(QMessageBox.Information)
        buttonAceptar = msg.addButton("Открыть полученный каталог", QMessageBox.YesRole)
        buttonCancelar = msg.addButton("Закрыть", QMessageBox.RejectRole)
        msg.setDefaultButton(buttonAceptar)
        msg.exec_()
        if msg.clickedButton() == buttonAceptar:
            os.startfile(self.cwd + "\\OutputReport\\FakeEvent")


if __name__ == '__main__':
    app = QApplication([])
    windows = Start()
    app.exec_()